#!/usr/bin/env python3
# dpo_train_with_val.py
# Resume or train DPO with a proper validation file (dpo_val.json).
# - Uses robust dataset conversion
# - Computes training-mode log-probs (grad enabled)
# - Evaluates on /scratch/aowais2/data/dpo_val.json and checkpoints best adapter

import os
import math
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, get_peft_model, LoraConfig
from torch.utils.data import DataLoader
from typing import Any, List, Dict

# === USER CONFIG ===
BASE_MODEL = "/scratch/aowais2/llama"
RESUME_ADAPTER = "/scratch/aowais2/llama32_lora_dpo_1epoch"  # set to "" to reinit
SFT_ADAPTER_PATH = "/scratch/aowais2/llama32_lora_sft"       # adapter used as starting policy if needed
DPO_TRAIN_JSON = "/scratch/aowais2/data/dpo_train.json"
DPO_VAL_JSON = "/scratch/aowais2/data/dpo_val.json"          # <--- validation file you requested
OUTPUT_DIR = "/scratch/aowais2/dpo_resume_checkpoints"
BATCH = 8
PROMPT_MAX = 256
ANSWER_MAX = 512
MAX_TOTAL = PROMPT_MAX + ANSWER_MAX
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 1e-5
EPOCHS = 3
SAVE_EVERY_STEPS = 500
EVAL_EVERY_STEPS = 200
BETA = 0.1
USE_LENGTH_NORMALIZED = True   # set False to use summed log-prob
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === HELPERS ===
def safe_tokenizer(preferred: str, fallback: str):
    if os.path.isdir(preferred):
        return AutoTokenizer.from_pretrained(preferred, local_files_only=True)
    return AutoTokenizer.from_pretrained(fallback, local_files_only=True)

def ensure_text(field: Any, tokenizer) -> str:
    if isinstance(field, str):
        return field
    if isinstance(field, (list, tuple)):
        try:
            return tokenizer.decode([int(x) for x in field], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        except Exception:
            return " ".join(map(str, field))
    return str(field)

def to_list_of_dicts(slice_ds):
    if slice_ds is None:
        return []
    if isinstance(slice_ds, dict):
        return [slice_ds]
    try:
        lst = list(slice_ds)
    except Exception:
        return []
    if len(lst) == 0:
        return []
    if isinstance(lst[0], dict):
        return lst
    try:
        keys = slice_ds.column_names
        return [{k: slice_ds[k][i] for k in keys} for i in range(len(slice_ds))]
    except Exception:
        return [{"value": x} for x in lst]

def collate_fn(batch: List[Dict]):
    prompts = [ensure_text(ex["prompt"], tokenizer) for ex in batch]
    chosens = [ensure_text(ex["chosen"], tokenizer) for ex in batch]
    rejects = [ensure_text(ex["rejected"], tokenizer) for ex in batch]
    enc_c = tokenizer([p + "\n" + c for p,c in zip(prompts, chosens)],
                      truncation=True, padding="longest", max_length=MAX_TOTAL, return_tensors="pt")
    enc_r = tokenizer([p + "\n" + r for p,r in zip(prompts, rejects)],
                      truncation=True, padding="longest", max_length=MAX_TOTAL, return_tensors="pt")
    prompt_enc = tokenizer(prompts, truncation=True, padding="longest", max_length=PROMPT_MAX, return_tensors="pt")
    prompt_lens = (prompt_enc["input_ids"] != tokenizer.pad_token_id).sum(dim=1)
    labels_c = enc_c["input_ids"].clone()
    labels_r = enc_r["input_ids"].clone()
    for i, plen in enumerate(prompt_lens):
        labels_c[i, :plen] = -100
        labels_r[i, :plen] = -100
    return {
        "in_c": enc_c["input_ids"], "attn_c": enc_c["attention_mask"], "labels_c": labels_c,
        "in_r": enc_r["input_ids"], "attn_r": enc_r["attention_mask"], "labels_r": labels_r,
    }

def summed_and_avg_logprob_train(input_ids, attention_mask, labels, model):
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    labels = labels.to(DEVICE)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    safe_labels = labels.clone()
    safe_labels[safe_labels == -100] = 0
    token_logps = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    valid_mask = (labels != -100).to(token_logps.dtype)
    token_logps = token_logps * valid_mask
    per_sample_sum = token_logps.sum(dim=1)
    token_counts = valid_mask.sum(dim=1).clamp(min=1)
    per_sample_avg = per_sample_sum / token_counts
    return per_sample_sum, per_sample_avg, token_counts

def summed_and_avg_logprob_eval(input_ids, attention_mask, labels, model):
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    labels = labels.to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        safe_labels = labels.clone()
        safe_labels[safe_labels == -100] = 0
        token_logps = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        valid_mask = (labels != -100).to(token_logps.dtype)
        token_logps = token_logps * valid_mask
        per_sample_sum = token_logps.sum(dim=1).cpu()
        token_counts = valid_mask.sum(dim=1).cpu().clamp(min=1)
        per_sample_avg = per_sample_sum / token_counts
    return per_sample_sum, per_sample_avg, token_counts

# === Load tokenizer and models ===
tokenizer = safe_tokenizer(RESUME_ADAPTER if RESUME_ADAPTER else SFT_ADAPTER_PATH, BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, local_files_only=True)
base.to(DEVICE)

# Load or initialize adapter
if RESUME_ADAPTER and os.path.isdir(RESUME_ADAPTER):
    policy = PeftModel.from_pretrained(base, RESUME_ADAPTER, local_files_only=True)
    print("Resumed adapter from", RESUME_ADAPTER)
else:
    # initialize a LoRA adapter if no resume adapter
    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj","v_proj","k_proj","o_proj"], bias="none", task_type="CAUSAL_LM")
    policy = get_peft_model(base, lora_config)
    print("Initialized new adapter")
policy.to(DEVICE)

# Freeze base and enable adapter params
for n,p in policy.named_parameters():
    if "lora" in n.lower() or "adapter" in n.lower():
        p.requires_grad = True
    else:
        p.requires_grad = False

trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
print("Trainable params:", trainable)

# Reference model (base-only)
ref = AutoModelForCausalLM.from_pretrained(BASE_MODEL, local_files_only=True).to(DEVICE)
ref.eval()

# === Load datasets and DataLoaders ===
train_ds = load_dataset("json", data_files=DPO_TRAIN_JSON)["train"]
val_ds = load_dataset("json", data_files=DPO_VAL_JSON)["train"]
print("Train examples:", len(train_ds), "Val examples:", len(val_ds))

def dataset_to_list(dataset):
    out = []
    for i in range(len(dataset)):
        ex = dataset[i]
        if isinstance(ex, dict):
            out.append(ex)
        else:
            try:
                keys = dataset.column_names
                out.append({k: dataset[k][i] for k in keys})
            except Exception:
                continue
    return out

train_list = dataset_to_list(train_ds)
val_list = dataset_to_list(val_ds)
print("Converted train/val to lists:", len(train_list), len(val_list))

train_loader = DataLoader(train_list, batch_size=BATCH, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_list, batch_size=BATCH, shuffle=False, collate_fn=collate_fn)

optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=LR)

# === Training loop with validation and checkpointing ===
best_val_acc = -1.0
patience = 0
global_step = 0

for epoch in range(EPOCHS):
    policy.train()
    for batch in train_loader:
        global_step += 1
        in_c = batch["in_c"]; attn_c = batch["attn_c"]; labels_c = batch["labels_c"]
        in_r = batch["in_r"]; attn_r = batch["attn_r"]; labels_r = batch["labels_r"]

        sum_c, avg_c, cnt_c = summed_and_avg_logprob_train(in_c, attn_c, labels_c, policy)
        sum_r, avg_r, cnt_r = summed_and_avg_logprob_train(in_r, attn_r, labels_r, policy)

        reward = (avg_c - avg_r) if USE_LENGTH_NORMALIZED else (sum_c - sum_r)
        loss = -torch.log(torch.sigmoid(BETA * reward)).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], GRAD_CLIP)
        optimizer.step()

        if global_step % EVAL_EVERY_STEPS == 0:
            # quick training-batch diagnostics
            with torch.no_grad():
                pref_rate = ( (sum_c - sum_r) > 0 ).float().mean().item()
                print(f"step {global_step} epoch {epoch+1} loss {loss.item():.4e} pref_rate {pref_rate:.3f}")

        if global_step % SAVE_EVERY_STEPS == 0:
            ckpt = os.path.join(OUTPUT_DIR, f"adapter_step_{global_step}")
            policy.save_pretrained(ckpt)
            print("Saved checkpoint", ckpt)

    # End-of-epoch validation
    policy.eval()
    total = 0; correct = 0; ties = 0; margins = []
    with torch.no_grad():
        for batch in val_loader:
            in_c = batch["in_c"]; attn_c = batch["attn_c"]; labels_c = batch["labels_c"]
            in_r = batch["in_r"]; attn_r = batch["attn_r"]; labels_r = batch["labels_r"]
            sum_c_e, avg_c_e, cnt_c_e = summed_and_avg_logprob_eval(in_c, attn_c, labels_c, policy)
            sum_r_e, avg_r_e, cnt_r_e = summed_and_avg_logprob_eval(in_r, attn_r, labels_r, policy)
            margin = (avg_c_e - avg_r_e) if USE_LENGTH_NORMALIZED else (sum_c_e - sum_r_e)
            for m in margin.cpu().numpy():
                total += 1
                margins.append(float(m))
                if m > 0:
                    correct += 1
                elif abs(m) <= 1e-12:
                    ties += 1
    val_acc = correct / total if total else 0.0
    val_tie = ties / total if total else 0.0
    val_mean_margin = sum(margins)/len(margins) if margins else 0.0
    print(f"Epoch {epoch+1} VAL acc {val_acc:.4f} tie {val_tie:.4f} mean_margin {val_mean_margin:.4f}")

    # checkpoint by best validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience = 0
        best_ckpt = os.path.join(OUTPUT_DIR, f"best_adapter_epoch_{epoch+1}")
        policy.save_pretrained(best_ckpt)
        print("Saved best adapter to", best_ckpt)
    else:
        patience += 1
        if patience >= EARLY_STOP_PATIENCE:
            print("Early stopping triggered.")
            break

print("Training finished. Best val acc:", best_val_acc)

