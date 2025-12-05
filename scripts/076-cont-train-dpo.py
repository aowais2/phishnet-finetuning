#!/usr/bin/env python3
# 076-cont-train-dpo.py
# Patched: robust tokenizer loading, automatic latest-checkpoint detection,
# safe resume of adapter weights, optional optimizer/scheduler/meta restore.

import os
import re
import json
import math
import time
import signal
from typing import Any, List, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from peft import PeftModel, get_peft_model, LoraConfig

# === USER CONFIG ===
BASE_MODEL = "/scratch/aowais2/llama"
# Either set RESUME_ADAPTER_DIR to a specific checkpoint folder (recommended),
# or set PARENT_CHECKPOINT_DIR to the directory that contains adapter_step_* folders.
RESUME_ADAPTER_DIR = "/scratch/aowais2/dpo_resume_checkpoints/adapter_step_5500"
PARENT_CHECKPOINT_DIR = "/scratch/aowais2/dpo_resume_checkpoints"  # used if RESUME_ADAPTER_DIR is empty

SFT_ADAPTER_PATH = "/scratch/aowais2/llama32_lora_sft"
TRAIN_JSON = "/scratch/aowais2/data/dpo_train.json"
VAL_JSON = "/scratch/aowais2/data/dpo_val.json"
OUTPUT_DIR = "/scratch/aowais2/llama32_dpo"
LOG_FILE = os.path.join(OUTPUT_DIR, "train.log")

BATCH = 8
PROMPT_MAX = 256
ANSWER_MAX = 512
MAX_TOTAL = PROMPT_MAX + ANSWER_MAX
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 3
LR = 1e-5
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 500
GRAD_CLIP = 1.0
BETA = 0.1
USE_LENGTH_NORMALIZED = True
USE_AMP = True
EVAL_EVERY_STEPS = 200
VAL_SAMPLE_SIZE = 200
SAVE_EVERY_STEPS = 500
EARLY_STOP_PATIENCE = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Utilities ===
def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def find_latest_checkpoint(parent_dir: str, pattern: str = r"adapter_step_(\d+)$"):
    if not parent_dir or not os.path.isdir(parent_dir):
        return None
    best = None
    best_step = -1
    for name in os.listdir(parent_dir):
        full = os.path.join(parent_dir, name)
        if not os.path.isdir(full):
            continue
        m = re.search(pattern, name)
        if not m:
            continue
        step = int(m.group(1))
        if step > best_step:
            best_step = step
            best = full
    return best

def safe_tokenizer(base_model_dir: str, adapter_dir: str = None):
    """
    Load tokenizer robustly:
      1) Try base_model_dir first.
      2) If that fails and adapter_dir contains tokenizer files, try adapter_dir.
      3) Try base_model_dir with use_fast=False as fallback.
    """
    # Try base model dir first
    try:
        return AutoTokenizer.from_pretrained(base_model_dir, local_files_only=True)
    except Exception as e_base:
        base_err = e_base

    # If adapter dir has tokenizer files, try it
    if adapter_dir:
        candidates = ["tokenizer.json", "tokenizer_config.json", "spiece.model", "sentencepiece.model", "vocab.json"]
        has_tokenizer_files = any(os.path.isfile(os.path.join(adapter_dir, f)) for f in candidates)
        if has_tokenizer_files:
            try:
                return AutoTokenizer.from_pretrained(adapter_dir, local_files_only=True)
            except Exception:
                pass

    # Fallback: try base model with use_fast=False
    try:
        return AutoTokenizer.from_pretrained(base_model_dir, local_files_only=True, use_fast=False)
    except Exception:
        raise base_err

def ensure_text(field: Any, tokenizer) -> str:
    if isinstance(field, str):
        return field
    if isinstance(field, (list, tuple)):
        try:
            return tokenizer.decode([int(x) for x in field], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        except Exception:
            return " ".join(map(str, field))
    return str(field)

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

def collate_fn(batch: List[Dict], tokenizer):
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
# Determine resume adapter dir: explicit or auto-detect latest
if not RESUME_ADAPTER_DIR:
    latest = find_latest_checkpoint(PARENT_CHECKPOINT_DIR)
    if latest:
        RESUME_ADAPTER_DIR = latest
        log(f"Auto-detected latest checkpoint: {RESUME_ADAPTER_DIR}")
    else:
        log("No resume adapter specified and no checkpoints found in parent; starting fresh or using SFT adapter path.")

log("Loading tokenizer and models")
tokenizer = safe_tokenizer(BASE_MODEL, RESUME_ADAPTER_DIR if RESUME_ADAPTER_DIR else SFT_ADAPTER_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, local_files_only=True)
base.to(DEVICE)

# Load adapter (resume) or initialize new
if RESUME_ADAPTER_DIR and os.path.isdir(RESUME_ADAPTER_DIR):
    policy = PeftModel.from_pretrained(base, RESUME_ADAPTER_DIR, local_files_only=True)
    log(f"Resumed adapter from {RESUME_ADAPTER_DIR}")
else:
    # initialize LoRA adapter as fallback
    lora_config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj","v_proj","k_proj","o_proj"], bias="none", task_type="CAUSAL_LM")
    policy = get_peft_model(base, lora_config)
    log("Initialized new adapter (no resume checkpoint found)")
policy.to(DEVICE)

# Freeze base and enable adapter params
for n,p in policy.named_parameters():
    if "lora" in n.lower() or "adapter" in n.lower():
        p.requires_grad = True
    else:
        p.requires_grad = False

trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
log(f"Trainable params: {trainable:,}")

# Reference model
ref = AutoModelForCausalLM.from_pretrained(BASE_MODEL, local_files_only=True).to(DEVICE)
ref.eval()

# === Load datasets ===
log("Loading datasets")
train_ds = load_dataset("json", data_files=TRAIN_JSON)["train"]
val_ds = load_dataset("json", data_files=VAL_JSON)["train"]
log(f"Train examples: {len(train_ds)} Val examples: {len(val_ds)}")

train_list = dataset_to_list(train_ds)
val_list = dataset_to_list(val_ds)
log(f"Converted train/val to lists: {len(train_list)} {len(val_list)}")

train_loader = DataLoader(train_list, batch_size=BATCH, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer))
val_loader_full = DataLoader(val_list, batch_size=BATCH, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer))

# Precompute quick validation sample
val_sample = val_list[:VAL_SAMPLE_SIZE]
val_sample_collated = [collate_fn(val_sample[i:i+BATCH], tokenizer) for i in range(0, len(val_sample), BATCH)]

# === Optimizer and scheduler ===
optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=LR, weight_decay=WEIGHT_DECAY)
steps_per_epoch = math.ceil(len(train_list) / BATCH)
total_steps = steps_per_epoch * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

# Try to load optimizer/scheduler/meta if present in resume dir
global_step = 0
start_epoch = 1
if RESUME_ADAPTER_DIR and os.path.isdir(RESUME_ADAPTER_DIR):
    opt_path = os.path.join(RESUME_ADAPTER_DIR, "optimizer.pt")
    sched_path = os.path.join(RESUME_ADAPTER_DIR, "scheduler.pt")
    meta_path = os.path.join(RESUME_ADAPTER_DIR, "meta.json")
    if os.path.isfile(opt_path):
        try:
            optimizer.load_state_dict(torch.load(opt_path, map_location=DEVICE))
            log("Loaded optimizer state from resume checkpoint")
        except Exception as e:
            log(f"Could not load optimizer state: {e}")
    if os.path.isfile(sched_path):
        try:
            scheduler.load_state_dict(torch.load(sched_path, map_location=DEVICE))
            log("Loaded scheduler state from resume checkpoint")
        except Exception as e:
            log(f"Could not load scheduler state: {e}")
    if os.path.isfile(meta_path):
        try:
            meta = json.load(open(meta_path))
            global_step = int(meta.get("global_step", 0))
            start_epoch = int(meta.get("epoch", 1))
            log(f"Resuming from meta: global_step={global_step}, start_epoch={start_epoch}")
        except Exception as e:
            log(f"Could not load meta.json: {e}")
    else:
        # try to infer global_step from folder name like adapter_step_5500
        m = re.search(r"adapter_step_(\d+)", RESUME_ADAPTER_DIR)
        if m:
            global_step = int(m.group(1))
            log(f"Inferred global_step={global_step} from folder name")

# AMP scaler
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP and DEVICE.startswith("cuda"))

# safe shutdown
stop_requested = False
def _signal_handler(sig, frame):
    global stop_requested
    log(f"Signal {sig} received. Will save checkpoint and stop after current step.")
    stop_requested = True
signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# === Training loop ===
best_val_acc = -1.0
patience = 0

log(f"Starting training from epoch {start_epoch} for {EPOCHS} epochs, {steps_per_epoch} steps/epoch (global_step starts at {global_step})")

for epoch in range(start_epoch, EPOCHS + 1):
    policy.train()
    epoch_start = time.time()
    for batch in train_loader:
        global_step += 1
        in_c = batch["in_c"]; attn_c = batch["attn_c"]; labels_c = batch["labels_c"]
        in_r = batch["in_r"]; attn_r = batch["attn_r"]; labels_r = batch["labels_r"]

        optimizer.zero_grad()
        if USE_AMP and DEVICE.startswith("cuda"):
            with torch.cuda.amp.autocast():
                sum_c, avg_c, cnt_c = summed_and_avg_logprob_train(in_c, attn_c, labels_c, policy)
                sum_r, avg_r, cnt_r = summed_and_avg_logprob_train(in_r, attn_r, labels_r, policy)
                reward = (avg_c - avg_r) if USE_LENGTH_NORMALIZED else (sum_c - sum_r)
                loss = -torch.log(torch.sigmoid(BETA * reward)).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            sum_c, avg_c, cnt_c = summed_and_avg_logprob_train(in_c, attn_c, labels_c, policy)
            sum_r, avg_r, cnt_r = summed_and_avg_logprob_train(in_r, attn_r, labels_r, policy)
            reward = (avg_c - avg_r) if USE_LENGTH_NORMALIZED else (sum_c - sum_r)
            loss = -torch.log(torch.sigmoid(BETA * reward)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad], GRAD_CLIP)
            optimizer.step()

        scheduler.step()

        if global_step % EVAL_EVERY_STEPS == 0:
            with torch.no_grad():
                pref_rate = float(((sum_c - sum_r) > 0).float().mean().item())
                avg_margin = float(((avg_c - avg_r).mean().item()))
                log(f"step {global_step} epoch {epoch} loss {loss.item():.4e} pref_rate {pref_rate:.3f} avg_margin {avg_margin:.4f}")

        if global_step % EVAL_EVERY_STEPS == 0:
            policy.eval()
            total = 0; correct = 0; ties = 0; margins = []
            with torch.no_grad():
                for vb in val_sample_collated:
                    sc, ac, cc = summed_and_avg_logprob_eval(vb["in_c"], vb["attn_c"], vb["labels_c"], policy)
                    sr, ar, cr = summed_and_avg_logprob_eval(vb["in_r"], vb["attn_r"], vb["labels_r"], policy)
                    margin = (ac - ar) if USE_LENGTH_NORMALIZED else (sc - sr)
                    for m in margin.cpu().numpy():
                        total += 1
                        margins.append(float(m))
                        if m > 0:
                            correct += 1
                        elif abs(m) <= 1e-12:
                            ties += 1
            quick_acc = correct / total if total else 0.0
            quick_tie = ties / total if total else 0.0
            log(f"Quick VAL step {global_step} acc {quick_acc:.4f} tie {quick_tie:.4f} mean_margin {sum(margins)/len(margins):.4f}")
            policy.train()

        if global_step % SAVE_EVERY_STEPS == 0:
            ckpt_dir = os.path.join(OUTPUT_DIR, f"adapter_step_{global_step}")
            policy.save_pretrained(ckpt_dir)
            torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(ckpt_dir, "scheduler.pt"))
            with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
                json.dump({"global_step": global_step, "epoch": epoch}, f)
            log(f"Saved checkpoint {ckpt_dir}")

        if stop_requested:
            log("Stop requested. Saving interrupted checkpoint and exiting.")
            final_ckpt = os.path.join(OUTPUT_DIR, f"adapter_interrupted_step_{global_step}")
            policy.save_pretrained(final_ckpt)
            torch.save(optimizer.state_dict(), os.path.join(final_ckpt, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(final_ckpt, "scheduler.pt"))
            with open(os.path.join(final_ckpt, "meta.json"), "w") as f:
                json.dump({"global_step": global_step, "epoch": epoch}, f)
            log(f"Saved interrupted checkpoint {final_ckpt}")
            raise SystemExit("Training interrupted by signal")

    # end epoch: full validation
    policy.eval()
    total = 0; correct = 0; ties = 0; margins = []
    with torch.no_grad():
        for batch in val_loader_full:
            sc, ac, cc = summed_and_avg_logprob_eval(batch["in_c"], batch["attn_c"], batch["labels_c"], policy)
            sr, ar, cr = summed_and_avg_logprob_eval(batch["in_r"], batch["attn_r"], batch["labels_r"], policy)
            margin = (ac - ar) if USE_LENGTH_NORMALIZED else (sc - sr)
            for m in margin.cpu().numpy():
                total += 1
                margins.append(float(m))
                if m > 0:
                    correct += 1
                elif abs(m) <= 1e-12:
                    ties += 1
    val_acc = correct / total if total else 0.0
    val_tie = ties / total if total else 0.0
    val_mean_margin = sum(margins) / len(margins) if margins else 0.0
    log(f"Epoch {epoch} FULL VAL acc {val_acc:.4f} tie {val_tie:.4f} mean_margin {val_mean_margin:.4f}")

    # checkpoint by validation
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience = 0
        best_ckpt = os.path.join(OUTPUT_DIR, f"best_adapter_epoch_{epoch}_step_{global_step}")
        policy.save_pretrained(best_ckpt)
        torch.save(optimizer.state_dict(), os.path.join(best_ckpt, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(best_ckpt, "scheduler.pt"))
        with open(os.path.join(best_ckpt, "meta.json"), "w") as f:
            json.dump({"global_step": global_step, "epoch": epoch}, f)
        log(f"Saved best adapter to {best_ckpt}")
    else:
        patience += 1
        log(f"No improvement on val acc. patience {patience}/{EARLY_STOP_PATIENCE}")
        if patience >= EARLY_STOP_PATIENCE:
            log("Early stopping triggered. Ending training.")
            break

    epoch_time = time.time() - epoch_start
    log(f"Epoch {epoch} finished in {epoch_time/60:.2f} minutes")

# final save
final_dir = os.path.join(OUTPUT_DIR, f"adapter_final_step_{global_step}")
policy.save_pretrained(final_dir)
torch.save(optimizer.state_dict(), os.path.join(final_dir, "optimizer.pt"))
torch.save(scheduler.state_dict(), os.path.join(final_dir, "scheduler.pt"))
with open(os.path.join(final_dir, "meta.json"), "w") as f:
    json.dump({"global_step": global_step, "epoch": epoch}, f)
log(f"Training complete. Final adapter saved to {final_dir}")

