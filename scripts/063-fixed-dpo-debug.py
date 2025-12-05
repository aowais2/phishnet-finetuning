#!/usr/bin/env python3
# dpo_debug_short_run.py
import os
from typing import List, Dict, Any
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import DPOTrainer, DPOConfig
from torch.utils.data import DataLoader

# === User config: adjust these paths and sizes ===
BASE_PATH = "/scratch/aowais2/llama"                 # base model path
SFT_ADAPTER_PATH = "/scratch/aowais2/llama32_lora_sft"
DPO_OUT = "/scratch/aowais2/llama32_lora_dpo"

DPO_TRAIN_JSON = "/scratch/aowais2/data/dpo_train.json"
DPO_VAL_JSON   = "/scratch/aowais2/data/dpo_val.json"

PROMPT_MAX = 256
ANSWER_MAX = 512

# Short-run hyperparams for debugging
DEBUG_EPOCHS = 1
DEBUG_LR = 5e-6        # smaller LR for stability
DEBUG_BETA = 0.05      # smaller beta to reduce extreme updates
DEBUG_BATCH = 2        # small batch for quick runs
DEBUG_ACCUM = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# === Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
PAD_ID = tokenizer.pad_token_id

# === Helper tokenization used for preprocessing (standalone, deterministic) ===
def tok(text: str, max_len: int):
    return tokenizer(
        text=str(text),
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )

def apply_simple_template(prompt: str, answer: str) -> str:
    return f"{prompt}\n{answer}"

# === Load models ===
print("Loading base model and attaching SFT adapter (local files only)...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_PATH, local_files_only=True, device_map="auto")
policy_model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH, local_files_only=True)
policy_model.to(DEVICE)
policy_model.eval()

# Load reference model as base-only (no adapter attached)
print("Loading reference model (base-only)...")
ref_model = AutoModelForCausalLM.from_pretrained(BASE_PATH, local_files_only=True, device_map="auto")
ref_model.to(DEVICE)
ref_model.eval()

# === Ensure adapter params are trainable ===
# Freeze everything then enable LoRA/adapter-like params by name heuristics
for n, p in policy_model.named_parameters():
    p.requires_grad = False

enabled = 0
enabled_names = []
for n, p in policy_model.named_parameters():
    lname = n.lower()
    if any(k in lname for k in ("lora", "adapter", "lora_a", "lora_b", "lora_")):
        p.requires_grad = True
        enabled += p.numel()
        enabled_names.append(n)

trainable_total = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
print(f"Enabled approx {enabled:,} adapter params; trainable param total: {trainable_total:,}")
print("Sample enabled param names (first 40):", enabled_names[:40])

# === Load raw datasets ===
print("Loading datasets...")
train_raw = load_dataset("json", data_files=DPO_TRAIN_JSON)["train"]
valid_raw = load_dataset("json", data_files=DPO_VAL_JSON)["train"]

# === Preprocess/tokenize once and store fixed-length numeric fields + raw text ===
def preprocess(example):
    chosen_text = apply_simple_template(example["prompt"], example["chosen"])
    rejected_text = apply_simple_template(example["prompt"], example["rejected"])

    p = tok(example["prompt"], PROMPT_MAX)
    c = tok(chosen_text, ANSWER_MAX)
    r = tok(rejected_text, ANSWER_MAX)

    return {
        "prompt_input_ids": p["input_ids"][0].tolist(),
        "prompt_attention_mask": p["attention_mask"][0].tolist(),
        "chosen_input_ids": c["input_ids"][0].tolist(),
        "chosen_attention_mask": c["attention_mask"][0].tolist(),
        "rejected_input_ids": r["input_ids"][0].tolist(),
        "rejected_attention_mask": r["attention_mask"][0].tolist(),
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }

print("Tokenizing train dataset (deterministic)...")
train_tok = train_raw.map(preprocess, batched=False)
print("Tokenizing eval dataset (deterministic)...")
valid_tok = valid_raw.map(preprocess, batched=False)

KEEP = {
    "prompt_input_ids", "prompt_attention_mask",
    "chosen_input_ids", "chosen_attention_mask",
    "rejected_input_ids", "rejected_attention_mask",
    "prompt", "chosen", "rejected",
}
train_tok = train_tok.remove_columns([c for c in train_tok.column_names if c not in KEEP])
valid_tok = valid_tok.remove_columns([c for c in valid_tok.column_names if c not in KEEP])

print("Train size:", len(train_tok), "Eval size:", len(valid_tok))

# === Deterministic collator that enforces fixed lengths ===
def enforce_fixed_length(seq: List[int], target_len: int, pad_id: int) -> List[int]:
    if len(seq) >= target_len:
        return seq[:target_len]
    return seq + [pad_id] * (target_len - len(seq))

def dpo_fixed_collator(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    prompts = [enforce_fixed_length(f["prompt_input_ids"], PROMPT_MAX, PAD_ID) for f in features]
    out["prompt_input_ids"] = torch.tensor(prompts, dtype=torch.long)
    prompt_masks = [[0 if token == PAD_ID else 1 for token in p] for p in prompts]
    out["prompt_attention_mask"] = torch.tensor(prompt_masks, dtype=torch.long)

    chosens = [enforce_fixed_length(f["chosen_input_ids"], ANSWER_MAX, PAD_ID) for f in features]
    out["chosen_input_ids"] = torch.tensor(chosens, dtype=torch.long)
    chosen_masks = [[0 if token == PAD_ID else 1 for token in c] for c in chosens]
    out["chosen_attention_mask"] = torch.tensor(chosen_masks, dtype=torch.long)

    rejects = [enforce_fixed_length(f["rejected_input_ids"], ANSWER_MAX, PAD_ID) for f in features]
    out["rejected_input_ids"] = torch.tensor(rejects, dtype=torch.long)
    rejected_masks = [[0 if token == PAD_ID else 1 for token in r] for r in rejects]
    out["rejected_attention_mask"] = torch.tensor(rejected_masks, dtype=torch.long)

    return out

# === Quick evaluation helpers (small-batch diagnostics) ===
def tokenize_pairs(prompts, answers):
    texts = [p + "\n" + a for p, a in zip(prompts, answers)]
    enc = tokenizer(
        texts,
        truncation=True,
        padding="longest",
        max_length=PROMPT_MAX + ANSWER_MAX,
        return_tensors="pt",
    )
    prompt_enc = tokenizer(prompts, truncation=True, padding="longest", max_length=PROMPT_MAX, return_tensors="pt")
    prompt_lens = (prompt_enc["input_ids"] != PAD_ID).sum(dim=1)
    labels = enc["input_ids"].clone()
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100
    return enc["input_ids"], enc["attention_mask"], labels

@torch.no_grad()
def compute_logprobs(input_ids, attention_mask, labels, model):
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
    per_sample_logp = token_logps.sum(dim=1)
    return per_sample_logp.cpu()

# Sample a tiny eval batch for pre/post diagnostics
eval_dl = DataLoader(valid_tok.select(range(min(32, len(valid_tok)))), batch_size=DEBUG_BATCH, shuffle=False, collate_fn=lambda x: x)
sample_batch = next(iter(eval_dl))
prompts = [ex["prompt"] for ex in sample_batch]
chosen = [ex["chosen"] for ex in sample_batch]
rejected = [ex["rejected"] for ex in sample_batch]

# Pre-train diagnostics
in_ids_c, attn_c, labels_c = tokenize_pairs(prompts, chosen)
in_ids_r, attn_r, labels_r = tokenize_pairs(prompts, rejected)
logp_c_pre = compute_logprobs(in_ids_c, attn_c, labels_c, policy_model)
logp_r_pre = compute_logprobs(in_ids_r, attn_r, labels_r, policy_model)
print("Pre-train mean chosen - rejected:", float((logp_c_pre - logp_r_pre).mean()))
print("Pre-train batch accuracy (chosen preferred):", float((logp_c_pre > logp_r_pre).float().mean()))

# === DPO config (short run) ===
config = DPOConfig(
    output_dir=DPO_OUT,
    beta=DEBUG_BETA,
    learning_rate=DEBUG_LR,
    per_device_train_batch_size=DEBUG_BATCH,
    gradient_accumulation_steps=DEBUG_ACCUM,
    num_train_epochs=DEBUG_EPOCHS,
    logging_steps=50,
    save_steps=500,
    eval_steps=200,
    report_to=[],
    padding_value=PAD_ID,
)
config.remove_unused_columns = False

# === Trainer: prevent TRL from reprocessing by setting processing_class=None ===
trainer = DPOTrainer(
    model=policy_model,
    ref_model=ref_model,
    args=config,
    train_dataset=train_tok,
    eval_dataset=valid_tok,
    data_collator=dpo_fixed_collator,
    processing_class=None,
)

# Sanity check batch shapes
dl = trainer.get_train_dataloader()
batch = next(iter(dl))
print("\nSanity check batch shapes:")
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {tuple(v.shape)}")
    else:
        print(f"  {k}: type={type(v)}")

# === Train (short run) ===
print("\nStarting short DPO run (1 epoch)...")
trainer.train()

# === Post-train diagnostics on the same sample batch ===
policy_model.eval()
logp_c_post = compute_logprobs(in_ids_c, attn_c, labels_c, policy_model)
logp_r_post = compute_logprobs(in_ids_r, attn_r, labels_r, policy_model)
print("\nPost-train mean chosen - rejected:", float((logp_c_post - logp_r_post).mean()))
print("Post-train batch accuracy (chosen preferred):", float((logp_c_post > logp_r_post).float().mean()))

# Save adapter and tokenizer
print("Saving policy adapter and tokenizer to", DPO_OUT)
policy_model.save_pretrained(DPO_OUT)
tokenizer.save_pretrained(DPO_OUT)
print("Done.")

