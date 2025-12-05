#!/usr/bin/env python3
# dpo_run_test.py
# Run full test evaluation for a saved PEFT adapter checkpoint.

import os
import json
import time
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# === USER CONFIG ===
BASE_MODEL = "/scratch/aowais2/llama"  # base model used during training
ADAPTER_CHECKPOINT = "/scratch/aowais2/llama32_dpo/best_adapter_epoch_2_step_15920"
TEST_JSON = "/scratch/aowais2/data/dpo_test.json"
OUTPUT_DIR = ADAPTER_CHECKPOINT  # results will be written here
BATCH = 8
PROMPT_MAX = 256
ANSWER_MAX = 512
MAX_TOTAL = PROMPT_MAX + ANSWER_MAX
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_LENGTH_NORMALIZED = True   # must match training evaluation mode
TIE_THRESHOLD = 1e-6          # margin threshold to count as tie
SAMPLE_MISCLASS = 400          # how many misclassified examples to save for inspection

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Utilities ===
def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{ts}] {msg}", flush=True)

def safe_tokenizer(base_model_dir: str, adapter_dir: str = None):
    """
    Load tokenizer robustly: prefer base model tokenizer; only try adapter dir if it contains tokenizer files.
    """
    try:
        return AutoTokenizer.from_pretrained(base_model_dir, local_files_only=True)
    except Exception as e_base:
        base_err = e_base

    if adapter_dir:
        candidates = ["tokenizer.json", "tokenizer_config.json", "spiece.model", "sentencepiece.model", "vocab.json"]
        has_tokenizer_files = any(os.path.isfile(os.path.join(adapter_dir, f)) for f in candidates)
        if has_tokenizer_files:
            try:
                return AutoTokenizer.from_pretrained(adapter_dir, local_files_only=True)
            except Exception:
                pass

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
        "raw": [{"prompt": p, "chosen": c, "rejected": r} for p,c,r in zip(prompts, chosens, rejects)]
    }

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
log("Loading tokenizer and models")
tokenizer = safe_tokenizer(BASE_MODEL, ADAPTER_CHECKPOINT)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, local_files_only=True)
base.to(DEVICE)

policy = PeftModel.from_pretrained(base, ADAPTER_CHECKPOINT, local_files_only=True)
policy.to(DEVICE)
policy.eval()

# === Load test dataset and precompute tokenized batches ===
log("Loading test dataset")
test_ds = load_dataset("json", data_files=TEST_JSON)["train"]
log(f"Test examples: {len(test_ds)}")

# Convert to list for deterministic indexing
test_list = []
for i in range(len(test_ds)):
    ex = test_ds[i]
    if isinstance(ex, dict):
        test_list.append(ex)
    else:
        # fallback if dataset returns row objects
        try:
            keys = test_ds.column_names
            test_list.append({k: test_ds[k][i] for k in keys})
        except Exception:
            continue

log("Tokenizing and collating test batches (cached)")
test_tokenized = [collate_fn(test_list[i:i+BATCH], tokenizer) for i in range(0, len(test_list), BATCH)]

# === Run full evaluation ===
log("Starting full test evaluation")
total = 0
correct = 0
ties = 0
margins = []
misclassified = []

for batch_idx, batch in enumerate(test_tokenized):
    sc, ac, _ = summed_and_avg_logprob_eval(batch["in_c"], batch["attn_c"], batch["labels_c"], policy)
    sr, ar, _ = summed_and_avg_logprob_eval(batch["in_r"], batch["attn_r"], batch["labels_r"], policy)
    margin = (ac - ar) if USE_LENGTH_NORMALIZED else (sc - sr)
    margin_np = margin.cpu().numpy()
    for i, m in enumerate(margin_np):
        total += 1
        margins.append(float(m))
        if m > 0:
            correct += 1
        elif abs(m) <= TIE_THRESHOLD:
            ties += 1
        else:
            # misclassified: model prefers rejected
            if len(misclassified) < SAMPLE_MISCLASS:
                raw = batch.get("raw", [None]*len(margin_np))[i]
                misclassified.append({"batch_idx": batch_idx, "idx_in_batch": i, "margin": float(m), "example": raw})

# === Summarize and save results ===
test_acc = correct / total if total else 0.0
test_tie = ties / total if total else 0.0
test_mean_margin = sum(margins) / len(margins) if margins else 0.0

results = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    "adapter_checkpoint": ADAPTER_CHECKPOINT,
    "test_file": TEST_JSON,
    "num_examples": total,
    "accuracy": test_acc,
    "tie_rate": test_tie,
    "mean_margin": test_mean_margin,
    "tie_threshold": TIE_THRESHOLD,
    "sample_misclassified_saved": len(misclassified),
}

out_path = os.path.join(OUTPUT_DIR, "test_results.json")
with open(out_path, "w") as f:
    json.dump({"results": results, "misclassified_examples": misclassified}, f, indent=2)

log(f"TEST acc {test_acc:.4f} tie {test_tie:.4f} mean_margin {test_mean_margin:.4f}")
log(f"Saved test results to {out_path}")

