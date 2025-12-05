#!/usr/bin/env python3
# minimal_collator_test.py
import sys
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, default_data_collator

# === Config ===
BASE_PATH = "/scratch/aowais2/llama"
DPO_TRAIN_JSON = "/scratch/aowais2/data/dpo_train.json"
PROMPT_MAX = 256
ANSWER_MAX = 512

# === Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
pad_id = tokenizer.pad_token_id

def tok(text: str, max_len: int):
    return tokenizer(text=str(text), truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")

def apply_simple_template(prompt: str, answer: str) -> str:
    return f"{prompt}\n{answer}"

# === Load and preprocess dataset (standalone) ===
raw = load_dataset("json", data_files=DPO_TRAIN_JSON)["train"]

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
        # keep raw text if you want to inspect later
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }

print("Tokenizing dataset (this may take a while)...")
ds = raw.map(preprocess, batched=False)

# Prune unexpected columns but keep the six fixed keys + raw text
KEEP = {
    "prompt_input_ids", "prompt_attention_mask",
    "chosen_input_ids", "chosen_attention_mask",
    "rejected_input_ids", "rejected_attention_mask",
    "prompt", "chosen", "rejected"
}
drop = [c for c in ds.column_names if c not in KEEP]
if drop:
    ds = ds.remove_columns(drop)

print("Columns in processed dataset:", ds.column_names)
print("Dataset size:", len(ds))

# === Build raw DataLoader that returns lists of raw examples ===
dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=lambda x: x, num_workers=0)

# Grab one raw features list and attempt to collate
features = next(iter(dl))  # list of 4 dicts
print("\nNumber of examples in features:", len(features))
print("Keys in first example:", list(features[0].keys()))

try:
    batch = default_data_collator(features)
    print("\nCollation succeeded. Batch keys and shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)}")
        else:
            print(f"  {k}: type={type(v)}")
    sys.exit(0)
except Exception as e:
    print("\ndefault_data_collator raised:", repr(e))

# Diagnose which key fails by attempting to stack each key individually
print("\nDiagnosing per-key stacking ability and per-sample lengths:")
for k in features[0].keys():
    try:
        vals = [f[k] for f in features]
        # attempt to convert to tensor to reproduce the exact error
        torch.tensor(vals, dtype=torch.long)
        print(f"Key '{k}': OK to stack.")
    except Exception as inner_e:
        lengths = []
        for i, f in enumerate(features):
            v = f.get(k, None)
            try:
                lengths.append(len(v))
            except Exception:
                lengths.append(f"type={type(v)}")
        print(f"Key '{k}' FAILED to stack. per-sample lengths/types: {lengths}")
        # Print small previews for offending samples
        for i, f in enumerate(features):
            v = f.get(k, None)
            preview = v if (isinstance(v, str) and len(v) < 200) else (v[:20] if isinstance(v, (list, tuple)) else str(v)[:200])
            print(f"  sample {i} preview: {preview}")

