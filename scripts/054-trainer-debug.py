#!/usr/bin/env python3
# trainer_debug_run.py
# Run a short, instrumented training run that prints the exact 'features' passed to the collator.

import traceback
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
from peft import PeftModel
from trl import DPOTrainer, DPOConfig

# === Config (adjust paths if needed) ===
BASE_PATH = "/scratch/aowais2/llama"
SFT_ADAPTER_PATH = "/scratch/aowais2/llama32_lora_sft"
DPO_OUT = "/scratch/aowais2/llama32_lora_dpo"
DPO_TRAIN_JSON = "/scratch/aowais2/data/dpo_train.json"
DPO_VAL_JSON   = "/scratch/aowais2/data/dpo_val.json"

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

# === Load models (small overhead but required for Trainer) ===
base_model = AutoModelForCausalLM.from_pretrained(BASE_PATH, device_map="auto", dtype="auto")
policy_model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)

ref_model_base = AutoModelForCausalLM.from_pretrained(BASE_PATH, device_map="auto", dtype="auto")
ref_model = PeftModel.from_pretrained(ref_model_base, SFT_ADAPTER_PATH)
ref_model.requires_grad_(False)

# === Load and preprocess dataset (same as your pipeline) ===
train_raw = load_dataset("json", data_files=DPO_TRAIN_JSON)["train"]
valid_raw = load_dataset("json", data_files=DPO_VAL_JSON)["train"]

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

print("Tokenizing (may take a minute)...")
train_tok = train_raw.map(preprocess, batched=False)
valid_tok = valid_raw.map(preprocess, batched=False)

# Keep only the six numeric fields + raw text (same as before)
KEEP = {
    "prompt_input_ids", "prompt_attention_mask",
    "chosen_input_ids", "chosen_attention_mask",
    "rejected_input_ids", "rejected_attention_mask",
    "prompt", "chosen", "rejected"
}
train_tok = train_tok.remove_columns([c for c in train_tok.column_names if c not in KEEP])
valid_tok = valid_tok.remove_columns([c for c in valid_tok.column_names if c not in KEEP])

print("Columns after pruning:", train_tok.column_names)
print("Train size:", len(train_tok), "Eval size:", len(valid_tok))

# === Use a small subset to reproduce quickly ===
SUBSET = 512  # small but enough to exercise dataloader; increase if needed
train_subset = train_tok.select(range(min(SUBSET, len(train_tok))))
valid_subset = valid_tok.select(range(min(256, len(valid_tok))))

# === DPO config tuned for quick debug ===
config = DPOConfig(
    output_dir=DPO_OUT,
    beta=0.1,
    learning_rate=1e-5,
    per_device_train_batch_size=4,   # keep small
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=1000,
    eval_steps=100,
    report_to=[],
    padding_value=pad_id,
)

# === Monkeypatch debug collator ===
_orig_collator = transformers.data.data_collator.default_data_collator

def debug_collator(features):
    # Print a concise but complete diagnostic of the raw features list
    print("\n=== debug_collator invoked ===")
    print("Number of feature dicts:", len(features))
    print("Keys in first feature:", list(features[0].keys()))
    for k in features[0].keys():
        lens = []
        for i, f in enumerate(features):
            v = f.get(k, None)
            try:
                lens.append(len(v))
            except Exception:
                lens.append("type=" + str(type(v)))
        print(f"  {k}: {lens}")
    # Print small previews for each sample for keys that look suspicious (lengths not equal to expected)
    expected = {
        "prompt_input_ids": PROMPT_MAX,
        "prompt_attention_mask": PROMPT_MAX,
        "chosen_input_ids": ANSWER_MAX,
        "chosen_attention_mask": ANSWER_MAX,
        "rejected_input_ids": ANSWER_MAX,
        "rejected_attention_mask": ANSWER_MAX,
    }
    for k in features[0].keys():
        # if any sample length differs from expected, print previews
        try:
            lengths = [len(f[k]) for f in features]
        except Exception:
            lengths = None
        if lengths is not None:
            if k in expected and any(l != expected[k] for l in lengths):
                print(f"  >>> Offending key {k} lengths: {lengths}")
                for i, f in enumerate(features):
                    v = f.get(k, None)
                    preview = v if (isinstance(v, str) and len(v) < 200) else (v[:20] if isinstance(v, (list, tuple)) else str(v)[:200])
                    print(f"    sample {i} preview: {preview}")
    # Call original collator and let it raise if it fails (we already printed features)
    try:
        return _orig_collator(features)
    except Exception:
        print("Original collator raised an exception. Traceback follows:")
        traceback.print_exc()
        raise

# Ensure Trainer uses our debug_collator
transformers.data.data_collator.default_data_collator = debug_collator

# === Create trainer (processing_class=tokenizer as before) ===
trainer = DPOTrainer(
    model=policy_model,
    ref_model=ref_model,
    args=config,
    train_dataset=train_subset,
    eval_dataset=valid_subset,
    data_collator=debug_collator,
    processing_class=tokenizer,
)

# Force single-worker dataloader to avoid multi-worker nondeterminism
# (Trainer uses config and environment; this ensures num_workers=0 in DataLoader)
# Now run training and capture any exception
print("Starting instrumented trainer.train() — it will print raw features if collator fails.")
try:
    trainer.train()
    print("trainer.train() completed without collator error.")
except Exception as e:
    print("\ntrainer.train() raised an exception (collator failure likely).")
    print("Exception:", repr(e))
    print("The debug_collator above printed the raw 'features' list and per-sample lengths.")
    # Re-raise to preserve exit code if you want
    raise

