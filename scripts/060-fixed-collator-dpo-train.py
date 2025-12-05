#!/usr/bin/env python3
# final_dpo_fixed_collator.py

import os
from typing import List, Dict, Any
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import DPOTrainer, DPOConfig

# === User config: adjust these paths and sizes ===
BASE_PATH = "/scratch/aowais2/llama"                 # base model path
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
    # deterministic concatenation template
    return f"{prompt}\n{answer}"

# === Load models ===
base_model = AutoModelForCausalLM.from_pretrained(BASE_PATH, device_map="auto", dtype="auto")
policy_model = PeftModel.from_pretrained(base_model, SFT_ADAPTER_PATH)

ref_model_base = AutoModelForCausalLM.from_pretrained(BASE_PATH, device_map="auto", dtype="auto")
ref_model = PeftModel.from_pretrained(ref_model_base, SFT_ADAPTER_PATH)
ref_model.requires_grad_(False)

# === Load raw datasets ===
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
        # keep raw text so TRL won't complain if it expects them
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }

print("Tokenizing train dataset...")
train_tok = train_raw.map(preprocess, batched=False)
print("Tokenizing eval dataset...")
valid_tok = valid_raw.map(preprocess, batched=False)

# Keep only the six numeric fields plus raw text (no other variable-length debug columns)
KEEP = {
    "prompt_input_ids", "prompt_attention_mask",
    "chosen_input_ids", "chosen_attention_mask",
    "rejected_input_ids", "rejected_attention_mask",
    "prompt", "chosen", "rejected",
}
train_tok = train_tok.remove_columns([c for c in train_tok.column_names if c not in KEEP])
valid_tok = valid_tok.remove_columns([c for c in valid_tok.column_names if c not in KEEP])

print("Train columns:", train_tok.column_names)
print("Eval columns:", valid_tok.column_names)
print("Train size:", len(train_tok), "Eval size:", len(valid_tok))

# === Deterministic collator that enforces fixed lengths ===
def enforce_fixed_length(seq: List[int], target_len: int, pad_id: int) -> List[int]:
    if len(seq) >= target_len:
        return seq[:target_len]
    return seq + [pad_id] * (target_len - len(seq))

def dpo_fixed_collator(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}

    # prompt ids and mask
    prompts = [enforce_fixed_length(f["prompt_input_ids"], PROMPT_MAX, PAD_ID) for f in features]
    out["prompt_input_ids"] = torch.tensor(prompts, dtype=torch.long)
    prompt_masks = [[0 if token == PAD_ID else 1 for token in p] for p in prompts]
    out["prompt_attention_mask"] = torch.tensor(prompt_masks, dtype=torch.long)

    # chosen ids and mask
    chosens = [enforce_fixed_length(f["chosen_input_ids"], ANSWER_MAX, PAD_ID) for f in features]
    out["chosen_input_ids"] = torch.tensor(chosens, dtype=torch.long)
    chosen_masks = [[0 if token == PAD_ID else 1 for token in c] for c in chosens]
    out["chosen_attention_mask"] = torch.tensor(chosen_masks, dtype=torch.long)

    # rejected ids and mask
    rejects = [enforce_fixed_length(f["rejected_input_ids"], ANSWER_MAX, PAD_ID) for f in features]
    out["rejected_input_ids"] = torch.tensor(rejects, dtype=torch.long)
    rejected_masks = [[0 if token == PAD_ID else 1 for token in r] for r in rejects]
    out["rejected_attention_mask"] = torch.tensor(rejected_masks, dtype=torch.long)

    return out

# === DPO config ===
config = DPOConfig(
    output_dir=DPO_OUT,
    beta=0.1,
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=500,
    eval_steps=200,
    report_to=[],
    padding_value=PAD_ID,
)
# Ensure Trainer does not drop columns you rely on
config.remove_unused_columns = False

# === Trainer: prevent TRL from reprocessing by setting processing_class=None ===
trainer = DPOTrainer(
    model=policy_model,
    ref_model=ref_model,
    args=config,
    train_dataset=train_tok,
    eval_dataset=valid_tok,
    data_collator=dpo_fixed_collator,  # deterministic collator
    processing_class=None,             # IMPORTANT: prevent TRL re-tokenization
)

# === Sanity check one batch shapes before training ===
dl = trainer.get_train_dataloader()
batch = next(iter(dl))
print("\nSanity check batch shapes:")
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {tuple(v.shape)}")
    else:
        print(f"  {k}: type={type(v)}")

# Expect shapes:
# prompt_input_ids: (B, PROMPT_MAX)
# prompt_attention_mask: (B, PROMPT_MAX)
# chosen_input_ids: (B, ANSWER_MAX)
# chosen_attention_mask: (B, ANSWER_MAX)
# rejected_input_ids: (B, ANSWER_MAX)
# rejected_attention_mask: (B, ANSWER_MAX)

# === Train and save ===
trainer.train()
policy_model.save_pretrained(DPO_OUT)
tokenizer.save_pretrained(DPO_OUT)
print("Training complete. Model and tokenizer saved to", DPO_OUT)

