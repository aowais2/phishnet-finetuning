# Diagnostic + fix script: find ragged column(s) and either remove or filter rows
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
from peft import PeftModel
from trl import DPOTrainer, DPOConfig

# Paths (adjust if needed)
BASE_PATH = "/scratch/aowais2/llama"
SFT_ADAPTER_PATH = "/scratch/aowais2/llama32_lora_sft"
DPO_OUT = "/scratch/aowais2/llama32_lora_dpo"
DPO_TRAIN_JSON = "/scratch/aowais2/data/dpo_train.json"
DPO_VAL_JSON   = "/scratch/aowais2/data/dpo_val.json"

# Tokenizer + sizes
tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
pad_id = tokenizer.pad_token_id

PROMPT_MAX = 256
ANSWER_MAX = 512

# Helper tokenization used earlier (keeps raw text)
def tok(text: str, max_len: int):
    return tokenizer(text=str(text), truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")

def apply_simple_template(prompt: str, answer: str) -> str:
    return f"{prompt}\n{answer}"

# Load and preprocess exactly as before
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
        # keep raw text columns
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }

train_tok = train_raw.map(preprocess, batched=False)
valid_tok = valid_raw.map(preprocess, batched=False)

# Prune unexpected columns except the raw text and the six fixed fields
KEEP = {
    "prompt_input_ids", "prompt_attention_mask",
    "chosen_input_ids", "chosen_attention_mask",
    "rejected_input_ids", "rejected_attention_mask",
    "prompt", "chosen", "rejected"
}
drop = [c for c in train_tok.column_names if c not in KEEP]
if drop:
    train_tok = train_tok.remove_columns(drop)
    valid_tok = valid_tok.remove_columns([c for c in valid_tok.column_names if c not in KEEP])

print("Columns now:", train_tok.column_names)

# === Diagnostic: iterate raw examples (no collator) to find ragged columns ===
# Use a DataLoader with collate_fn=lambda x: x so we get raw examples lists
dl = DataLoader(train_tok, batch_size=1, shuffle=False, collate_fn=lambda x: x, num_workers=0)

# We'll scan the first M examples (set to len(train_tok) to scan all)
M = len(train_tok)  # start with 2000; increase if needed
if M > len(train_tok):
    M = len(train_tok)

# Expected lengths per key
expected = {
    "prompt_input_ids": PROMPT_MAX,
    "prompt_attention_mask": PROMPT_MAX,
    "chosen_input_ids": ANSWER_MAX,
    "chosen_attention_mask": ANSWER_MAX,
    "rejected_input_ids": ANSWER_MAX,
    "rejected_attention_mask": ANSWER_MAX,
}

# Collect ragged info: {col: [(idx, actual_len), ...]}
ragged = {k: [] for k in expected.keys()}

print(f"Scanning first {M} examples for ragged columns...")
for idx, batch in enumerate(dl):
    if idx >= M:
        break
    ex = batch[0]  # single example dict
    for k, exp_len in expected.items():
        if k not in ex:
            print(f"Warning: key {k} missing in example {idx}")
            continue
        val = ex[k]
        # If it's a list-like, get its length; otherwise skip
        try:
            actual_len = len(val)
        except Exception:
            actual_len = None
        if actual_len is None:
            print(f"Non-list value for {k} at idx {idx}: type={type(val)}")
        elif actual_len != exp_len:
            ragged[k].append((idx, actual_len))
    if idx % 500 == 0 and idx > 0:
        print(f"Scanned {idx} examples...")

# Summarize ragged columns
any_ragged = False
for k, items in ragged.items():
    if items:
        any_ragged = True
        print(f"\nColumn '{k}' has {len(items)} ragged examples (sample up to 10):")
        for i, l in items[:10]:
            print(f"  idx={i}, len={l}")
if not any_ragged:
    print("No ragged lengths found in the first", M, "examples for the six fixed keys.")

# If ragged found, print raw offending examples for inspection
if any_ragged:
    # Collect unique indices to inspect
    bad_indices = sorted({i for items in ragged.values() for (i, _) in items})
    print("\nInspecting offending examples (raw fields):")
    for i in bad_indices[:20]:
        row = train_tok[i]
        print(f"\n=== idx {i} ===")
        for k in row.keys():
            v = row[k]
            try:
                ln = len(v)
            except Exception:
                ln = "?"
            # Print only short preview for long lists
            preview = v if (isinstance(v, str) and len(v) < 200) else (v[:10] if isinstance(v, (list, tuple)) else str(v)[:200])
            print(f"{k}: len={ln}; preview={preview}")
    # Offer two fixes: remove offending column(s) or filter rows
    print("\nYou can fix this by either:")
    print("1) Removing the offending column(s) from the dataset (if they are debug/intermediate).")
    print("   Example: train_tok = train_tok.remove_columns(['COLUMN_NAME'])")
    print("2) Filtering out malformed rows by index or by length.")
    print("   Example: train_tok = train_tok.filter(lambda ex, idx: len(ex['prompt_input_ids'])==PROMPT_MAX, with_indices=False)")
    print("\nIf you want, I can generate the exact remove_columns() or filter() call for the columns/indices above.")

else:
    # If no ragged found in first M, scan entire dataset (slower) or inspect trainer dataloader
    print("No ragged examples found in the scanned window. If the collator still fails, the raggedness may appear later in the dataset or be introduced by TRL's internal processing.")
    print("Next step: inspect the trainer dataloader batch directly (but that will trigger the collator error).")
    print("If you want, I can now (A) scan the entire dataset, or (B) attempt a safe training run with a defensive collator that only batches the six keys.")

# End of diagnostic script

