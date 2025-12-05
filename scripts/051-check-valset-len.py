# Validate the eval dataset for ragged lengths (run in same env as previous script)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

# Paths and sizes (match your training script)
BASE_PATH = "/scratch/aowais2/llama"
DPO_VAL_JSON = "/scratch/aowais2/data/dpo_val.json"
PROMPT_MAX = 256
ANSWER_MAX = 512

# Load tokenizer (only needed for pad token preview if desired)
tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
pad_id = tokenizer.pad_token_id

# Load the already-preprocessed validation dataset if available, otherwise re-run preprocess
# If you have `valid_tok` from previous run, skip load_dataset and use it directly.
valid_raw = load_dataset("json", data_files=DPO_VAL_JSON)["train"]

# If you already have a preprocess function, reuse it. Otherwise, minimal re-tokenize:
def tok(text: str, max_len: int):
    return tokenizer(text=str(text), truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")

def apply_simple_template(prompt: str, answer: str) -> str:
    return f"{prompt}\n{answer}"

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

valid_tok = valid_raw.map(preprocess, batched=False)

# Prune unexpected columns but keep raw text and the six fixed fields
KEEP = {
    "prompt_input_ids", "prompt_attention_mask",
    "chosen_input_ids", "chosen_attention_mask",
    "rejected_input_ids", "rejected_attention_mask",
    "prompt", "chosen", "rejected"
}
drop = [c for c in valid_tok.column_names if c not in KEEP]
if drop:
    valid_tok = valid_tok.remove_columns(drop)

print("Validation columns:", valid_tok.column_names)
print("Validation size:", len(valid_tok))

# Diagnostic scan over validation examples
from torch.utils.data import DataLoader
dl = DataLoader(valid_tok, batch_size=1, shuffle=False, collate_fn=lambda x: x, num_workers=0)

expected = {
    "prompt_input_ids": PROMPT_MAX,
    "prompt_attention_mask": PROMPT_MAX,
    "chosen_input_ids": ANSWER_MAX,
    "chosen_attention_mask": ANSWER_MAX,
    "rejected_input_ids": ANSWER_MAX,
    "rejected_attention_mask": ANSWER_MAX,
}

M = len(valid_tok)  # scan entire validation set; reduce if you want a sample
ragged = {k: [] for k in expected.keys()}

print(f"Scanning {M} validation examples for ragged columns...")
for idx, batch in enumerate(dl):
    if idx >= M:
        break
    ex = batch[0]
    for k, exp_len in expected.items():
        if k not in ex:
            ragged[k].append((idx, "missing"))
            continue
        try:
            actual_len = len(ex[k])
        except Exception:
            actual_len = None
        if actual_len != exp_len:
            ragged[k].append((idx, actual_len))
    if idx % 1000 == 0 and idx > 0:
        print(f"Scanned {idx} examples...")

# Report results
any_ragged = False
for k, items in ragged.items():
    if items:
        any_ragged = True
        print(f"\nColumn '{k}' ragged count: {len(items)}; sample up to 10:")
        for i, l in items[:10]:
            print(f"  idx={i}, len={l}")

if not any_ragged:
    print("No ragged lengths found in validation dataset for the six fixed keys.")

# If ragged found, print raw offending examples for inspection
if any_ragged:
    bad_indices = sorted({i for items in ragged.values() for (i, _) in items})
    print("\nInspecting offending validation examples (preview):")
    for i in bad_indices[:20]:
        row = valid_tok[i]
        print(f"\n=== idx {i} ===")
        for k in row.keys():
            v = row[k]
            try:
                ln = len(v)
            except Exception:
                ln = "?"
            preview = v if (isinstance(v, str) and len(v) < 200) else (v[:10] if isinstance(v, (list, tuple)) else str(v)[:200])
            print(f"{k}: len={ln}; preview={preview}")

    print("\nSuggested fixes:")
    print("- If the offending column is an intermediate debug field, remove it:")
    print("    valid_tok = valid_tok.remove_columns(['COLUMN_NAME'])")
    print("- If rows are malformed, filter them out by length:")
    print("    valid_tok = valid_tok.filter(lambda ex: len(ex['the_key']) == EXPECTED_LEN)")

