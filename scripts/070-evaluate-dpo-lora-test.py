#!/usr/bin/env python3
# evaluate_dpo_and_sft_fixed.py
import os
import csv
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import List, Tuple, Any

# -------------------------
# User configuration
# -------------------------
BASE_PATH = "/scratch/aowais2/llama"
SFT_ADAPTER_PATH = "/scratch/aowais2/llama32_lora_sft"
DPO_OUT = "/scratch/aowais2/llama32_lora_dpo_1epoch"
DPO_TEST_JSON = "/scratch/aowais2/data/dpo_test.json"
SFT_TEST_JSON = "/scratch/aowais2/data/sft_test.json"

PROMPT_MAX = 256
ANSWER_MAX = 512
MAX_TOTAL = PROMPT_MAX + ANSWER_MAX
BATCH = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_A = "This is a phishing attempt."
LABEL_B = "This is a safe message."

DPO_CSV = "/scratch/aowais2/data/dpo_test_results.csv"
SFT_CSV = "/scratch/aowais2/data/sft_test_results.csv"

# -------------------------
# Helpers
# -------------------------
def load_tokenizer(preferred_dir: str, fallback_dir: str):
    if os.path.isdir(preferred_dir):
        return AutoTokenizer.from_pretrained(preferred_dir, local_files_only=True)
    return AutoTokenizer.from_pretrained(fallback_dir, local_files_only=True)

def ensure_text(field: Any, tokenizer) -> str:
    """
    If field is a string, return it.
    If field is a list of ints (token ids), decode to text.
    Otherwise try str(field).
    """
    if isinstance(field, str):
        return field
    if isinstance(field, (list, tuple)):
        try:
            # remove padding ids (common pad token id) heuristically if present
            return tokenizer.decode([int(x) for x in field], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        except Exception:
            return " ".join(map(str, field))
    return str(field)

def tokenize_pair_batch(tokenizer, prompts: List[str], answers: List[str], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    texts = [p + "\n" + a for p, a in zip(prompts, answers)]
    enc = tokenizer(
        texts,
        truncation=True,
        padding="longest",
        max_length=MAX_TOTAL,
        return_tensors="pt",
    )
    prompt_enc = tokenizer(prompts, truncation=True, padding="longest", max_length=PROMPT_MAX, return_tensors="pt")
    prompt_lens = (prompt_enc["input_ids"] != pad_id).sum(dim=1)
    labels = enc["input_ids"].clone()
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100
    return enc["input_ids"], enc["attention_mask"], labels

def summed_logprob(input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor, model, device: str) -> torch.Tensor:
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        safe_labels = labels.clone()
        safe_labels[safe_labels == -100] = 0
        token_logps = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        valid_mask = (labels != -100).to(token_logps.dtype)
        token_logps = token_logps * valid_mask
        per_sample = token_logps.sum(dim=1).cpu()
    return per_sample

def ensure_list_of_dicts(x):
    if x is None:
        return []
    if isinstance(x, dict):
        return [x]
    try:
        return list(x)
    except Exception:
        return [x]

# -------------------------
# Load tokenizer and models
# -------------------------
print("Loading tokenizer...")
tokenizer = load_tokenizer(DPO_OUT, BASE_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
PAD_ID = tokenizer.pad_token_id

print("Loading DPO policy (base + adapter)...")
base_for_dpo = AutoModelForCausalLM.from_pretrained(BASE_PATH, local_files_only=True)
dpo_policy = PeftModel.from_pretrained(base_for_dpo, DPO_OUT, local_files_only=True)
dpo_policy.to(DEVICE)
dpo_policy.eval()

print("Loading SFT policy (base + adapter)...")
base_for_sft = AutoModelForCausalLM.from_pretrained(BASE_PATH, local_files_only=True)
sft_policy = PeftModel.from_pretrained(base_for_sft, SFT_ADAPTER_PATH, local_files_only=True)
sft_policy.to(DEVICE)
sft_policy.eval()

# -------------------------
# Load datasets (no splitting)
# -------------------------
print("Loading DPO test dataset:", DPO_TEST_JSON)
dpo_ds = load_dataset("json", data_files=DPO_TEST_JSON)["train"]  # 'train' is just the split name

print("Loading SFT test dataset:", SFT_TEST_JSON)
sft_ds = load_dataset("json", data_files=SFT_TEST_JSON)["train"]

# -------------------------
# Evaluate DPO test
# -------------------------
print("\nEvaluating DPO policy on test pairs...")
dpo_results = []
dpo_correct = 0
dpo_ties = 0
dpo_total = 0
dpo_margins = []

for i in range(0, len(dpo_ds), BATCH):
    slice_ds = dpo_ds[i : i + BATCH]
    batch = ensure_list_of_dicts(slice_ds)
    if len(batch) == 0:
        continue

    prompts = []
    chosens = []
    rejects = []
    for ex in batch:
        if not isinstance(ex, dict):
            continue
        # Accept either raw text fields or token lists
        if "prompt" not in ex or "chosen" not in ex or "rejected" not in ex:
            continue
        prompts.append(ensure_text(ex["prompt"], tokenizer))
        chosens.append(ensure_text(ex["chosen"], tokenizer))
        rejects.append(ensure_text(ex["rejected"], tokenizer))

    if len(prompts) == 0:
        continue

    in_c, attn_c, labels_c = tokenize_pair_batch(tokenizer, prompts, chosens, PAD_ID)
    in_r, attn_r, labels_r = tokenize_pair_batch(tokenizer, prompts, rejects, PAD_ID)

    logp_c = summed_logprob(in_c, attn_c, labels_c, dpo_policy, DEVICE)
    logp_r = summed_logprob(in_r, attn_r, labels_r, dpo_policy, DEVICE)

    for j in range(len(prompts)):
        lc = float(logp_c[j].item())
        lr = float(logp_r[j].item())
        margin = lc - lr
        dpo_results.append({
            "index": i + j,
            "prompt": prompts[j][:400],
            "chosen_logp": lc,
            "rejected_logp": lr,
            "margin": margin,
            "pred_prefers_chosen": int(lc > lr),
            "tie": int(lc == lr),
        })
        dpo_total += 1
        if lc > lr:
            dpo_correct += 1
        elif lc == lr:
            dpo_ties += 1
        dpo_margins.append(margin)

dpo_accuracy = dpo_correct / dpo_total if dpo_total else 0.0
dpo_tie_rate = dpo_ties / dpo_total if dpo_total else 0.0
dpo_avg_margin = float(torch.tensor(dpo_margins).mean()) if dpo_margins else 0.0

print("\nDPO policy evaluation summary:")
print(f"  examples: {dpo_total}")
print(f"  accuracy (chosen preferred): {dpo_accuracy:.4f}")
print(f"  tie rate: {dpo_tie_rate:.4f}")
print(f"  avg margin (chosen - rejected): {dpo_avg_margin:.4f}")

if dpo_results:
    with open(DPO_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dpo_results[0].keys()))
        writer.writeheader()
        writer.writerows(dpo_results)
    print("Saved DPO per-example results to", DPO_CSV)

# -------------------------
# Evaluate SFT test
# -------------------------
print("\nEvaluating SFT policy on test prompts (canonical-label comparison)...")
sft_results = []
sft_correct = 0
sft_ties = 0
sft_total = 0
sft_margins = []

for i in range(0, len(sft_ds), BATCH):
    slice_ds = sft_ds[i : i + BATCH]
    batch = ensure_list_of_dicts(slice_ds)
    if len(batch) == 0:
        continue

    prompts = []
    responses = []
    for ex in batch:
        if not isinstance(ex, dict):
            continue
        if "prompt" not in ex or "response" not in ex:
            continue
        prompts.append(ensure_text(ex["prompt"], tokenizer))
        responses.append(ensure_text(ex["response"], tokenizer))

    if len(prompts) == 0:
        continue

    in_a, attn_a, labels_a = tokenize_pair_batch(tokenizer, prompts, [LABEL_A] * len(prompts), PAD_ID)
    in_b, attn_b, labels_b = tokenize_pair_batch(tokenizer, prompts, [LABEL_B] * len(prompts), PAD_ID)

    logp_a = summed_logprob(in_a, attn_a, labels_a, sft_policy, DEVICE)
    logp_b = summed_logprob(in_b, attn_b, labels_b, sft_policy, DEVICE)

    for j in range(len(prompts)):
        la = float(logp_a[j].item())
        lb = float(logp_b[j].item())
        model_prefers = LABEL_A if la > lb else LABEL_B if lb > la else "tie"

        gt = responses[j]
        if gt.strip() == LABEL_A:
            gt_label = LABEL_A
        elif gt.strip() == LABEL_B:
            gt_label = LABEL_B
        else:
            gt_label = LABEL_A if "phish" in gt.lower() else LABEL_B

        correct = int((model_prefers == gt_label))
        tie = int(model_prefers == "tie")
        margin = la - lb

        sft_results.append({
            "index": i + j,
            "prompt": prompts[j][:400],
            "response": responses[j][:200],
            "logp_label_A": la,
            "logp_label_B": lb,
            "model_prefers": model_prefers,
            "gt_label": gt_label,
            "correct": correct,
            "tie": tie,
            "margin_A_minus_B": margin,
        })
        sft_total += 1
        sft_correct += correct
        sft_ties += tie
        sft_margins.append(margin)

sft_accuracy = sft_correct / sft_total if sft_total else 0.0
sft_tie_rate = sft_ties / sft_total if sft_total else 0.0
sft_avg_margin = float(torch.tensor(sft_margins).mean()) if sft_margins else 0.0

print("\nSFT policy evaluation summary (canonical-label comparison):")
print(f"  examples: {sft_total}")
print(f"  accuracy (model prefers ground-truth canonical label): {sft_accuracy:.4f}")
print(f"  tie rate: {sft_tie_rate:.4f}")
print(f"  avg margin (A - B): {sft_avg_margin:.4f}")

if sft_results:
    with open(SFT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sft_results[0].keys()))
        writer.writeheader()
        writer.writerows(sft_results)
    print("Saved SFT per-example results to", SFT_CSV)

print("\nEvaluation complete.")

