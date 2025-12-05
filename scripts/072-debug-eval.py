#!/usr/bin/env python3
# evaluate_verbose_debug.py
# Verbose end-to-end evaluation with diagnostics for DPO and SFT test files.

import os
import csv
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import Any, List, Tuple

# === USER CONFIG ===
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

# === Helpers ===
def load_tokenizer(preferred_dir: str, fallback_dir: str):
    if os.path.isdir(preferred_dir):
        print(f"Loading tokenizer from {preferred_dir}")
        return AutoTokenizer.from_pretrained(preferred_dir, local_files_only=True)
    print(f"Loading tokenizer from {fallback_dir}")
    return AutoTokenizer.from_pretrained(fallback_dir, local_files_only=True)

def ensure_text(field: Any, tokenizer) -> str:
    if isinstance(field, str):
        return field
    if isinstance(field, (list, tuple)):
        try:
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

def summed_and_avg_logprob(input_ids, attention_mask, labels, model, device):
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
        per_sample_sum = token_logps.sum(dim=1).cpu()
        token_counts = valid_mask.sum(dim=1).cpu().clamp(min=1)
        per_sample_avg = per_sample_sum / token_counts
    return per_sample_sum, per_sample_avg, token_counts, logits

def ensure_list_of_dicts(x):
    if x is None:
        return []
    if isinstance(x, dict):
        return [x]
    try:
        return list(x)
    except Exception:
        return [x]

# === Load tokenizer and models ===
tokenizer = load_tokenizer(DPO_OUT, BASE_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
PAD_ID = tokenizer.pad_token_id
print("Tokenizer pad token id:", PAD_ID)

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

# === Load datasets and print basic info ===
print("\nLoading DPO test dataset:", DPO_TEST_JSON)
dpo_ds = load_dataset("json", data_files=DPO_TEST_JSON)["train"]
print("DPO dataset length:", len(dpo_ds))

print("\nLoading SFT test dataset:", SFT_TEST_JSON)
sft_ds = load_dataset("json", data_files=SFT_TEST_JSON)["train"]
print("SFT dataset length:", len(sft_ds))

# Print first 5 raw examples (types and keys)
def inspect_examples(ds, name, n=5):
    print(f"\nInspecting first {n} examples of {name}:")
    for i in range(min(n, len(ds))):
        ex = ds[i]
        print(f"  index {i}: type={type(ex)} keys={list(ex.keys()) if isinstance(ex, dict) else 'N/A'}")
        if isinstance(ex, dict):
            for k in ["prompt", "chosen", "rejected", "response"]:
                if k in ex:
                    v = ex[k]
                    t = type(v)
                    if isinstance(v, (list, tuple)):
                        print(f"    {k}: list(len={len(v)}) sample tokens:", v[:10])
                    else:
                        print(f"    {k}: str(len={len(str(v))}) preview:", str(v)[:200])
    print("----")

inspect_examples(dpo_ds, "DPO test")
inspect_examples(sft_ds, "SFT test")

# === Quick single-sample tokenization and scoring test ===
print("\nSingle-sample tokenization and scoring test (DPO):")
# find first valid DPO example
sample = None
for i in range(len(dpo_ds)):
    ex = dpo_ds[i]
    if isinstance(ex, dict) and "prompt" in ex and "chosen" in ex and "rejected" in ex:
        sample = ex
        sample_idx = i
        break
if sample is None:
    raise RuntimeError("No valid DPO example found in dataset.")

p = ensure_text(sample["prompt"], tokenizer)
c = ensure_text(sample["chosen"], tokenizer)
r = ensure_text(sample["rejected"], tokenizer)
print("sample index:", sample_idx)
print("prompt preview:", p[:300])
print("chosen preview:", c[:200])
print("rejected preview:", r[:200])

in_c, attn_c, labels_c = tokenize_pair_batch(tokenizer, [p], [c], PAD_ID)
in_r, attn_r, labels_r = tokenize_pair_batch(tokenizer, [p], [r], PAD_ID)
print("tokenized shapes chosen:", in_c.shape, "labels unique:", set(labels_c[0].tolist()))
print("tokenized shapes rejected:", in_r.shape, "labels unique:", set(labels_r[0].tolist()))

sum_c, avg_c, cnt_c, logits_c = summed_and_avg_logprob(in_c, attn_c, labels_c, dpo_policy, DEVICE)
sum_r, avg_r, cnt_r, logits_r = summed_and_avg_logprob(in_r, attn_r, labels_r, dpo_policy, DEVICE)
print("sample summed logp chosen:", float(sum_c[0].item()), "token_count:", int(cnt_c[0].item()))
print("sample summed logp rejected:", float(sum_r[0].item()), "token_count:", int(cnt_r[0].item()))
print("sample avg logp chosen:", float(avg_c[0].item()), "avg rejected:", float(avg_r[0].item()))
print("logits NaN chosen:", torch.isnan(logits_c).any().item(), "rejected:", torch.isnan(logits_r).any().item())

# If labels are all -100, warn
if (labels_c[0] == -100).all().item():
    print("WARNING: chosen labels are all -100 (continuation tokens masked out). This causes ties/zeros.")
if (labels_r[0] == -100).all().item():
    print("WARNING: rejected labels are all -100 (continuation tokens masked out). This causes ties/zeros.")

# === Full evaluation with verbose counters and skip reasons ===
print("\nRunning full verbose evaluation (DPO)...")
dpo_rows = []
dpo_counts = {"total":0, "processed":0, "skipped_missing_fields":0, "skipped_non_dict":0}
for i in range(0, len(dpo_ds), BATCH):
    slice_ds = dpo_ds[i : i + BATCH]
    batch = list(slice_ds)
    if len(batch) == 0:
        continue
    prompts, chosens, rejects, idxs = [], [], [], []
    for j, ex in enumerate(batch):
        if not isinstance(ex, dict):
            dpo_counts["skipped_non_dict"] += 1
            continue
        if "prompt" not in ex or "chosen" not in ex or "rejected" not in ex:
            dpo_counts["skipped_missing_fields"] += 1
            continue
        ptxt = ensure_text(ex["prompt"], tokenizer)
        ctxt = ensure_text(ex["chosen"], tokenizer)
        rtxt = ensure_text(ex["rejected"], tokenizer)
        prompts.append(ptxt); chosens.append(ctxt); rejects.append(rtxt); idxs.append(i + j)
    dpo_counts["total"] += len(batch)
    if len(prompts) == 0:
        continue

    in_c, attn_c, labels_c = tokenize_pair_batch(tokenizer, prompts, chosens, PAD_ID)
    in_r, attn_r, labels_r = tokenize_pair_batch(tokenizer, prompts, rejects, PAD_ID)
    sum_c, avg_c, cnt_c, _ = summed_and_avg_logprob(in_c, attn_c, labels_c, dpo_policy, DEVICE)
    sum_r, avg_r, cnt_r, _ = summed_and_avg_logprob(in_r, attn_r, labels_r, dpo_policy, DEVICE)

    for k in range(len(prompts)):
        lc = float(sum_c[k].item()); lr = float(sum_r[k].item())
        ac = float(avg_c[k].item()); ar = float(avg_r[k].item())
        raw_pref = 0 if abs(lc - lr) <= 1e-12 else 1 if lc > lr else -1
        norm_pref = 0 if abs(ac - ar) <= 1e-12 else 1 if ac > ar else -1
        dpo_rows.append({
            "index": idxs[k],
            "chosen_logp_sum": lc,
            "rejected_logp_sum": lr,
            "chosen_logp_avg": ac,
            "rejected_logp_avg": ar,
            "raw_pref": raw_pref,
            "norm_pref": norm_pref,
            "chosen_token_count": int(cnt_c[k].item()),
            "rejected_token_count": int(cnt_r[k].item()),
        })
        dpo_counts["processed"] += 1

print("DPO counters:", dpo_counts)
if dpo_rows:
    print("DPO sample rows (first 5):")
    for r in dpo_rows[:5]:
        print(r)
    with open("dpo_verbose_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dpo_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dpo_rows)
    print("Saved dpo_verbose_results.csv")
else:
    print("No DPO rows produced. Check skipped counts above.")

# === SFT evaluation verbose ===
print("\nRunning full verbose evaluation (SFT canonical-label)...")
sft_rows = []
sft_counts = {"total":0, "processed":0, "skipped_missing_fields":0, "skipped_non_dict":0}
for i in range(0, len(sft_ds), BATCH):
    slice_ds = sft_ds[i : i + BATCH]
    batch = list(slice_ds)
    if len(batch) == 0:
        continue
    prompts, responses, idxs = [], [], []
    for j, ex in enumerate(batch):
        if not isinstance(ex, dict):
            sft_counts["skipped_non_dict"] += 1
            continue
        if "prompt" not in ex or "response" not in ex:
            sft_counts["skipped_missing_fields"] += 1
            continue
        prompts.append(ensure_text(ex["prompt"], tokenizer))
        responses.append(ensure_text(ex["response"], tokenizer))
        idxs.append(i + j)
    sft_counts["total"] += len(batch)
    if len(prompts) == 0:
        continue

    in_a, attn_a, labels_a = tokenize_pair_batch(tokenizer, prompts, [LABEL_A]*len(prompts), PAD_ID)
    in_b, attn_b, labels_b = tokenize_pair_batch(tokenizer, prompts, [LABEL_B]*len(prompts), PAD_ID)
    sum_a, avg_a, cnt_a, _ = summed_and_avg_logprob(in_a, attn_a, labels_a, sft_policy, DEVICE)
    sum_b, avg_b, cnt_b, _ = summed_and_avg_logprob(in_b, attn_b, labels_b, sft_policy, DEVICE)

    for k in range(len(prompts)):
        la = float(sum_a[k].item()); lb = float(sum_b[k].item())
        aa = float(avg_a[k].item()); ab = float(avg_b[k].item())
        raw_pref = 0 if abs(la - lb) <= 1e-12 else 1 if la > lb else -1
        norm_pref = 0 if abs(aa - ab) <= 1e-12 else 1 if aa > ab else -1
        sft_rows.append({
            "index": idxs[k],
            "logp_A_sum": la,
            "logp_B_sum": lb,
            "logp_A_avg": aa,
            "logp_B_avg": ab,
            "raw_pref": raw_pref,
            "norm_pref": norm_pref,
            "token_count_A": int(cnt_a[k].item()),
            "token_count_B": int(cnt_b[k].item()),
        })
        sft_counts["processed"] += 1

print("SFT counters:", sft_counts)
if sft_rows:
    print("SFT sample rows (first 5):")
    for r in sft_rows[:5]:
        print(r)
    with open("sft_verbose_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(sft_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sft_rows)
    print("Saved sft_verbose_results.csv")
else:
    print("No SFT rows produced. Check skipped counts above.")

print("\nVerbose evaluation complete.")

