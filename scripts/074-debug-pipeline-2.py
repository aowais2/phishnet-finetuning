#!/usr/bin/env python3
# debug_pipeline_fix.py
# Robust end-to-end debug:
# - fixes dataset slice conversion
# - computes training log-probs with gradients enabled
# - prints loss vs swapped-loss for sign check
# - checks length bias (sum vs avg)
# - runs synthetic sanity training step

import os
import math
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import Any, List

# === USER CONFIG ===
BASE_PATH = "/scratch/aowais2/llama"
SFT_ADAPTER_PATH = "/scratch/aowais2/llama32_lora_sft"
DPO_ADAPTER_PATH = "/scratch/aowais2/llama32_lora_dpo_1epoch"  # optional
DPO_TEST_JSON = "/scratch/aowais2/data/dpo_test.json"
SFT_TEST_JSON = "/scratch/aowais2/data/sft_test.json"

PROMPT_MAX = 256
ANSWER_MAX = 512
MAX_TOTAL = PROMPT_MAX + ANSWER_MAX
BATCH = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BETA = 0.1
LR = 1e-5
SAMPLE_CHECK_N = 200

# === HELPERS ===
def safe_tokenizer(preferred: str, fallback: str):
    if os.path.isdir(preferred):
        return AutoTokenizer.from_pretrained(preferred, local_files_only=True)
    return AutoTokenizer.from_pretrained(fallback, local_files_only=True)

def ensure_text(field: Any, tokenizer) -> str:
    if isinstance(field, str):
        return field
    if isinstance(field, (list, tuple)):
        try:
            return tokenizer.decode([int(x) for x in field], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        except Exception:
            return " ".join(map(str, field))
    return str(field)

def to_list_of_dicts(slice_ds):
    # Robust conversion of a dataset slice to list of dicts
    if slice_ds is None:
        return []
    if isinstance(slice_ds, dict):
        return [slice_ds]
    try:
        lst = list(slice_ds)
    except Exception:
        return []
    if len(lst) == 0:
        return []
    if isinstance(lst[0], dict):
        return lst
    # If elements are strings (column names) or scalars, reconstruct from columns
    try:
        keys = slice_ds.column_names
        return [{k: slice_ds[k][i] for k in keys} for i in range(len(slice_ds))]
    except Exception:
        # fallback: wrap scalars
        return [{"value": x} for x in lst]

def tokenize_pair_batch(tokenizer, prompts: List[str], answers: List[str], pad_id: int):
    texts = [p + "\n" + a for p, a in zip(prompts, answers)]
    enc = tokenizer(texts, truncation=True, padding="longest", max_length=MAX_TOTAL, return_tensors="pt")
    prompt_enc = tokenizer(prompts, truncation=True, padding="longest", max_length=PROMPT_MAX, return_tensors="pt")
    prompt_lens = (prompt_enc["input_ids"] != pad_id).sum(dim=1)
    labels = enc["input_ids"].clone()
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100
    return enc["input_ids"], enc["attention_mask"], labels

def summed_and_avg_logprob_train(input_ids, attention_mask, labels, model, device):
    # Training-mode computation: no torch.no_grad so gradients flow
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # (B, L, V)
    log_probs = F.log_softmax(logits, dim=-1)
    safe_labels = labels.clone()
    safe_labels[safe_labels == -100] = 0
    token_logps = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # (B, L)
    valid_mask = (labels != -100).to(token_logps.dtype)
    token_logps = token_logps * valid_mask
    per_sample_sum = token_logps.sum(dim=1)
    token_counts = valid_mask.sum(dim=1).clamp(min=1)
    per_sample_avg = per_sample_sum / token_counts
    return per_sample_sum, per_sample_avg, token_counts, logits

def summed_and_avg_logprob_eval(input_ids, attention_mask, labels, model, device):
    # Evaluation-mode computation: use no_grad
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

# === LOAD MODELS AND TOKENIZER ===
print("Loading tokenizer and models...")
tokenizer = safe_tokenizer(DPO_ADAPTER_PATH, BASE_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
PAD_ID = tokenizer.pad_token_id
print("pad token id:", PAD_ID)

base = AutoModelForCausalLM.from_pretrained(BASE_PATH, local_files_only=True)
policy = PeftModel.from_pretrained(base, SFT_ADAPTER_PATH, local_files_only=True)
policy.to(DEVICE)
policy.eval()

# Try load DPO adapter for eval if present
dpo_policy = None
if os.path.isdir(DPO_ADAPTER_PATH):
    try:
        base_dpo = AutoModelForCausalLM.from_pretrained(BASE_PATH, local_files_only=True)
        dpo_policy = PeftModel.from_pretrained(base_dpo, DPO_ADAPTER_PATH, local_files_only=True)
        dpo_policy.to(DEVICE)
        dpo_policy.eval()
        print("Loaded DPO adapter for evaluation.")
    except Exception as e:
        print("Could not load DPO adapter, using SFT policy for eval:", e)
        dpo_policy = policy
else:
    dpo_policy = policy

ref = AutoModelForCausalLM.from_pretrained(BASE_PATH, local_files_only=True).to(DEVICE)
ref.eval()

# === LOAD DATASETS ===
print("Loading datasets...")
dpo_ds = load_dataset("json", data_files=DPO_TEST_JSON)["train"]
sft_ds = load_dataset("json", data_files=SFT_TEST_JSON)["train"]
print("DPO dataset length:", len(dpo_ds))
print("SFT dataset length:", len(sft_ds))

# === QUICK CHECK: what list(slice) yields ===
print("\nBatch element type checks (first 5 slices):")
for i in range(0, min(5, len(dpo_ds)), BATCH):
    slice_ds = dpo_ds[i : i + BATCH]
    lst = list(slice_ds)
    print(f" slice {i}:{i+BATCH} -> list len {len(lst)}; first element type {type(lst[0])}")
    print(" repr of first element (truncated):", repr(lst[0])[:200])

# === ROBUST SINGLE-BATCH EVAL TEST ===
print("\nRobust single-batch evaluation test (DPO policy):")
slice_ds = dpo_ds[0 : BATCH]
batch = to_list_of_dicts(slice_ds)
print("converted batch length:", len(batch), "first element type:", type(batch[0]) if batch else None)
if len(batch) > 0:
    ex = batch[0]
    print("first element keys:", list(ex.keys()))
    p = ensure_text(ex.get("prompt", ""), tokenizer)
    c = ensure_text(ex.get("chosen", ""), tokenizer)
    r = ensure_text(ex.get("rejected", ""), tokenizer)
    print("prompt len chars:", len(p), "chosen len chars:", len(c), "rejected len chars:", len(r))
    in_c, attn_c, labels_c = tokenize_pair_batch(tokenizer, [p], [c], PAD_ID)
    in_r, attn_r, labels_r = tokenize_pair_batch(tokenizer, [p], [r], PAD_ID)
    print("token shapes:", in_c.shape, in_r.shape, "labels unique chosen:", set(labels_c[0].tolist()) )
    sum_c, avg_c, cnt_c, logits_c = summed_and_avg_logprob_eval(in_c, attn_c, labels_c, dpo_policy, DEVICE)
    sum_r, avg_r, cnt_r, logits_r = summed_and_avg_logprob_eval(in_r, attn_r, labels_r, dpo_policy, DEVICE)
    print("eval sum_c", float(sum_c[0].item()), "sum_r", float(sum_r[0].item()))
    print("eval avg_c", float(avg_c[0].item()), "avg_r", float(avg_r[0].item()))
    print("token_count chosen", int(cnt_c[0].item()), "rejected", int(cnt_r[0].item()))
    print("logits NaN chosen:", torch.isnan(logits_c).any().item(), "rejected:", torch.isnan(logits_r).any().item())

# === SINGLE-BATCH TRAINING-MODE FORWARD/BACKWARD AND SIGN CHECK ===
print("\nSingle-batch training-mode forward/backward and reward sign test")
train_batch = to_list_of_dicts(dpo_ds[0 : BATCH])
prompts, chosens, rejects = [], [], []
for ex in train_batch:
    if not isinstance(ex, dict): continue
    if "prompt" in ex and "chosen" in ex and "rejected" in ex:
        prompts.append(ensure_text(ex["prompt"], tokenizer))
        chosens.append(ensure_text(ex["chosen"], tokenizer))
        rejects.append(ensure_text(ex["rejected"], tokenizer))
if len(prompts) == 0:
    raise RuntimeError("No valid training examples in sample batch")

# Tokenize
in_c, attn_c, labels_c = tokenize_pair_batch(tokenizer, prompts, chosens, PAD_ID)
in_r, attn_r, labels_r = tokenize_pair_batch(tokenizer, prompts, rejects, PAD_ID)

# Ensure adapter params are trainable
for n,p in policy.named_parameters():
    if "lora" in n.lower() or "adapter" in n.lower():
        p.requires_grad = True

policy.train()
# Compute logprobs with gradients enabled
sum_c, avg_c, cnt_c, logits_c = summed_and_avg_logprob_train(in_c, attn_c, labels_c, policy, DEVICE)
sum_r, avg_r, cnt_r, logits_r = summed_and_avg_logprob_train(in_r, attn_r, labels_r, policy, DEVICE)
print("train-mode summed logp chosen:", sum_c.tolist())
print("train-mode summed logp rejected:", sum_r.tolist())
reward = sum_c - sum_r
print("reward per sample:", reward.tolist(), "mean reward:", float(reward.mean().item()))

# Loss and swapped loss for sign check
loss = -torch.log(torch.sigmoid(BETA * reward)).mean()
loss_swapped = -torch.log(torch.sigmoid(BETA * (sum_r - sum_c))).mean()
print("loss:", float(loss.item()), "loss_swapped:", float(loss_swapped.item()))

# Backward step
opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=LR)
opt.zero_grad()
loss.backward()
num_with_grad = sum(1 for p in policy.parameters() if p.requires_grad and p.grad is not None)
grad_norm = math.sqrt(sum((p.grad.detach()**2).sum().item() for p in policy.parameters() if p.requires_grad and p.grad is not None)) if num_with_grad>0 else 0.0
print("num params with grad:", num_with_grad, "grad_norm:", grad_norm)
opt.step()

# === LENGTH-BIAS CHECK ===
print("\nLength-bias check on first", SAMPLE_CHECK_N, "examples")
counts = []
margins_sum = []
margins_avg = []
for i in range(0, min(SAMPLE_CHECK_N, len(dpo_ds)), 1):
    ex = dpo_ds[i]
    if not isinstance(ex, dict): continue
    p = ensure_text(ex["prompt"], tokenizer)
    c = ensure_text(ex["chosen"], tokenizer)
    r = ensure_text(ex["rejected"], tokenizer)
    in_c, attn_c, labels_c = tokenize_pair_batch(tokenizer, [p], [c], PAD_ID)
    in_r, attn_r, labels_r = tokenize_pair_batch(tokenizer, [p], [r], PAD_ID)
    sum_c, avg_c, cnt_c, _ = summed_and_avg_logprob_eval(in_c, attn_c, labels_c, dpo_policy, DEVICE)
    sum_r, avg_r, cnt_r, _ = summed_and_avg_logprob_eval(in_r, attn_r, labels_r, dpo_policy, DEVICE)
    counts.append((int(cnt_c[0].item()), int(cnt_r[0].item())))
    margins_sum.append(float((sum_c - sum_r).item()))
    margins_avg.append(float((avg_c - avg_r).item()))
print("sample token counts (chosen,rejected) first 10:", counts[:10])
print("sample summed margins first 10:", margins_sum[:10])
print("sample avg margins first 10:", margins_avg[:10])
avg_chosen_len = sum(c for c,r in counts)/len(counts)
avg_rejected_len = sum(r for c,r in counts)/len(counts)
print("avg chosen token count:", avg_chosen_len, "avg rejected token count:", avg_rejected_len)
print("mean summed margin over sample:", sum(margins_sum)/len(margins_sum))
print("mean avg margin over sample:", sum(margins_avg)/len(margins_avg))

# === SYNTHETIC SANITY TEST ===
print("\nSynthetic sanity test")
synth_prompts = ["Q: 2+2 ="] * 8
synth_chosen = ["4"] * 8
synth_rejected = ["banana"] * 8
in_c_s, attn_c_s, labels_c_s = tokenize_pair_batch(tokenizer, synth_prompts, synth_chosen, PAD_ID)
in_r_s, attn_r_s, labels_r_s = tokenize_pair_batch(tokenizer, synth_prompts, synth_rejected, PAD_ID)
policy.train()
opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=LR)
sum_c_s, avg_c_s, cnt_c_s, _ = summed_and_avg_logprob_train(in_c_s, attn_c_s, labels_c_s, policy, DEVICE)
sum_r_s, avg_r_s, cnt_r_s, _ = summed_and_avg_logprob_train(in_r_s, attn_r_s, labels_r_s, policy, DEVICE)
print("before step synth sums chosen/rejected:", sum_c_s.tolist(), sum_r_s.tolist())
reward_s = sum_c_s - sum_r_s
loss_s = -torch.log(torch.sigmoid(BETA * reward_s)).mean()
opt.zero_grad()
loss_s.backward()
opt.step()
sum_c_s_after, avg_c_s_after, _, _ = summed_and_avg_logprob_train(in_c_s, attn_c_s, labels_c_s, policy, DEVICE)
sum_r_s_after, avg_r_s_after, _, _ = summed_and_avg_logprob_train(in_r_s, attn_r_s, labels_r_s, policy, DEVICE)
print("after step synth sums chosen/rejected:", sum_c_s_after.tolist(), sum_r_s_after.tolist())
print("synthetic margin before:", float((sum_c_s - sum_r_s).mean().item()), "after:", float((sum_c_s_after - sum_r_s_after).mean().item()))


