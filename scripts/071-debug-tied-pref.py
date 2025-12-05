#!/usr/bin/env python3
# single_batch_debug_end_to_end.py
# End-to-end single-batch debug for DPO training: loads data, models, runs forward/backward,
# checks for NaNs, grad norms, optimizer state, and prints parameter stats.

import os
import math
import json
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# === USER CONFIG: edit these paths to match your environment ===
BASE_PATH = "/scratch/aowais2/llama"                 # base model folder
SFT_ADAPTER_PATH = "/scratch/aowais2/llama32_lora_sft"  # SFT LoRA adapter folder
DPO_OUT = "/scratch/aowais2/llama32_lora_dpo_1epoch"        # optional: saved DPO adapter (not required)
SAMPLE_JSON = "/scratch/aowais2/data/dpo_train.json" # dataset to sample a batch from
PROMPT_MAX = 256
ANSWER_MAX = 512
MAX_TOTAL = PROMPT_MAX + ANSWER_MAX
BATCH_SIZE = 4
BETA = 0.1
LR = 1e-5
USE_AMP = False   # set True if you want to test mixed precision (try False first)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_TOP_PARAMS = 10  # how many param stats to print

# === Utilities ===
def safe_load_tokenizer(preferred_dir, fallback_dir):
    if os.path.isdir(preferred_dir):
        return AutoTokenizer.from_pretrained(preferred_dir, local_files_only=True)
    return AutoTokenizer.from_pretrained(fallback_dir, local_files_only=True)

def ensure_text(field, tokenizer):
    if isinstance(field, str):
        return field
    if isinstance(field, (list, tuple)):
        try:
            return tokenizer.decode([int(x) for x in field], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        except Exception:
            return " ".join(map(str, field))
    return str(field)

def tokenize_pair_batch(tokenizer, prompts, answers, pad_id):
    texts = [p + "\n" + a for p, a in zip(prompts, answers)]
    enc = tokenizer(texts, truncation=True, padding="longest", max_length=MAX_TOTAL, return_tensors="pt")
    prompt_enc = tokenizer(prompts, truncation=True, padding="longest", max_length=PROMPT_MAX, return_tensors="pt")
    prompt_lens = (prompt_enc["input_ids"] != pad_id).sum(dim=1)
    labels = enc["input_ids"].clone()
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100
    return enc["input_ids"], enc["attention_mask"], labels

def summed_logprob_from_model(input_ids, attention_mask, labels, model, device):
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    safe_labels = labels.clone()
    safe_labels[safe_labels == -100] = 0
    token_logps = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    valid_mask = (labels != -100).to(token_logps.dtype)
    token_logps = token_logps * valid_mask
    per_sample = token_logps.sum(dim=1)
    return per_sample, logits

def param_stats(params):
    vals = torch.cat([p.detach().view(-1).cpu() for p in params if p is not None and p.numel() > 0])
    return float(vals.mean().item()), float(vals.std().item()), int(vals.numel())

# === Load tokenizer and models ===
print("Loading tokenizer and models...")
tokenizer = safe_load_tokenizer(DPO_OUT, BASE_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
PAD_ID = tokenizer.pad_token_id

print("Loading base model for policy and attaching SFT adapter...")
base = AutoModelForCausalLM.from_pretrained(BASE_PATH, local_files_only=True)
policy = PeftModel.from_pretrained(base, SFT_ADAPTER_PATH, local_files_only=True)
policy.to(DEVICE)

print("Loading reference model (base-only)...")
ref_base = AutoModelForCausalLM.from_pretrained(BASE_PATH, local_files_only=True)
ref = ref_base.to(DEVICE)
ref.eval()

# === Ensure adapter params are trainable ===
# Freeze all then enable LoRA-like params
for n, p in policy.named_parameters():
    p.requires_grad = False

enabled = 0
for n, p in policy.named_parameters():
    lname = n.lower()
    if any(k in lname for k in ("lora", "adapter", "lora_a", "lora_b", "lora_")):
        p.requires_grad = True
        enabled += p.numel()
trainable_total = sum(p.numel() for p in policy.parameters() if p.requires_grad)
print(f"Enabled approx {enabled:,} adapter params; trainable param total: {trainable_total:,}")

# Build optimizer
trainable_params = [p for p in policy.parameters() if p.requires_grad]
if len(trainable_params) == 0:
    raise RuntimeError("No trainable parameters found. Check adapter names and requires_grad flags.")
optimizer = torch.optim.AdamW(trainable_params, lr=LR)

# === Load a small batch from dataset ===
print("Loading dataset and sampling a small batch...")
ds = load_dataset("json", data_files=SAMPLE_JSON)["train"]
# pick first BATCH_SIZE examples that have required fields
prompts, chosens, rejects = [], [], []
i = 0
while len(prompts) < BATCH_SIZE and i < len(ds):
    ex = ds[i]
    i += 1
    if not isinstance(ex, dict):
        continue
    if "prompt" in ex and "chosen" in ex and "rejected" in ex:
        prompts.append(ensure_text(ex["prompt"], tokenizer))
        chosens.append(ensure_text(ex["chosen"], tokenizer))
        rejects.append(ensure_text(ex["rejected"], tokenizer))

if len(prompts) == 0:
    raise RuntimeError("No valid examples found in dataset.")

print(f"Sampled {len(prompts)} examples for debug batch.")

# Tokenize pairs
in_c, attn_c, labels_c = tokenize_pair_batch(tokenizer, prompts, chosens, PAD_ID)
in_r, attn_r, labels_r = tokenize_pair_batch(tokenizer, prompts, rejects, PAD_ID)

# Move policy to train mode for gradient test
policy.train()

# === Single forward/backward step with diagnostics ===
print("\n=== Single-step forward/backward debug ===")
# Print basic shapes
print("input shapes:", "chosen", in_c.shape, "rejected", in_r.shape, "labels", labels_c.shape)

# Forward chosen and rejected to get summed logprobs and raw logits
sum_c, logits_c = summed_logprob_from_model(in_c, attn_c, labels_c, policy, DEVICE)
sum_r, logits_r = summed_logprob_from_model(in_r, attn_r, labels_r, policy, DEVICE)

print("summed logp chosen:", sum_c.tolist())
print("summed logp rejected:", sum_r.tolist())
print("mean margin chosen - rejected:", float((sum_c - sum_r).mean().item()))

# NaN checks on logits
nan_logits_c = torch.isnan(logits_c).any().item()
nan_logits_r = torch.isnan(logits_r).any().item()
print("logits NaN for chosen:", nan_logits_c, "for rejected:", nan_logits_r)

# Check params for NaN before backward
any_param_nan_before = any(torch.isnan(p).any().item() for p in policy.parameters() if p.requires_grad)
print("any trainable param NaN before backward:", any_param_nan_before)

# Compute reward and DPO loss
# Use policy-only reward here (policy logps). If you want policy vs ref, compute ref logps similarly.
reward = sum_c - sum_r  # shape (B,)
# clamp reward to finite range to avoid extreme values
reward = torch.clamp(reward, min=-1e6, max=1e6)
loss = -torch.log(torch.sigmoid(BETA * reward)).mean()
print("computed loss:", float(loss.detach().cpu().item()))

# Backward with optional AMP
use_amp = USE_AMP and DEVICE.startswith("cuda")
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

optimizer.zero_grad()
if use_amp:
    with torch.cuda.amp.autocast():
        # recompute forward under autocast for gradient correctness
        sum_c_train, _ = summed_logprob_from_model(in_c, attn_c, labels_c, policy, DEVICE)
        sum_r_train, _ = summed_logprob_from_model(in_r, attn_r, labels_r, policy, DEVICE)
        reward_train = torch.clamp(sum_c_train - sum_r_train, min=-1e6, max=1e6)
        loss_train = -torch.log(torch.sigmoid(BETA * reward_train)).mean()
    scaler.scale(loss_train).backward()
    # gradient diagnostics
    scaler.unscale_(optimizer)
else:
    # compute gradients in fp32
    loss.backward()

# Gradient diagnostics
num_with_grad = 0
total_grad_norm_sq = 0.0
for n, p in policy.named_parameters():
    if p.requires_grad and p.grad is not None:
        num_with_grad += 1
        total_grad_norm_sq += (p.grad.detach() ** 2).sum().item()
grad_norm = math.sqrt(total_grad_norm_sq) if num_with_grad > 0 else 0.0
print("num params with grad:", num_with_grad, "grad_norm:", grad_norm)

# Print optimizer param group info
for i, g in enumerate(optimizer.param_groups):
    print(f"optimizer group {i} lr {g['lr']} num_params {sum(p.numel() for p in g['params'])}")

# Optional gradient clipping
torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad and p.grad is not None], max_norm=1.0)

# Step optimizer
if use_amp:
    scaler.step(optimizer)
    scaler.update()
else:
    optimizer.step()

# Check for NaNs in parameters after step
any_param_nan_after = any(torch.isnan(p).any().item() for p in policy.parameters() if p.requires_grad)
print("any trainable param NaN after step:", any_param_nan_after)

# Print a few parameter stats before/after by reloading a small sample of params
print("\nSample parameter statistics after step (first parameters):")
printed = 0
for n, p in policy.named_parameters():
    if p.requires_grad:
        mean, std, cnt = param_stats([p])
        print(f"  {n}: mean {mean:.6e} std {std:.6e} numel {cnt}")
        printed += 1
        if printed >= PRINT_TOP_PARAMS:
            break

# Final quick checks: recompute summed logprobs to see immediate effect
policy.eval()
with torch.no_grad():
    sum_c_after, _ = summed_logprob_from_model(in_c, attn_c, labels_c, policy, DEVICE)
    sum_r_after, _ = summed_logprob_from_model(in_r, attn_r, labels_r, policy, DEVICE)
print("\nPost-step summed logp chosen:", sum_c_after.tolist())
print("Post-step summed logp rejected:", sum_r_after.tolist())
print("Post-step mean margin:", float((sum_c_after - sum_r_after).mean().item()))

print("\nSingle-batch debug complete.")

