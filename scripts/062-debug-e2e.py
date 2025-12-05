#!/usr/bin/env python3
# dpo_debug_end_to_end.py
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

# === USER CONFIG: edit these paths to match your environment ===
BASE_PATH = "/scratch/aowais2/llama"                 # base model folder used for training
SFT_ADAPTER_PATH = "/scratch/aowais2/llama32_lora_sft"  # folder with adapter (adapter_config.json)
DATA_JSON = "/scratch/aowais2/data/dpo_train.json"   # dataset JSON to sample a batch
PROMPT_MAX = 256
ANSWER_MAX = 512
MAX_TOTAL = PROMPT_MAX + ANSWER_MAX
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BETA = 0.1  # DPO beta for single-step test
LR = 1e-5

# === Load tokenizer and models (local-only) ===
print("Loading tokenizer and models...")
tokenizer = AutoTokenizer.from_pretrained(BASE_PATH, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
PAD_ID = tokenizer.pad_token_id

# load base model then attach SFT adapter (force local files only)
base = AutoModelForCausalLM.from_pretrained(BASE_PATH, local_files_only=True)
policy = PeftModel.from_pretrained(base, SFT_ADAPTER_PATH, local_files_only=True)
policy.to(DEVICE)
policy.eval()

# load a reference model that is base-only (no adapter attached)
ref_base = AutoModelForCausalLM.from_pretrained(BASE_PATH, local_files_only=True)
ref_model = ref_base.to(DEVICE)
ref_model.eval()

print("Loaded policy (base + adapter) and ref (base-only). Device:", DEVICE)

# === Utility helpers ===
def tokenize_pairs(prompts, answers):
    texts = [p + "\n" + a for p, a in zip(prompts, answers)]
    enc = tokenizer(
        texts,
        truncation=True,
        padding="longest",
        max_length=MAX_TOTAL,
        return_tensors="pt",
    )
    prompt_enc = tokenizer(prompts, truncation=True, padding="longest", max_length=PROMPT_MAX, return_tensors="pt")
    prompt_lens = (prompt_enc["input_ids"] != PAD_ID).sum(dim=1)
    labels = enc["input_ids"].clone()
    for i, plen in enumerate(prompt_lens):
        labels[i, :plen] = -100
    return enc["input_ids"], enc["attention_mask"], labels

@torch.no_grad()
def compute_logprobs(input_ids, attention_mask, labels, model):
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    labels = labels.to(DEVICE)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    safe_labels = labels.clone()
    safe_labels[safe_labels == -100] = 0
    token_logps = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    valid_mask = (labels != -100).to(token_logps.dtype)
    token_logps = token_logps * valid_mask
    per_sample_logp = token_logps.sum(dim=1)
    return per_sample_logp.cpu()

def compute_logprobs_train(input_ids, attention_mask, labels, model):
    outputs = model(input_ids=input_ids.to(DEVICE), attention_mask=attention_mask.to(DEVICE))
    logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)
    safe_labels = labels.clone().to(DEVICE)
    safe_labels[safe_labels == -100] = 0
    token_logps = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    valid_mask = (labels.to(DEVICE) != -100).to(token_logps.dtype)
    token_logps = token_logps * valid_mask
    per_sample_logp = token_logps.sum(dim=1)
    return per_sample_logp

# === Ensure adapter params are trainable (enable LoRA params) ===
# Freeze all params first, then enable LoRA-like params by name heuristics.
for n, p in policy.named_parameters():
    p.requires_grad = False

enabled_count = 0
enabled_names = []
for n, p in policy.named_parameters():
    lname = n.lower()
    if any(k in lname for k in ("lora", "adapter", "lora_a", "lora_b", "lora_")):
        p.requires_grad = True
        enabled_count += p.numel()
        enabled_names.append(n)

trainable_total = sum(p.numel() for p in policy.parameters() if p.requires_grad)
print(f"Enabled approx {enabled_count:,} adapter params; trainable param total: {trainable_total:,}")
print("Sample enabled param names (first 40):", enabled_names[:40])

# === Load dataset and sample a small raw batch ===
print("Loading dataset and sampling a batch...")
ds = load_dataset("json", data_files=DATA_JSON)["train"]
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x, num_workers=0)
features = next(iter(dl))
prompts = [ex["prompt"] for ex in features]
chosen = [ex["chosen"] for ex in features]
rejected = [ex["rejected"] for ex in features]
print(f"Sampled {len(features)} examples.")

# === 1) Per-sample chosen vs rejected log-prob ===
print("\n=== 1) chosen vs rejected log-probs (policy) ===")
in_ids_c, attn_c, labels_c = tokenize_pairs(prompts, chosen)
in_ids_r, attn_r, labels_r = tokenize_pairs(prompts, rejected)
logp_c = compute_logprobs(in_ids_c, attn_c, labels_c, policy)
logp_r = compute_logprobs(in_ids_r, attn_r, labels_r, policy)
diffs = (logp_c - logp_r)
print("chosen logp:", logp_c.tolist())
print("rejected logp:", logp_r.tolist())
print("chosen - rejected diffs:", diffs.tolist())
print("mean diff:", float(diffs.mean()), "std diff:", float(diffs.std()))

# === 2) Policy vs ref on chosen continuations ===
print("\n=== 2) policy vs ref on chosen ===")
logp_c_ref = compute_logprobs(in_ids_c, attn_c, labels_c, ref_model)
print("policy chosen mean:", float(logp_c.mean()), "ref chosen mean:", float(logp_c_ref.mean()))
print("policy - ref diffs (sample):", (logp_c - logp_c_ref).tolist())

# === 3) Raw token lengths and content preview ===
print("\n=== 3) token lengths and previews ===")
for i, ex in enumerate(features):
    pids = tokenizer(ex["prompt"], truncation=True, max_length=PROMPT_MAX)["input_ids"]
    cids = tokenizer(ex["prompt"] + "\n" + ex["chosen"], truncation=True, max_length=MAX_TOTAL)["input_ids"]
    rids = tokenizer(ex["prompt"] + "\n" + ex["rejected"], truncation=True, max_length=MAX_TOTAL)["input_ids"]
    print(f"sample {i}: prompt_len={len(pids)}, chosen_total_len={len(cids)}, rejected_total_len={len(rids)}")
    print("  chosen preview:", ex["chosen"][:200].replace("\n", " "))
    print("  rejected preview:", ex["rejected"][:200].replace("\n", " "))
    print("---")

# === 4) Labels and attention mask sanity for first sample ===
print("\n=== 4) labels and masks (first sample) ===")
i = 0
in_ids_c1, attn_c1, labels_c1 = tokenize_pairs([prompts[i]], [chosen[i]])
in_ids_r1, attn_r1, labels_r1 = tokenize_pairs([prompts[i]], [rejected[i]])
print("chosen input_ids:", in_ids_c1[0].tolist())
print("chosen attention_mask:", attn_c1[0].tolist())
print("chosen labels (prompt masked -> -100):", labels_c1[0].tolist())
prompt_enc = tokenizer([prompts[i]], truncation=True, padding="longest", max_length=PROMPT_MAX, return_tensors="pt")
prompt_len = (prompt_enc["input_ids"] != PAD_ID).sum().item()
print("expected prompt_len:", prompt_len)
print("labels first prompt_len tokens (should be -100):", labels_c1[0][:prompt_len].tolist())

# === 5) Optimizer and gradients single-step test ===
print("\n=== 5) optimizer & gradients single-step test ===")
trainable_params = [p for p in policy.parameters() if p.requires_grad]
print("trainable param count:", sum(p.numel() for p in trainable_params))
if len(trainable_params) == 0:
    print("ERROR: no trainable parameters. Ensure adapter params exist and names matched the heuristic.")
else:
    optimizer = torch.optim.AdamW(trainable_params, lr=LR)
    policy.train()
    logp_c_train = compute_logprobs_train(in_ids_c, attn_c, labels_c, policy)
    logp_r_train = compute_logprobs_train(in_ids_r, attn_r, labels_r, policy)
    reward = logp_c_train - logp_r_train
    loss = -torch.log(torch.sigmoid(BETA * reward)).mean()
    print("loss before step:", float(loss.detach().cpu()))
    optimizer.zero_grad()
    loss.backward()
    total_grad_norm = 0.0
    grad_count = 0
    for p in trainable_params:
        if p.grad is not None:
            total_grad_norm += (p.grad.detach() ** 2).sum().item()
            grad_count += 1
    total_grad_norm = total_grad_norm ** 0.5 if grad_count > 0 else 0.0
    print("grad norm before step:", total_grad_norm, "num params with grad:", grad_count)
    optimizer.step()
    policy.eval()
    with torch.no_grad():
        logp_c_after = compute_logprobs(in_ids_c, attn_c, labels_c, policy)
        logp_r_after = compute_logprobs(in_ids_r, attn_r, labels_r, policy)
        reward_after = logp_c_after - logp_r_after
        loss_after = -torch.log(torch.sigmoid(BETA * reward_after)).mean()
    print("loss after one optimizer.step():", float(loss_after))

print("\nAll diagnostics complete. If trainable param count is 0 or policy==ref, follow the printed hints.")

