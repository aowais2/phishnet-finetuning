#!/usr/bin/env python3
# dpo_debug_all_checks.py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset

# === User config: adjust paths and sizes ===
BASE_PATH = "/scratch/aowais2/llama"                 # base model path
SFT_ADAPTER_PATH = "/scratch/aowais2/llama32_lora_sft"  # your SFT adapter folder
TEST_OR_TRAIN_JSON = "/scratch/aowais2/data/dpo_train.json"  # dataset used for sampling a batch
PROMPT_MAX = 256
ANSWER_MAX = 512
MAX_TOTAL = PROMPT_MAX + ANSWER_MAX
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BETA = 0.1  # DPO beta used for the single-step loss test

# === Load tokenizer and policy (base + SFT adapter) ===
print("Loading tokenizer and policy (base + SFT adapter)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
PAD_ID = tokenizer.pad_token_id

base = AutoModelForCausalLM.from_pretrained(BASE_PATH, device_map="auto")
policy = PeftModel.from_pretrained(base, SFT_ADAPTER_PATH)
policy.to(DEVICE)
policy.eval()

# Optional: load a reference model for comparisons (here we attach same SFT adapter by default)
ref_base = AutoModelForCausalLM.from_pretrained(BASE_PATH, device_map="auto")
ref_model = PeftModel.from_pretrained(ref_base, SFT_ADAPTER_PATH)
ref_model.to(DEVICE)
ref_model.eval()

print("Model and tokenizer loaded. Device:", DEVICE)

# === Load a small portion of the dataset (or full dataset if you prefer) ===
print("Loading dataset to sample a raw batch...")
ds = load_dataset("json", data_files=TEST_OR_TRAIN_JSON)["train"]

# If your dataset is already preprocessed (train_tok), you can skip tokenization here and adapt the code.
# We'll assume raw dataset has fields: 'prompt', 'chosen', 'rejected'.
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: x, num_workers=0)
features = next(iter(dl))  # list of dicts (raw examples)
print(f"Sampled {len(features)} examples from dataset for diagnostics.")

# === Helpers: tokenization and log-prob computation ===
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
    logits = outputs.logits  # (B, L, V)
    log_probs = F.log_softmax(logits, dim=-1)  # (B, L, V)
    safe_labels = labels.clone()
    safe_labels[safe_labels == -100] = 0
    token_logps = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)  # (B, L)
    valid_mask = (labels != -100).to(token_logps.dtype)
    token_logps = token_logps * valid_mask
    per_sample_logp = token_logps.sum(dim=1)  # (B,)
    return per_sample_logp.cpu()

# Training-mode logprobs (with gradients) for single-step test
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

# === Prepare prompts/chosen/rejected lists from sampled features ===
prompts = [ex["prompt"] for ex in features]
chosen = [ex["chosen"] for ex in features]
rejected = [ex["rejected"] for ex in features]

# === 1) Check per-sample rewards / log-prob difference ===
print("\n=== 1) Per-sample chosen vs rejected log-prob ===")
in_ids_c, attn_c, labels_c = tokenize_pairs(prompts, chosen)
in_ids_r, attn_r, labels_r = tokenize_pairs(prompts, rejected)
logp_c = compute_logprobs(in_ids_c, attn_c, labels_c, policy)
logp_r = compute_logprobs(in_ids_r, attn_r, labels_r, policy)
diffs = (logp_c - logp_r)
print("chosen logp:", logp_c.tolist())
print("rejected logp:", logp_r.tolist())
print("chosen - rejected diffs:", diffs.tolist())
print("mean diff:", float(diffs.mean()), "std diff:", float(diffs.std()))

# === 2) Check reference vs policy outputs (policy vs ref on chosen) ===
print("\n=== 2) Policy vs Ref on chosen continuations ===")
logp_c_ref = compute_logprobs(in_ids_c, attn_c, labels_c, ref_model)
print("policy chosen mean:", float(logp_c.mean()), "ref chosen mean:", float(logp_c_ref.mean()))
print("policy - ref diffs (sample):", (logp_c - logp_c_ref).tolist())

# === 3) Inspect raw token lengths and content for the batch ===
print("\n=== 3) Raw token lengths and content preview ===")
for i, ex in enumerate(features):
    pids = tokenizer(ex["prompt"], truncation=True, max_length=PROMPT_MAX)["input_ids"]
    cids = tokenizer(ex["prompt"] + "\n" + ex["chosen"], truncation=True, max_length=MAX_TOTAL)["input_ids"]
    rids = tokenizer(ex["prompt"] + "\n" + ex["rejected"], truncation=True, max_length=MAX_TOTAL)["input_ids"]
    print(f"sample {i}: prompt_len={len(pids)}, chosen_total_len={len(cids)}, rejected_total_len={len(rids)}")
    print("  chosen preview:", ex["chosen"][:200].replace("\n", " "))
    print("  rejected preview:", ex["rejected"][:200].replace("\n", " "))
    print("---")

# === 4) Verify labels/masks are correct for first sample ===
print("\n=== 4) Labels and attention mask sanity (first sample) ===")
i = 0
in_ids_c1, attn_c1, labels_c1 = tokenize_pairs([prompts[i]], [chosen[i]])
in_ids_r1, attn_r1, labels_r1 = tokenize_pairs([prompts[i]], [rejected[i]])
print("chosen input_ids:", in_ids_c1[0].tolist())
print("chosen attention_mask:", attn_c1[0].tolist())
print("chosen labels (masked prompt -> -100):", labels_c1[0].tolist())
print("rejected input_ids:", in_ids_r1[0].tolist())
print("rejected attention_mask:", attn_r1[0].tolist())
print("rejected labels (masked prompt -> -100):", labels_r1[0].tolist())
prompt_enc = tokenizer([prompts[i]], truncation=True, padding="longest", max_length=PROMPT_MAX, return_tensors="pt")
prompt_len = (prompt_enc["input_ids"] != PAD_ID).sum().item()
print("expected prompt_len:", prompt_len)
print("labels first prompt_len tokens (should be -100):", labels_c1[0][:prompt_len].tolist())

# === 5) Check optimizer and gradients (single-step update test) ===
print("\n=== 5) Optimizer and gradients single-step test ===")
trainable_params = [p for p in policy.parameters() if p.requires_grad]
print("trainable param count:", sum(p.numel() for p in trainable_params))
if len(trainable_params) == 0:
    print("WARNING: no trainable parameters found. Ensure adapter params are not frozen.")
else:
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-5)
    policy.train()
    # compute train-mode logprobs (with gradients)
    logp_c_train = compute_logprobs_train(in_ids_c, attn_c, labels_c, policy)
    logp_r_train = compute_logprobs_train(in_ids_r, attn_r, labels_r, policy)
    reward = logp_c_train - logp_r_train  # (B,)
    loss = -torch.log(torch.sigmoid(BETA * reward)).mean()
    print("loss before step:", float(loss.detach().cpu()))
    optimizer.zero_grad()
    loss.backward()
    # gradient diagnostics
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
    # recompute loss after one step (no_grad)
    with torch.no_grad():
        logp_c_after = compute_logprobs(in_ids_c, attn_c, labels_c, policy)
        logp_r_after = compute_logprobs(in_ids_r, attn_r, labels_r, policy)
        reward_after = logp_c_after - logp_r_after
        loss_after = -torch.log(torch.sigmoid(BETA * reward_after)).mean()
    print("loss after one optimizer.step():", float(loss_after))

print("\nDiagnostics complete. Paste outputs here if you want interpretation or next steps.")

