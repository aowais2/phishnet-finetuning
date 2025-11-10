# Tiny LLM fine tuned with RL

## Problem Statement

Small LMs do not perform well on phishing detection tasks out of the box [1]. Phishing detection using smaller LMs has been proposed before [1], however these methods only utilize simpler methods without finetuning, such as prompt engineering.

We consider small language models (LMs) for finetuning with Reinforcement Learning (RL) methods. Small language models in this context refer to 3B-8B paramater models.

To remedy this, finetuning methods such as Low Rank Adaptation (LORA) are proposed. This, along with minimal further finetuning with RL methods using Direct Preference Optimization (DPO) [2] will be utilized. This is in contrast to other time and data expensive methods such as RLHF (RL from Human Feedback).

Other methods were considered, such as Retrieval Augmented Generation (RAG). However, these were deemed insufficient because the problem statement is defined as a classification problem (phishing vs regular email classification), not a knowledge retrieval issue.

Further, we propose the utilization of DPO as opposed to other RL methods such as PPO or GRPO because GRPO is mainly used for complex reasoning improvements, and adds unnecessary complexity for a binary classification task.

The following pipeline is proposed for finetuning a small LM for phishing detection:

1.   Supervised finetuning with LoRA on phishing vs legitimate labels.
2.   Direct Preference Optimization (DPO)
3.   Evaluation of the model

[1] Lin, Zijie, Zikang Liu, and Hanbo Fan. "Improving Phishing Email Detection Performance of Small Large Language Models." arXiv preprint arXiv:2505.00034 (2025).

[2] Rafailov, Rafael, et al. "Direct preference optimization: Your language model is secretly a reward model." Advances in neural information processing systems 36 (2023): 53728-53741.



## Methodology

This workflow fine-tunes a small, instruction-tuned decoder model with parameter-efficient adapters (LoRA/QLoRA) for supervised phishing classification, then optionally applies DPO to improve preference-aligned outputs (clearer rationales, better calibration). LoRA/QLoRA drastically reduce trainable parameters and memory needs, enabling efficient adaptation on modest hardware. DPO aligns model outputs using human preferences without reward models or online rollouts, making it lighter than PPO-style RLHF.

Data schema and prompting

Labels:

Primary: phishing vs. legitimate.

Secondary: impersonated entity, lure type (urgency, fear, refund), channel (fake portal, eTransfer), indicators (attachments, shortened links).

Input fields:

text: raw email body (strip HTML, preserve headers if relevant).

label: 0 = legitimate, 1 = phishing.

rationale (optional): brief human-written reason for classification.

Prompt template (SFT):

“Classify the following email as Phishing or Legitimate and briefly explain your reasoning.\n\nEmail:\n{email_text}\n\nAnswer with label and a one-sentence rationale.”

Tip: Parameter-efficient LoRA adapters typically train only 0.3%–0.6% of parameters in transformer layers, keeping base weights frozen while achieving near full fine-tuning performance.

Baseline supervised fine-tuning with LoRA/QLoRA

Model and libraries

Model: small instruction-tuned decoder LLM (3–7B).

Why LoRA/QLoRA: dramatically lower VRAM and compute vs full fine-tuning while preserving accuracy; QLoRA quantizes the base model to 4-bit to further reduce memory.


### Example code: supervised LoRA fine-tune (Hugging Face Transformers + PEFT)

!pip install transformers datasets peft accelerate bitsandbytes evaluate

```python
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
'''

1) Load base model (decoder-only) and tokenizer

```python
model_name = "meta-llama/Llama-3-8b-Instruct"  # example; pick a small instruction-tuned model you can run
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
'''
2) Prepare dataset: fields -> {"text": email_text, "label": 0/1, "rationale": optional}

```python
ds = load_dataset("json", data_files={"train": "train.jsonl", "validation": "val.jsonl"})

def format_example(example):
    prompt = (
        "Classify the following email as Phishing or Legitimate and briefly explain your reasoning.\n\n"
        f"Email:\n{example['text']}\n\n"
        "Answer in the format: Label: <Phishing|Legitimate>; Reason: <one sentence>."
    )
    return {"input_text": prompt, "label": example["label"]}

ds = ds.map(format_example)
'''

3) Tokenize

```python
max_len = 2048
def tokenize(batch):
    return tokenizer(
        batch["input_text"], max_length=max_len, truncation=True, padding="max_length"
    )

tokenized = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)
'''

4) Load model and attach LoRA adapters
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

lora_config = LoraConfig(
    r=16,                # rank
    lora_alpha=32,       # scaling
    lora_dropout=0.05,   # dropout
    target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"]  # typical LLM layers
)
model = get_peft_model(model, lora_config)
'''

Optional: Freeze base weights is handled by PEFT; only adapter params will be trainable
```python
model.print_trainable_parameters()
'''

5) Training settings

```python
args = TrainingArguments(
    output_dir="outputs/lora-phish-classifier",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    fp16=True,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_total_limit=2,
)
'''

6) Trainer
```python
data_collator = DataCollatorWithPadding(tokenizer)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("outputs/lora-phish-classifier")
'''

Sources: Overview and benefits of LoRA/QLoRA as parameter-efficient fine-tuning approaches; empirical evidence of LoRA’s effectiveness for phishing/malicious URL detection tasks.

### Optional preference tuning with DPO

DPO optimizes model outputs directly from preference pairs (preferred vs. dispreferred) and requires no reward model or online rollouts, which simplifies alignment and reduces compute compared to RLHF.

Data format for DPO

Fields: prompt, chosen_response (preferred), rejected_response (less preferred).

Example: For the same email, pair a correct, concise classification+rationale vs. a vague or incorrect one.

Example code: DPO training (TRL-style)

'''
pip install trl transformers datasets peft accelerate bitsandbytes
'''

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import DPOTrainer, DPOConfig

model_name = "outputs/lora-phish-classifier"  # start from your SFT LoRA checkpoint
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

Dataset with fields: {"prompt": str, "chosen": str, "rejected": str}

dpo_ds = load_dataset("json", data_files={"train": "dpo_train.jsonl", "validation": "dpo_val.jsonl"})

def tokenize_batch(batch):
    return {
        "prompt": batch["prompt"],
        "chosen": batch["chosen"],
        "rejected": batch["rejected"]
    }

dpo_ds = dpo_ds.map(tokenize_batch, batched=True)
'''

Load model with LoRA adapters active

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
'''

DPO configuration

```python
dpo_config = DPOConfig(
    output_dir="outputs/dpo-phish-classifier",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    num_train_epochs=2,
    beta=0.1,               # preference strength
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_total_limit=2,
)
'''

Trainer

```python
trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    tokenizer=tokenizer,
    train_dataset=dpo_ds["train"],
    eval_dataset=dpo_ds["validation"],
)

trainer.train()
trainer.save_model("outputs/dpo-phish-classifier")
'''

References: Official DPO implementation and technical descriptions demonstrating offline preference optimization without reward models and with lighter infrastructure than RLHF.

Evaluation and deployment

### Metrics:

- Accuracy/F1 (macro): class balance.
- AUROC: threshold-insensitive separability.
- Calibration: Brier score, ECE; add abstain threshold for human review.

Stress tests:

- Adversarial text: obfuscated URLs, unicode confusables, homograph domains, fake university portals.
- Domain shift: new lures (housing, OSAP, parking) and benign emails with similar keywords.

Inference prompt:

“Classify this email as Phishing or Legitimate and provide one concise reason:\n\nEmail:\n{email}\n\nReturn JSON: {‘label’: ‘Phishing|Legitimate’, ‘reason’: ‘...’}.”

Note: LoRA adapters enable task-specific swaps on a single base model; this is practical for maintaining separate classifiers or styles without retraining the base network.

Practical tips

Label quality: Balanced classes, hard negatives, and clear rationales improve DPO signal.

Adapter management: Save and version adapters per task; they’re small and portable.

When to skip DPO: If explanations aren’t required and your SFT accuracy is strong, DPO is optional. Prefer DPO when you want consistent, concise rationales and better calibration.

Evidence of LoRA’s effectiveness in phishing/malicious URL tasks and its dramatic reduction in trainable parameters supports this setup for efficient training on modest hardware. DPO sources confirm its simplicity and suitability for preference alignment without the overhead of reward modeling and online RLHF.

### References (8)

Fine-Tuning using LoRA and QLoRA - GeeksforGeeks. https://www.geeksforgeeks.org/deep-learning/fine-tuning-using-lora-and-qlora/

LoRA Fine-Tuning Tutorial: Reduce GPU Memory Usage by 90% in 2025. https://markaicode.com/lora-fine-tuning-tutorial-reduce-gpu-memory/

DPO: Direct Preference Optimization - GitHub. https://github.com/eric-mitchell/direct-preference-optimization

Direct preference optimization - Azure OpenAI | Microsoft Learn. https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/fine-tuning-direct-preference-optimization

Direct Preference Optimization: A Technical Deep Dive. https://www.together.ai/blog/direct-preference-optimization

Direct Preference Optimization: A New RLHF Approach. https://web.stanford.edu/class/cs234/CS234Spr2024/slides/dpo_slides.pdf

PhishURLDetect: A parameter efficient fine-tuning of LLMs using LoRA .... https://dl.acm.org/doi/epdf/10.1145/3700838.3703658

Direct Preference Optimization (DPO) - Open Instruct. https://allenai.github.io/open-instruct/algorithms/dpo/


