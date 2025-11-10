# pip install transformers datasets peft accelerate evaluate
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from peft import LoraConfig, get_peft_model

MODEL_NAME = "meta-llama/Llama-3-8b-Instruct"  # example small model

# 1) Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# 2) Load dataset
ds = load_dataset("json", data_files={"train": "data/train.jsonl", "validation": "data/val.jsonl"})

def format_example(ex):
    prompt = (
        "Classify the following email as Phishing or Legitimate and briefly explain your reasoning.\n\n"
        f"Email:\n{ex['text']}\n\n"
        "Answer: Label: <Phishing|Legitimate>; Reason: <one sentence>."
    )
    return {"input_text": prompt}

ds = ds.map(format_example)

def tokenize(batch):
    return tokenizer(batch["input_text"], max_length=1024, truncation=True, padding="max_length")

tokenized = ds.map(tokenize, batched=True, remove_columns=ds["train"].column_names)

# 3) Load model (no quantization, pure LoRA)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

# 4) Attach LoRA adapters
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"]
)
model = get_peft_model(model, lora_cfg)

model.print_trainable_parameters()

# 5) Training settings
args = TrainingArguments(
    output_dir="outputs/lora-phish-classifier",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    fp16=True,  # safe for GPU; remove if CPU-only
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_total_limit=2,
)

# 6) Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
)

trainer.train()
trainer.save_model("outputs/lora-phish-classifier")
