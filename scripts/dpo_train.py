import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig

BASE_CKPT = os.environ.get("BASE_CKPT", "outputs/lora-phish-classifier")

tokenizer = AutoTokenizer.from_pretrained(BASE_CKPT, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

ds = load_dataset("json", data_files={"train": "data/dpo_train.jsonl", "validation": "data/dpo_val.jsonl"})

model = AutoModelForCausalLM.from_pretrained(BASE_CKPT, device_map="auto", torch_dtype="auto")

cfg = DPOConfig(
    output_dir="outputs/dpo-phish-classifier",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    num_train_epochs=2,
    beta=0.1,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=200,
    save_steps=200,
    save_total_limit=2,
)

trainer = DPOTrainer(
    model=model,
    args=cfg,
    tokenizer=tokenizer,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
)

trainer.train()
trainer.save_model("outputs/dpo-phish-classifier")
