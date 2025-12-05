import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model

# 1. Load tokenizer + model (full precision, no bitsandbytes)
model_path = "/scratch/aowais2/llama"
tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",        # automatically place on GPU
    torch_dtype=torch.float16 # efficient precision without bitsandbytes
)

# 2. Attach LoRA adapters
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj","v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# 3. Load your single JSON dataset
dataset = load_dataset("json", data_files="/scratch/aowais2/data/sft.json")["train"]

# 4. Split into train/validation (90/10 split)
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
valid_dataset = split_dataset["test"]


def tokenize(batch):
    texts = [p + "\n" + r for p, r in zip(batch["prompt"], batch["response"])]
    tokens = tokenizer(texts, truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)
valid_dataset = valid_dataset.map(tokenize, batched=True, remove_columns=valid_dataset.column_names)

# 6. Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# 7. Training arguments
training_args = TrainingArguments(
    output_dir="/scratch/aowais2/llama32_lora_4epochs",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=4,
    fp16=True,
    logging_steps=50,
    save_steps=500,
    eval_steps=200,
    report_to="none"   # disables wandb/tensorboard auto-logging
)

# 8. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 9. Train + save
trainer.train()
model.save_pretrained("/scratch/aowais2/llama32_lora")
tokenizer.save_pretrained("/scratch/aowais2/llama32_lora")
