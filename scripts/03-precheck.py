import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# Load tokenizer + model (GPU required)
model_path = "/scratch/aowais2/llama"
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model directly (no quantization, no bitsandbytes)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto"   # automatically place on GPU
)

# Configure LoRA
lora_config = LoraConfig(
    r=8,                 # rank
    lora_alpha=16,       # scaling alpha
    lora_dropout=0.05,   # regularization
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM"
)

# Attach adapters to targeted modules
model = get_peft_model(model, lora_config)

# Freeze base weights, ensure only LoRA params are trainable
trainable, total = 0, 0
for n, p in model.named_parameters():
    total += p.numel()
    if p.requires_grad:
        trainable += p.numel()

print(f"Total params: {total:,}")
print(f"Trainable params (LoRA only): {trainable:,}")
print(f"Trainable ratio: {100 * trainable / total:.4f}%")

# Find a typical attention block and show what got wrapped
wrapped = defaultdict(list)
for name, module in model.named_modules():
    if any(t in name for t in ["q_proj", "v_proj"]):
        wrapped[type(module).__name__].append(name)

for t, names in wrapped.items():
    print(f"{t}:")
    for n in names[:6]:
        print("  ", n)

# Optional: demonstrate the modified forward with a small input
text = "Explain phishing indicators succinctly."
inputs = tokenizer(text, return_tensors="pt").to("cuda")
with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

