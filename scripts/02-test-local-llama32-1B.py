from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "/scratch/aowais2/llama"

# Load tokenizer and model (from cache since you've already downloaded)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,   # use half precision if supported
    device_map="cuda"             # keep it on CPU for login node
)

# Simple prompt
prompt = "What are the categories of phishing emails?"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=150)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

