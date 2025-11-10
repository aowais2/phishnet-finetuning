import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CKPT = "outputs/dpo-phish-classifier"  # or "outputs/lora-phish-classifier"

tokenizer = AutoTokenizer.from_pretrained(CKPT)
model = AutoModelForCausalLM.from_pretrained(CKPT, device_map="auto")

def classify(email_text):
    prompt = (
        "Classify the following email as Phishing or Legitimate and briefly explain your reasoning.\n\n"
        f"Email:\n{email_text}\n\n"
        "Answer: Label: <Phishing|Legitimate>; Reason: <one sentence>."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, temperature=0.2, do_sample=False)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    # Simple parsing; customize as needed
    label = "Phishing" if "Label: Phishing" in text else ("Legitimate" if "Label: Legitimate" in text else "Unknown")
    reason_start = text.find("Reason:")
    reason = text[reason_start+7:].strip() if reason_start != -1 else ""
    return {"label": label, "reason": reason, "raw": text}

if __name__ == "__main__":
    email = "Dear Student, your OSAP deposit failed. Send your SIN and bank info to osap.verify@ontario-aid.org."
    print(json.dumps(classify(email), indent=2))
