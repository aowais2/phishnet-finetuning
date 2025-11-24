import json

# Input file: your DPO dataset
input_file = "/home/aowais2/Downloads/phishnet-finetuning/data/dpo_dataset.json"
# Output file: new SFT dataset
output_file = "/home/aowais2/Downloads/phishnet-finetuning/data/sft.json"

sft_data = []

# Read the DPO JSON (assumes it's a list of dicts or JSONL-style lines)
with open(input_file, "r") as f:
    try:
        # Case 1: full JSON array
        dpo_data = json.load(f)
    except json.JSONDecodeError:
        # Case 2: JSONL (one JSON object per line)
        dpo_data = [json.loads(line) for line in f if line.strip()]

# Convert each entry: keep prompt + chosen
for entry in dpo_data:
    sft_entry = {
        "prompt": entry["prompt"],
        "response": entry["chosen"]
    }
    sft_data.append(sft_entry)

# Save as JSON array
with open(output_file, "w") as f:
    json.dump(sft_data, f, indent=2)

print(f"Converted {len(sft_data)} examples to SFT format → {output_file}")

