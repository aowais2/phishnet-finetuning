import pandas as pd
from pathlib import Path
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

def load_and_clean_csv(csv_path: str) -> pd.DataFrame:
    """
    Load the CSV and keep only relevant columns.
    Map numeric labels to 'phishing' or 'benign'.
    """
    df = pd.read_csv(csv_path, encoding="latin1")

    # Keep only text + label
    df = df[['text', 'label']].copy()

    # Map numeric labels to categories
    label_map = {
        1: 'phishing',   # spam/phishing
        0: 'benign'      # safe
    }
    df['label'] = df['label'].map(label_map)

    # Drop rows with unknown labels
    df = df.dropna(subset=['label'])

    return df

def build_dpo_pair(row) -> dict:
    """
    Convert a single row into a DPO-style preference pair.
    """
    text, label = row['text'], row['label']

    if label == 'phishing':
        chosen = "This is a phishing attempt."
        rejected = "This is a safe message."
    else:  # benign
        chosen = "This is a safe message."
        rejected = "This is a phishing attempt."

    return {
        "prompt": text,
        "chosen": chosen,
        "rejected": rejected
    }

def build_dpo_pairs_parallel(df: pd.DataFrame, workers = None) -> list:
    """
    Convert rows into DPO pairs using multiple workers.
    """
    if workers is None:
        workers = multiprocessing.cpu_count()
    rows = df.to_dict(orient="records")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        dpo_data = list(executor.map(build_dpo_pair, rows))
    return dpo_data

def save_to_json(dpo_data: list, output_path: str):
    """
    Save the DPO pairs to JSON file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dpo_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    input_csv = "/home/aowais2/Downloads/phishnet-finetuning/data/master_email_dataset_final.csv"
    output_json = "/home/aowais2/Downloads/phishnet-finetuning/data/dpo_dataset.json"

    df = load_and_clean_csv(input_csv)
    dpo_data = build_dpo_pairs_parallel(df)
    save_to_json(dpo_data, output_json)

    print(f"Saved {len(dpo_data)} DPO pairs to {output_json}")

