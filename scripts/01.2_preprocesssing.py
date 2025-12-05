import os
import re
import quopri
import multiprocessing
import warnings
import pandas as pd
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from tqdm.contrib.concurrent import process_map

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# =========================
# Configuration
# =========================
INPUT_FILE = "/scratch/aowais2/data/test.csv"
OUTPUT_FILE = "/scratch/aowais2/data/dpo_test.csv"

MIN_TOKEN_LENGTH = 5
MAX_TOKEN_LENGTH = 2000
NUM_WORKERS = multiprocessing.cpu_count()

# =========================
# Text Cleaning
# =========================
def clean_email_text(text: object) -> str:
    """Gentle cleaning for transformer input (preserve case/punctuation)."""
    if not isinstance(text, str):
        return ""

    # Fix encoding artifacts
    try:
        text_bytes = text.encode("latin-1", errors="ignore")
        decoded_bytes = quopri.decodestring(text_bytes)
        text = decoded_bytes.decode("utf-8", errors="ignore")
    except Exception:
        pass

    # Strip HTML
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")

    # Replace URLs and emails
    text = re.sub(r"(https?://\S+|www\.\S+)", " [URL] ", text)
    text = re.sub(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", " [EMAIL] ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

# =========================
# Pipeline
# =========================
def main():
    print("=== DPO Preprocessing Pipeline ===")
    print(f"Loading: {INPUT_FILE}")

    try:
        df = pd.read_csv(INPUT_FILE)
    except Exception as e:
        print(f"Error loading {INPUT_FILE}: {e}")
        return

    # Drop NaNs
    df = df.dropna(subset=["text"]).copy()

    # Clean text
    print(f"Cleaning text using {NUM_WORKERS} workers...")
    df["text"] = df["text"].fillna("")
    results = process_map(
        clean_email_text,
        df["text"],
        max_workers=NUM_WORKERS,
        chunksize=500,
        desc="Cleaning",
    )
    df["text"] = results

    # Drop empty after cleaning
    before = len(df)
    df = df[df["text"].str.strip().str.len() > 0]
    print(f"Dropped {before - len(df)} rows that became empty after cleaning.")

    # Deduplicate
    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first")
    print(f"Removed {before - len(df)} duplicate rows.")

    # Length filter
    df["token_count"] = df["text"].str.split().str.len()
    before = len(df)
    df = df[(df["token_count"] >= MIN_TOKEN_LENGTH) & (df["token_count"] <= MAX_TOKEN_LENGTH)]
    print(f"Removed {before - len(df)} rows outside length bounds.")
    df = df.drop(columns=["token_count"])

    # Save
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved cleaned dataset: {OUTPUT_FILE} ({len(df)} rows)")

if __name__ == "__main__":
    main()

