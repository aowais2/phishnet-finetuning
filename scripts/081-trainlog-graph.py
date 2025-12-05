#!/usr/bin/env python3
# visualize_train_log_simple.py
# Parse a training log file without raw regex literals and produce plots.

import os
import json
import argparse
import time
from typing import List, Dict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", context="talk", rc={"figure.figsize": (12, 6)})

def try_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def extract_after(token, line):
    # return substring after token or None
    idx = line.find(token)
    if idx == -1:
        return None
    return line[idx + len(token):].strip()

def parse_line_step(line):
    # expects: "[ts] step 200 epoch 1 loss 6.8875e-01 pref_rate 0.625 avg_margin 0.0887"
    if " step " not in line or " loss " not in line:
        return None
    try:
        parts = line.split()
        # find indices
        step_i = parts.index("step") + 1
        epoch_i = parts.index("epoch") + 1
        loss_i = parts.index("loss") + 1
        pref_i = parts.index("pref_rate") + 1
        avg_i = parts.index("avg_margin") + 1
        return {
            "step": int(parts[step_i]),
            "epoch": int(parts[epoch_i]),
            "loss": try_float(parts[loss_i]),
            "pref_rate": try_float(parts[pref_i]),
            "avg_margin": try_float(parts[avg_i]),
            "ts": parts[0].strip("[]")
        }
    except Exception:
        return None

def parse_line_quick(line):
    # expects: "[ts] Quick VAL step 200 acc 0.4550 tie 0.0800 mean_margin 0.0188"
    if "Quick VAL step" not in line or "mean_margin" not in line:
        return None
    try:
        parts = line.split()
        step_i = parts.index("step") + 1
        acc_i = parts.index("acc") + 1
        tie_i = parts.index("tie") + 1
        mm_i = parts.index("mean_margin") + 1
        return {
            "step": int(parts[step_i]),
            "acc": try_float(parts[acc_i]),
            "tie": try_float(parts[tie_i]),
            "mean_margin": try_float(parts[mm_i]),
            "ts": parts[0].strip("[]")
        }
    except Exception:
        return None

def parse_line_full(line):
    # expects: "[ts] Epoch 1 FULL VAL acc 0.9150 tie 0.0667 mean_margin 81.2737"
    if "FULL VAL acc" not in line or "mean_margin" not in line:
        return None
    try:
        parts = line.split()
        epoch_i = parts.index("Epoch") + 1
        acc_i = parts.index("acc") + 1
        tie_i = parts.index("tie") + 1
        mm_i = parts.index("mean_margin") + 1
        return {
            "epoch": int(parts[epoch_i]),
            "acc": try_float(parts[acc_i]),
            "tie": try_float(parts[tie_i]),
            "mean_margin": try_float(parts[mm_i]),
            "ts": parts[0].strip("[]")
        }
    except Exception:
        return None

def parse_misc(line, parsed):
    if "Saved best adapter to" in line:
        parsed["saved_best"] = line.split("Saved best adapter to",1)[1].strip()
    if "Saved checkpoint" in line:
        parsed.setdefault("saved_checkpoints", []).append(line.split("Saved checkpoint",1)[1].strip())
    if "Trainable params:" in line:
        try:
            parsed["trainable_params"] = int(line.split("Trainable params:")[1].strip())
        except Exception:
            pass
    if "Train examples:" in line and "Val examples:" in line:
        try:
            after = line.split("Train examples:")[1]
            train_str = after.split("Val examples:")[0].strip()
            val_str = after.split("Val examples:")[1].strip()
            parsed["dataset_counts"] = {"train": int(train_str.split()[0]), "val": int(val_str.split()[0])}
        except Exception:
            pass

def parse_log_file(path):
    train_rows: List[Dict] = []
    quick_rows: List[Dict] = []
    full_rows: List[Dict] = []
    parsed = {"saved_checkpoints": [], "saved_best": None, "trainable_params": None, "dataset_counts": None}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            s = parse_line_step(line)
            if s:
                train_rows.append(s); continue
            q = parse_line_quick(line)
            if q:
                quick_rows.append(q); continue
            fv = parse_line_full(line)
            if fv:
                full_rows.append(fv); continue
            parse_misc(line, parsed)
    return {
        "train": pd.DataFrame(train_rows).sort_values("step").reset_index(drop=True) if train_rows else pd.DataFrame(),
        "quick": pd.DataFrame(quick_rows).sort_values("step").reset_index(drop=True) if quick_rows else pd.DataFrame(),
        "full": pd.DataFrame(full_rows).sort_values("epoch").reset_index(drop=True) if full_rows else pd.DataFrame(),
        **parsed
    }

def plot_training(df, out_dir):
    if df.empty:
        return
    fig, ax1 = plt.subplots(figsize=(14,6))
    ax1.plot(df["step"], df["loss"], color="C0", label="loss")
    ax1.set_xlabel("global step"); ax1.set_ylabel("loss", color="C0"); ax1.tick_params(axis='y', labelcolor="C0")
    ax2 = ax1.twinx(); ax2.plot(df["step"], df["pref_rate"], color="C1", label="pref_rate"); ax2.set_ylabel("pref_rate", color="C1"); ax2.set_ylim(-0.05,1.05)
    ax3 = ax1.twinx(); ax3.spines["right"].set_position(("outward", 60)); ax3.plot(df["step"], df["avg_margin"], color="C2", label="avg_margin"); ax3.set_ylabel("avg_margin", color="C2")
    lines, labels = [], []
    for ax in (ax1, ax2, ax3):
        l, lab = ax.get_legend_handles_labels(); lines += l; labels += lab
    ax1.legend(lines, labels, loc="upper left")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "training_metrics.png")); plt.close(fig)

def plot_quick(df, out_dir):
    if df.empty:
        return
    fig, ax1 = plt.subplots(figsize=(14,6))
    ax1.plot(df["step"], df["acc"], color="C0", label="acc"); ax1.set_xlabel("global step"); ax1.set_ylabel("acc", color="C0"); ax1.set_ylim(0,1)
    ax2 = ax1.twinx(); ax2.plot(df["step"], df["mean_margin"], color="C1", label="mean_margin"); ax2.set_ylabel("mean_margin", color="C1")
    ax3 = ax1.twinx(); ax3.spines["right"].set_position(("outward", 60)); ax3.plot(df["step"], df["tie"], color="C2", label="tie"); ax3.set_ylabel("tie", color="C2")
    lines, labels = [], []
    for ax in (ax1, ax2, ax3):
        l, lab = ax.get_legend_handles_labels(); lines += l; labels += lab
    ax1.legend(lines, labels, loc="upper left")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "quick_val.png")); plt.close(fig)

def plot_full(df, out_dir):
    if df.empty:
        return
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(df["epoch"], df["acc"], marker="o", color="C0", label="acc"); ax1.set_xlabel("epoch"); ax1.set_ylabel("acc", color="C0"); ax1.set_ylim(0,1)
    ax2 = ax1.twinx(); ax2.plot(df["epoch"], df["mean_margin"], marker="s", color="C1", label="mean_margin"); ax2.set_ylabel("mean_margin", color="C1")
    ax3 = ax1.twinx(); ax3.spines["right"].set_position(("outward", 60)); ax3.plot(df["epoch"], df["tie"], marker="^", color="C2", label="tie"); ax3.set_ylabel("tie", color="C2")
    lines, labels = [], []
    for ax in (ax1, ax2, ax3):
        l, lab = ax.get_legend_handles_labels(); lines += l; labels += lab
    ax1.legend(lines, labels, loc="upper left")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "full_val.png")); plt.close(fig)

def save_summary(parsed, out_dir):
    summary = {
        "trainable_params": parsed.get("trainable_params"),
        "dataset_counts": parsed.get("dataset_counts"),
        "num_train_steps": int(parsed["train"].shape[0]) if parsed["train"] is not None else 0,
        "num_quick_val": int(parsed["quick"].shape[0]) if parsed["quick"] is not None else 0,
        "num_full_val": int(parsed["full"].shape[0]) if parsed["full"] is not None else 0,
        "saved_best": parsed.get("saved_best"),
        "saved_checkpoints_count": len(parsed.get("saved_checkpoints", [])),
    }
    out = os.path.join(out_dir, "summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return out

def main():
    p = argparse.ArgumentParser(description="Visualize training log without raw regex strings.")
    p.add_argument("--log-file", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="train_log_plots")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    parsed = parse_log_file(args.log_file)
    plot_training(parsed["train"], args.out_dir)
    plot_quick(parsed["quick"], args.out_dir)
    plot_full(parsed["full"], args.out_dir)
    summary_path = save_summary(parsed, args.out_dir)
    print("Saved summary to", summary_path)
    print("Saved plots in", args.out_dir)

if __name__ == "__main__":
    main()

