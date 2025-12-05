"""
Confusion matrix image for Scenario 1 (proportional overlap), Policy A (drop ties).
This version ensures percentages are computed against the same denominator (non-tie total)
and that displayed percentages sum to exactly 100%.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Parameters (Scenario 1) ---
TOTAL = 14085            # full dataset size
MODEL_REJ = 222          # model predicted 'rejected' count
PREVALENCE = 0.05        # assumed fraction of examples whose human_label == 'rejected'
NON_TIE_TOTAL = 13136    # non-tie total (ties dropped) — use 13136 per your Policy A

# --- Compute counts under proportional-overlap assumption ---
# actual_rej computed from TOTAL (prevalence assumption)
actual_rej = int(round(PREVALENCE * TOTAL))

# proportional overlap alpha
alpha = actual_rej / TOTAL if TOTAL > 0 else 0.0

# TP: proportion of model_rej that fall on actual_rej (rounded)
tp = int(round(alpha * min(MODEL_REJ, actual_rej)))

# FP: remaining model_rej predictions that are not true rejected
fp = MODEL_REJ - tp

# FN: actual_rej that model did not predict as rejected
fn = actual_rej - tp

# Now compute TN so that TP+FP+FN+TN == NON_TIE_TOTAL (ties dropped)
# This is the key step: percentages will be computed relative to NON_TIE_TOTAL
tn = NON_TIE_TOTAL - (tp + fp + fn)

# Guard against negative TN due to inconsistent assumptions
if tn < 0:
    raise ValueError(
        "Inconsistent assumptions: NON_TIE_TOTAL is smaller than TP+FP+FN. "
        "Either reduce PREVALENCE or increase NON_TIE_TOTAL."
    )

# --- Percentages relative to NON_TIE_TOTAL ---
denom = float(NON_TIE_TOTAL)
pct_tp = tp / denom * 100.0
pct_fp = fp / denom * 100.0
pct_fn = fn / denom * 100.0
# compute TN pct as remainder to guarantee sum to 100%
pct_tn = 100.0 - (pct_tp + pct_fp + pct_fn)

# Small numerical safety: if pct_tn is tiny negative due to floating error, clamp to 0
if pct_tn < 0 and pct_tn > -1e-8:
    pct_tn = 0.0

# --- Prepare matrix for plotting ---
matrix = np.array([[tp, fp],
                   [fn, tn]], dtype=float)

annot = np.array([
    [f"{tp}\n({pct_tp:.2f}%)", f"{fp}\n({pct_fp:.2f}%)"],
    [f"{fn}\n({pct_fn:.2f}%)", f"{tn}\n({pct_tn:.2f}%)"]
])

# Plot
sns.set(style="whitegrid")
plt.figure(figsize=(10,6))
ax = plt.gca()

# Use a blue colormap; zero cells will appear light
cmap = sns.color_palette("Blues", as_cmap=True)
mask = np.zeros_like(matrix, dtype=bool)
# Plot heatmap with annotations
sns.heatmap(matrix, annot=annot, fmt="", cmap=cmap, cbar=True, linewidths=0.8,
            linecolor="black", annot_kws={"size":12, "weight":"bold"},
            xticklabels=["actual=rejected", "actual=chosen"],
            yticklabels=["pred=rejected", "pred=chosen"], ax=ax)

# Titles and caption
plt.title("Confusion matrix", fontsize=14, weight="bold")
subtitle = f" "
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
plt.suptitle(subtitle, fontsize=10, y=0.92)
caption = ("TP = model predicted rejected and human_label=rejected; "
           "FP = model predicted rejected but human_label=chosen; "
           "FN = model predicted chosen but human_label=rejected; "
           "TN = model predicted chosen and human_label=chosen. "
           "Percentages are relative to the non-tie total and sum to 100%.")
plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=9)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save or show
out_path = "/scratch/aowais2/llama32_dpo/best_adapter_epoch_2_step_15920/confusion_matrix_scenario1_policyA.png"
plt.savefig(out_path, dpi=200)
print(f"Saved confusion matrix image to: {out_path}")

# Print numeric summary
print("\nNumeric confusion matrix (positive = rejected_preferred):")
print(f"TP = {tp}, FP = {fp}")
print(f"FN = {fn}, TN = {tn}")
print(f"Denominator for percentages = {NON_TIE_TOTAL}")
print(f"Percentages: TP {pct_tp:.2f}%, FP {pct_fp:.2f}%, FN {pct_fn:.2f}%, TN {pct_tn:.2f}%")

