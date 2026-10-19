"""Grouped bar chart: per-region Pearson across 4 branch ablations (subj01)."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent.parent
CSV = REPO / "rebutal/exp3/results/per_region_summary.csv"
OUT_PNG = REPO / "rebutal/exp3/results/fig_per_region.png"
OUT_PDF = REPO / "rebutal/exp3/results/fig_per_region.pdf"

rows = list(csv.DictReader(open(CSV)))
variants_order = ["both", "where_only", "swap", "what_only"]
labels = {"both": "Both (full)", "where_only": "Where only\n(no CLS)",
          "swap": "Swap\n(CLS↔patches)", "what_only": "What only\n(no patches)"}
csv_keys = {"both": "both", "where_only": "whereonly", "swap": "swap", "what_only": "whatonly"}
groups = ["early", "mid", "higher", "overall"]
group_titles = ["Early\n(V1/V2/V3)", "Mid", "Higher", "Overall"]

data = {r["variant"]: r for r in rows}
matrix = np.zeros((len(groups), len(variants_order)))
for j, v in enumerate(variants_order):
    r = data[csv_keys[v]]
    matrix[0, j] = float(r["early_mean"])
    matrix[1, j] = float(r["mid_mean"])
    matrix[2, j] = float(r["higher_mean"])
    matrix[3, j] = float(r["overall"])

colors = ["#2c7bb6", "#abd9e9", "#fdae61", "#d7191c"]
x = np.arange(len(groups))
w = 0.2

fig, ax = plt.subplots(figsize=(7, 3.6))
for j, v in enumerate(variants_order):
    bars = ax.bar(x + (j - 1.5) * w, matrix[:, j], w,
                  label=labels[v], color=colors[j], edgecolor="black", linewidth=0.5)
    for b, val in zip(bars, matrix[:, j]):
        ax.text(b.get_x() + b.get_width() / 2, val + 0.005, f"{val:.2f}",
                ha="center", va="bottom", fontsize=7)

# Drop annotations vs baseline (both)
baseline = matrix[:, 0]
for j in range(1, len(variants_order)):
    for i in range(len(groups)):
        drop = (matrix[i, j] - baseline[i]) / baseline[i] * 100
        ax.text(x[i] + (j - 1.5) * w, -0.04, f"{drop:+.0f}%",
                ha="center", va="top", fontsize=6,
                color="darkred" if drop < -10 else ("dimgray" if drop < 0 else "darkgreen"))

ax.set_xticks(x)
ax.set_xticklabels(group_titles, fontsize=9)
ax.set_ylabel("Pearson r (test set, subj01)", fontsize=9)
ax.set_ylim(-0.08, 0.62)
ax.axhline(0, color="black", linewidth=0.4)
ax.legend(loc="upper left", fontsize=7, ncol=2, frameon=False)
ax.set_title("Branch ablations — region-specific impact (subj01, n=524 ROIs)",
             fontsize=10)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
plt.savefig(OUT_PDF, bbox_inches="tight")
print(f"✓ {OUT_PNG}\n✓ {OUT_PDF}")
