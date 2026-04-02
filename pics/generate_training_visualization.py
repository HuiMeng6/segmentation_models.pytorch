"""
Generate a dual-panel training process visualization chart for deep learning models.
Academic paper style with clean, professional colors.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Data ────────────────────────────────────────────────────────────────────
epochs = [10, 15, 20, 25]

train_loss = [0.485, 0.312, 0.198, 0.153]
val_loss   = [0.521, 0.347, 0.229, 0.181]

train_miou = [0.512, 0.641, 0.734, 0.768]
val_miou   = [0.487, 0.612, 0.703, 0.741]

# Step-wise LR: constant then drops at epoch 20 and 25
lr = [1e-3, 1e-3, 1e-4, 1e-5]

# ── Figure setup ─────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(7.5, 8.5),
    dpi=150,
    facecolor="white",
)
fig.subplots_adjust(hspace=0.42)

PANEL_BG  = "#F5F5F5"
GRID_KW   = dict(color="white", linewidth=1.0, linestyle="-")

# ── Helper: shared panel styling ─────────────────────────────────────────────
def style_panel(ax, title):
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, **GRID_KW)
    ax.set_axisbelow(True)
    ax.set_xticks(epochs)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#AAAAAA")

# ── Upper panel: Loss ─────────────────────────────────────────────────────────
l1, = ax1.plot(epochs, train_loss, color="#2CA02C", linestyle="-",
               marker="o", markersize=6, linewidth=1.8, label="Train Loss")
l2, = ax1.plot(epochs, val_loss,   color="#1F77B4", linestyle="--",
               marker="o", markersize=6, linewidth=1.8, label="Val Loss")

style_panel(ax1, "Training & Validation Loss")
ax1.set_ylabel("Loss", fontsize=11)
ax1.set_ylim(0.08, 0.62)
ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
ax1.legend(handles=[l1, l2], loc="upper right", fontsize=10,
           framealpha=0.9, edgecolor="#CCCCCC")

# ── Lower panel: mIoU + LR ───────────────────────────────────────────────────
ax3 = ax2  # left axis (mIoU)
ax4 = ax2.twinx()  # right axis (LR)

l3, = ax3.plot(epochs, train_miou, color="#D62728", linestyle="-",
               marker="^", markersize=7, linewidth=1.8, label="Train mIoU")
l4, = ax3.plot(epochs, val_miou,   color="#FF7F0E", linestyle="--",
               marker="^", markersize=7, linewidth=1.8, label="Val mIoU")

# Draw LR as a step-function: horizontal segments between epoch points
lr_steps_x = [epochs[0]] + [x for pair in zip(epochs[1:], epochs[1:]) for x in pair] + [epochs[-1]]
lr_steps_y = [y for pair in zip(lr[:-1], lr[1:]) for y in pair] + [lr[-1]]
# Simpler: use drawstyle='steps-post' on a duplicated x array
l5, = ax4.plot(epochs, lr, color="#9467BD", linestyle="-.",
               marker="D", markersize=6, linewidth=1.8,
               drawstyle="steps-post", label="Learning Rate")

style_panel(ax3, "mIoU & Learning Rate")
ax3.set_ylabel("mIoU", fontsize=11, color="#D62728")
ax3.tick_params(axis="y", labelcolor="#D62728")
ax3.set_ylim(0.40, 0.85)
ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

ax4.set_ylabel("Learning Rate", fontsize=11, color="#9467BD")
ax4.tick_params(axis="y", labelcolor="#9467BD")
ax4.set_yscale("log")
ax4.set_ylim(5e-6, 5e-3)
ax4.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
for spine in ax4.spines.values():
    spine.set_linewidth(0.8)
    spine.set_color("#AAAAAA")

# Combined legend — placed at lower right to avoid overlap with the LR line
# (LR steps down to its minimum at epoch 25, and mIoU lines rise above this region)
handles = [l3, l4, l5]
ax3.legend(handles=handles, loc="lower right", fontsize=10,
           framealpha=0.9, edgecolor="#CCCCCC", ncol=1)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = "/home/runner/work/segmentation_models.pytorch/segmentation_models.pytorch/pics/training_visualization.png"
fig.savefig(out_path, bbox_inches="tight", dpi=150)
print(f"Saved to {out_path}")
