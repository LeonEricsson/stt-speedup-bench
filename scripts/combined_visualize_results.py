import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import gridspec
import matplotlib as mpl

mpl.rcParams.update(
    {
        "figure.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "text.usetex": True,  # change to True if LaTeX is installed on your system
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8,
        "lines.linewidth": 1.8,
    }
)

# Load and aggregate data
DATA_FILES = {
    "Whisper Large-v3 Turbo": "results/whisper-large-v3-turbo-full.csv",
    "Whisper Medium": "results/whisper-medium-full.csv",
    "Whisper Small": "results/whisper-small-full.csv",
    "GPT-4o Transcribe": "results/gpt-4o-transcribe-full.csv",
}
MODEL_COLOURS = {
    "Whisper Large-v3 Turbo": "#66C2A5",
    "Whisper Medium": "#FC8D62",
    "Whisper Small": "#8DA0CB",
    "GPT-4o Transcribe": "#CC79A7",
}
METRIC_STYLE = {"wer_macro": "solid"}

agg = {}
for model, path in DATA_FILES.items():
    df = pd.read_csv(path)
    df = df[df["speedup"] <= 2.5]
    df_mean = df.groupby("speedup")[["wer_macro"]].mean().sort_index() * 100
    agg[model] = df_mean

bottom_max = 25  # show 0–20% in bottom
top_min, top_max = 60, 150  # top panel shows 80–200%
height_ratios = [1, 0.8]  # top region half the height of bottom

# Create figure with adjusted margins
fig = plt.figure(figsize=(8, 6))
fig.subplots_adjust(left=0.1, right=0.95, bottom=0.12, top=0.93)

gs = gridspec.GridSpec(2, 1, height_ratios=height_ratios, hspace=0.1)
ax_top = fig.add_subplot(gs[0])
ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

# Plot WER only
for model, df in agg.items():
    color = MODEL_COLOURS[model]
    ax_bot.plot(
        df.index, df["wer_macro"], color=color, linestyle=METRIC_STYLE["wer_macro"]
    )
    ax_top.plot(
        df.index, df["wer_macro"], color=color, linestyle=METRIC_STYLE["wer_macro"]
    )

# Set axis limits
ax_bot.set_ylim(0, bottom_max)
ax_top.set_ylim(top_min, top_max)

ax_top.set_yticks([60, 80, 100, 120, 150])

# Hide x-axis labels & ticks on top subplot
ax_top.tick_params(labelbottom=False, bottom=False)

# Diagonal break markers
d = 0.015
kwargs_top = dict(transform=ax_top.transAxes, color="k", clip_on=False)
kwargs_bot = dict(transform=ax_bot.transAxes, color="k", clip_on=False)
ax_top.plot((-d, +d), (0, 0), **kwargs_top)
ax_top.plot((1 - d, 1 + d), (0, 0), **kwargs_top)
ax_bot.plot((-d, +d), (1, 1), **kwargs_bot)
ax_bot.plot((1 - d, 1 + d), (1, 1), **kwargs_bot)

# Grids
for ax in (ax_top, ax_bot):
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

# Centered overall labels
fig.supxlabel("Playback Speed Factor", fontsize=12, y=0.02)
fig.supylabel("Word Error Rate (\%)", fontsize=12, x=0.03)

# X-ticks on bottom only
ax_bot.set_xticks(list(agg[next(iter(agg))].index))

# Single legend of models in top region
model_handles = [
    Line2D([0], [0], color=MODEL_COLOURS[m], lw=1.8, label=m) for m in MODEL_COLOURS
]
ax_top.legend(
    handles=model_handles,
    title="Model",
    frameon=True,
    loc="upper left",
    bbox_to_anchor=(0.06, 0.9),
    borderpad=0.2,
)

# plt.show()
# fig.tight_layout()
plt.savefig("tldr.png", format="png", dpi=600, bbox_inches="tight")
