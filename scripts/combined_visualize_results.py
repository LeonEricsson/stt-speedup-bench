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
}
MODEL_COLOURS = {
    "Whisper Large-v3 Turbo": "#66C2A5",
    "Whisper Medium": "#FC8D62",
    "Whisper Small": "#8DA0CB",
}
METRIC_STYLE = {"wer_macro": "solid", "cer_macro": "dashed"}

agg = {}
for model, path in DATA_FILES.items():
    df = pd.read_csv(path)
    df = df[df["speedup"] != 3.0]  # Exclude speedup factor of 3.0
    df_mean = (
        df.groupby("speedup")[["wer_macro", "cer_macro"]].mean().sort_index() * 100
    )
    agg[model] = df_mean

bottom_max = 60  # 0–60%
top_min, top_max = 80, 200
# Proportional height ratios for matched slopes:
height_ratios = [(top_max - top_min), bottom_max]

# Create figure
fig = plt.figure(figsize=(8, 6))
# add extra bottom margin to separate xlabel
fig.subplots_adjust(bottom=0.18, top=0.93, left=0.1, right=0.95)
gs = gridspec.GridSpec(2, 1, height_ratios=height_ratios, hspace=0.1)
ax_top = fig.add_subplot(gs[0])
ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

for model, df in agg.items():
    color = MODEL_COLOURS[model]
    ax_bot.plot(
        df.index, df["wer_macro"], color=color, linestyle=METRIC_STYLE["wer_macro"]
    )
    ax_bot.plot(
        df.index, df["cer_macro"], color=color, linestyle=METRIC_STYLE["cer_macro"]
    )
    ax_top.plot(
        df.index, df["wer_macro"], color=color, linestyle=METRIC_STYLE["wer_macro"]
    )
    ax_top.plot(
        df.index, df["cer_macro"], color=color, linestyle=METRIC_STYLE["cer_macro"]
    )

# Set axis limits
ax_bot.set_ylim(0, bottom_max)
ax_top.set_ylim(top_min, top_max)

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

# Centered overall x-axis and y-axis labels with spacing
fig.supxlabel("Playback Speed Factor", fontsize=12, y=0.05)
fig.supylabel("Error Rate (\%)", fontsize=12, x=0.03)

# X-ticks on bottom only
ax_bot.set_xticks(list(agg[next(iter(agg))].index))

# Legends in top region
model_handles = [
    Line2D([0], [0], color=MODEL_COLOURS[m], lw=1.8, label=m) for m in MODEL_COLOURS
]
metric_handles = [
    Line2D(
        [0],
        [0],
        color="black",
        lw=1.8,
        linestyle=METRIC_STYLE["wer_macro"],
        label="WER",
    ),
    Line2D(
        [0],
        [0],
        color="black",
        lw=1.8,
        linestyle=METRIC_STYLE["cer_macro"],
        label="CER",
    ),
]

leg_models = ax_top.legend(
    handles=model_handles,
    title="Model",
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.05, 0.9),
    borderpad=0.2,
)
ax_top.add_artist(leg_models)

ax_top.legend(
    handles=metric_handles,
    title="Metric",
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.3, 0.9),  # moved slightly right
    borderpad=0.2,
)

plt.show()
