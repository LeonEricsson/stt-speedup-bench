import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl

mpl.rcParams.update(
    {
        "figure.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "text.usetex": True,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 8,
        "lines.linewidth": 1.8,
    }
)

DATA_FILES = {
    "Whisper Large-v3 Turbo": "results/whisper-large-v3-turbo-fine.csv",
    "Whisper Medium": "results/whisper-medium-fine.csv",
    "Whisper Small": "results/whisper-small-fine.csv",
}

# colour-blind-friendly palette → one colour per model
MODEL_COLOURS = {
    "Whisper Large-v3 Turbo": "#66C2A5",  # teal
    "Whisper Medium": "#FC8D62",  # salmon
    "Whisper Small": "#8DA0CB",  # lavender
}
METRIC_STYLE = {"wer_macro": "solid", "cer_macro": "dashed"}

# --------------------------------------------------------------------------
# load → average across languages → convert to %
agg = {}
for model, path in DATA_FILES.items():
    df = pd.read_csv(path)
    df = df[df["speedup"] != 3.0]
    df_mean = df.groupby("speedup")[["wer_macro", "cer_macro"]].mean().sort_index()
    df_mean *= 100
    agg[model] = df_mean

# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

for model, df in agg.items():
    colour = MODEL_COLOURS[model]
    for metric in ("wer_macro", "cer_macro"):
        ax.plot(df.index, df[metric], color=colour, linestyle=METRIC_STYLE[metric])

# axes & grid
ax.set_xlabel("Playback Speed Factor")
ax.set_ylabel("Error Rate (\%)")
ax.set_xticks(df.index)
ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

# --- dual, ultra-compact legends ------------------------------------------
model_handles = [
    Line2D([0], [0], color=MODEL_COLOURS[m], lw=1.8, label=m) for m in MODEL_COLOURS
]
metric_handles = [
    Line2D([0], [0], color="black", lw=1.8, linestyle="solid", label="WER"),
    Line2D([0], [0], color="black", lw=1.8, linestyle="dashed", label="CER"),
]
leg_models = ax.legend(
    handles=model_handles,
    title="Model",
    frameon=False,
    loc="upper left",  # anchor point is the *corner* chosen by 'loc'
    bbox_to_anchor=(0.05, 0.94),  # (x, y) in axes fraction units
    borderpad=0.2,
)

# --- metric legend: same anchor corner, but move it downward (e.g. y=0.78)
leg_metrics = ax.legend(
    handles=metric_handles,
    title="Metric",
    frameon=False,
    loc="upper left",
    bbox_to_anchor=(0.11, 0.76),  # adjust y until it looks right
    borderpad=0.2,
)

ax.add_artist(leg_models)  # keep both legends

fig.tight_layout()
# plt.show()
plt.savefig("error_rate_speedup.png", format="png", dpi=600, bbox_inches="tight")
