"""Render Inkling-Small training curves from the recorded metrics CSV."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).parent
with (ROOT / "metrics.csv").open(newline="") as source:
    rows = list(csv.DictReader(source))


def points(key):
    return [(int(row["step"]), float(row[key])) for row in rows if row[key]]


def trailing_mean(values, width=7):
    return [sum(values[max(0, i - width + 1) : i + 1]) / min(i + 1, width) for i in range(len(values))]


def make_line_chart(filename, key, title, ylabel, *, color, percent=False):
    samples = points(key)
    x = [step for step, _ in samples]
    y = [value * 100 if percent else value for _, value in samples]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    if percent:
        ax.plot(x, y, "o-", color=color, linewidth=2.3, markersize=5)
        for i in {0, max(range(len(y)), key=y.__getitem__), len(y) - 1}:
            ax.annotate(
                f"{round(y[i] * 0.3)}/30",
                (x[i], y[i]),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )
        ax.yaxis.set_major_formatter(PercentFormatter(100))
        ax.set_ylim(0, max(40, max(y) + 5))
    else:
        ax.plot(x, y, color=color, linewidth=1, alpha=0.28, label="Per step")
        ax.plot(x, trailing_mean(y), color=color, linewidth=2.5, label="7-step trailing mean")
        ax.legend(frameon=False)
    ax.set(title=title, xlabel="Global training step", ylabel=ylabel, xlim=(0, 100))
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ROOT / filename, dpi=180)
    plt.close(fig)


make_line_chart("training-reward.png", "critic/rewards/mean", "Training reward", "Mean 0/1 score", color="#155D8B")
make_line_chart(
    "heldout-aime2025.png",
    "val-core/miles_math/acc/mean@1",
    "AIME2025 held-out accuracy",
    "30 questions, temperature 1, n=1",
    color="#2A9D8F",
    percent=True,
)
make_line_chart(
    "train-rollout-probability-mae.png",
    "training/rollout_probs_diff_mean",
    "Train–rollout probability mismatch",
    "Mean absolute probability difference",
    color="#B33635",
)
make_line_chart(
    "train-rollout-correction-kl.png", "rollout_corr/kl", "Train–rollout correction KL", "KL", color="#7A5195"
)
