# Copyright 2026 The verl contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Plot exported measurements without interpolating missing evaluations.

Requires matplotlib. Run report.py first to refresh the CSV snapshot.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator

COLORS = {"caesar_cipher": "#7c3aed", "simple_geometry": "#0f766e"}
TRAINING_COLOR = "#2563eb"
TRAINING_MFU_COLOR = "#1e3a8a"
ROLLOUT_COLOR = "#ef4444"
ROLLOUT_P90_COLOR = "#991b1b"
NAMES = {"caesar_cipher": "Caesar Cipher", "simple_geometry": "Geometry Simple"}


def read_rows(root, name):
    with (root / f"{name}.csv").open() as f:
        return list(csv.DictReader(f))


def series(ax, rows, key, label, scale=1, **kwargs):
    present = [r for r in rows if r.get(key, "") != ""]
    ax.plot([int(r["step"]) for r in present], [float(r[key]) * scale for r in present], label=label, **kwargs)


def teacher_sequence_lengths(root, batches):
    requests = {}
    with (root / "events.jsonl").open() as f:
        for line in f:
            event = json.loads(line)
            if "uid" not in event:
                continue
            row = requests.setdefault((event["step"], event["uid"]), {})
            row.update(event)
            if event["event"] == "teacher_end":
                row["scoring_completed"] = True
    groups = defaultdict(list)
    for row in requests.values():
        if row.get("scoring_completed"):
            groups[(int(row["step"]), row["teacher"])].append(row["input_tokens"] + row["output_tokens"])
    result = []
    for batch in batches:
        step, teacher = int(batch["step"]), batch["teacher"]
        lengths = groups[(step, teacher)]
        if len(lengths) != int(batch["requests"]) or sum(lengths) != int(batch["scored_tokens"]):
            raise ValueError(f"Teacher sequence lengths disagree with batch totals: {step}, {teacher}")
        result.append({"step": step, "teacher": teacher, "requests": len(lengths),
                       "sequence_tokens_p50": median(lengths)})
    return sorted(result, key=lambda row: (row["step"], row["teacher"]))


def save(fig, axes, root, name, title, pdf, note):
    for ax in axes.flat:
        ax.set_xlabel("Training step")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=14)
    fig.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=8)
    fig.tight_layout(rect=(0, 0.065, 1, 0.96))
    path = root / f"{name}.png"
    fig.savefig(path, dpi=160)
    fig.savefig(root / f"{name}.pdf")
    pdf.savefig(fig)
    plt.close(fig)
    print(path)


def plot_evaluation_snapshots(evaluation, output, pdf):
    """Show sparse evaluations as discrete observations on a shared percentage scale."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True)
    for ax, (domain, color) in zip(axes, COLORS.items(), strict=True):
        key = f"eval/student/{domain}/avg_at_4"
        rows = sorted((r for r in evaluation if r.get(key, "") != ""), key=lambda r: int(r["step"]))
        values = [100 * float(r[key]) for r in rows]
        reference = 100 * float(rows[0][f"eval/teacher_reported/{domain}/avg_at_4"])
        bars = ax.bar(range(len(rows)), values, width=0.52, color=color, zorder=3)
        for bar, row in zip(bars, rows, strict=True):
            if int(row["step"]) == 0:
                bar.set_alpha(0.45)
        ax.bar_label(bars, labels=[f"{v:.1f}%" for v in values], padding=6, fontsize=12)
        ax.axhline(reference, color=color, linestyle="--", linewidth=1.4, zorder=4,
                   label=f"Published teacher: {reference:.1f}%")
        ax.set_xticks(range(len(rows)), ["Step 0" if int(r["step"]) == 0
                                       else f"Step {r['step']}" for r in rows])
        ax.set_ylim(0, 112)
        ax.set_yticks(range(0, 101, 25))
        ax.set_xlim(-0.65, len(rows) - 0.35)
        ax.set_title(NAMES[domain], loc="left", pad=36, fontsize=13)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, borderaxespad=0, fontsize=10)
        ax.grid(axis="y", alpha=0.15, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.tick_params(axis="both", length=0, pad=8)
    axes[0].set_ylabel("Student exact-match score · avg@4 (%)", labelpad=12)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.10, top=0.80, wspace=0.17)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"evaluation-snapshots.{extension}", dpi=180)
    pdf.savefig(fig)
    plt.close(fig)
    print(output / "evaluation-snapshots.png")


def plot_trainer_step_breakdown(train, output, pdf):
    rows = sorted(train, key=lambda r: int(r["step"]))
    steps = [int(r["step"]) for r in rows]
    totals = np.array([float(r["trainer/step_s"]) for r in rows])
    phases = (
        ("wait_s", "Wait", "#dbeafe", "#172033"),
        ("train_s", "Training", TRAINING_COLOR, "white"),
        ("log_prob_adv_s", "Log Prob + Advantage", "#93c5fd", "#172033"),
        ("weight_bcast_s", "Weight Bcast", TRAINING_MFU_COLOR, "white"),
    )
    wait = np.array([float(r["trainer/wait_s"]) for r in rows])
    training = np.array([float(r["trainer/train_s"]) for r in rows])
    log_prob_adv = np.array([float(r["timing_s/old_log_prob"]) + float(r["timing_s/adv"]) for r in rows])
    weight_bcast = np.array([float(r["timing_s/update_weights"]) for r in rows])
    remaining = np.array([float(r["trainer/other_s"]) for r in rows]) - log_prob_adv - weight_bcast
    values = [wait, training, log_prob_adv, weight_bcast]
    if not np.allclose(sum(values) + remaining, totals) or np.any(totals <= 0) or np.any(np.array(values) < 0):
        raise ValueError("Step timing components must sum to step time excluding evaluation")
    totals = sum(values)
    fig, ax = plt.subplots(figsize=(14, 9))
    bottom = np.zeros(len(rows))
    for phase, durations in zip(phases, values, strict=True):
        _, label, color, text_color = phase
        ax.bar(steps, durations, bottom=bottom, width=0.8, color=color, label=label)
        for step, duration, base, total in zip(steps, durations, bottom, totals, strict=True):
            if duration > 0:
                if duration < 3:
                    ax.annotate(f"{100 * duration / total:.1f}%", (step, base + duration),
                                xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                                color="#172033", fontsize=8)
                else:
                    ax.text(step, base + duration / 2, f"{100 * duration / total:.1f}%",
                            ha="center", va="center", color=text_color, fontsize=8)
        bottom += durations
    ax.set(xlabel="Training Step", ylabel="Time (Seconds)",
           xlim=(min(steps) - 0.7, max(steps) + 0.7), ylim=(0, max(totals) * 1.04))
    ax.set_xticks(steps)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(axis="both", length=0, pad=8)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, frameon=False)
    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.12, top=0.91)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"trainer-step-breakdown.{extension}", dpi=180)
    pdf.savefig(fig)
    plt.close(fig)
    print(output / "trainer-step-breakdown.png")


def plot_trainer_throughput_mfu(train, output, pdf):
    rows = sorted(train, key=lambda r: int(r["step"]))
    steps = [int(r["step"]) for r in rows]
    throughput = [float(r["trainer/train_tok_s_gpu"]) for r in rows]
    # Use the RPC-interval estimate so both metrics cover the same actor-update time.
    mfu = [100 * float(r["trainer/train_mfu_estimate"]) for r in rows]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    right = ax.twinx()
    bars = ax.bar(steps, throughput, width=0.65, color=TRAINING_COLOR, alpha=0.8,
                  label="Trainer Throughput")
    line, = right.plot(steps, mfu, color=TRAINING_MFU_COLOR, marker="o", markersize=5,
                       linewidth=2, label="Estimated Trainer MFU")
    ax.set(xlabel="Training Step", ylabel="Throughput (Tokens/s/GPU)",
           xlim=(min(steps) - 0.7, max(steps) + 0.7), ylim=(0, max(throughput) * 1.12))
    right.set(ylabel="Estimated MFU (%)", ylim=(0, max(mfu) * 1.12))
    ax.set_xticks(steps)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelcolor=TRAINING_COLOR, length=0, pad=8)
    right.tick_params(axis="y", labelcolor=TRAINING_MFU_COLOR, length=0, pad=8)
    ax.tick_params(axis="x", length=0, pad=8)
    ax.yaxis.label.set_color(TRAINING_COLOR)
    right.yaxis.label.set_color(TRAINING_MFU_COLOR)
    for axis in (ax, right):
        axis.spines["left"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["bottom"].set_color("#cbd5e1")
    ax.legend([bars, line], ["Trainer Throughput", "Estimated Trainer MFU"],
              loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.085, right=0.925, bottom=0.13, top=0.90)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"trainer-throughput-mfu.{extension}", dpi=180)
    pdf.savefig(fig)
    plt.close(fig)
    print(output / "trainer-throughput-mfu.png")


def plot_rollout_lengths_latency(root, rollout, output, pdf):
    # Pool requests across both tasks; compute percentiles from individual requests.
    rows = sorted(rollout, key=lambda r: int(r["step"]))
    steps = [int(r["step"]) for r in rows]
    request_lengths = defaultdict(dict)
    request_latency = defaultdict(dict)
    with (root / "events.jsonl").open() as f:
        for line in f:
            event = json.loads(line)
            if event["event"] == "generated":
                request_lengths[int(event["step"])][event["uid"]] = event["input_tokens"] + event["output_tokens"]
            elif event["event"] == "ready":
                request_latency[int(event["step"])][event["uid"]] = event["e2e_latency_s"]
    records = []
    for row in rows:
        step = int(row["step"])
        samples = list(request_lengths[step].values())
        if len(samples) != int(row["generated"]) or sum(samples) != int(row["input_tokens"]) + int(row["output_tokens"]):
            raise ValueError(f"Rollout sequence lengths disagree with batch totals at step {step}")
        if request_lengths[step].keys() != request_latency[step].keys():
            raise ValueError(f"Rollout latency and length request sets disagree at step {step}")
        record = {"step": step, "requests": len(samples)}
        for percentile in (50, 90):
            record[f"total_sequence_tokens_p{percentile}"] = float(np.percentile(samples, percentile, method="linear"))
            record[f"e2e_latency_s_p{percentile}"] = float(np.percentile(
                list(request_latency[step].values()), percentile, method="linear"))
        if not np.isclose(record["e2e_latency_s_p50"], float(row["request_e2e_latency_s_p50"])):
            raise ValueError(f"Recomputed median latency disagrees with batch export at step {step}")
        records.append(record)
    with (output / "rollout-lengths-latency.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "requests", "total_sequence_tokens_p50",
                                              "e2e_latency_s_p50", "total_sequence_tokens_p90", "e2e_latency_s_p90"])
        writer.writeheader()
        writer.writerows(records)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, ylabel in zip(axes, ("total_sequence_tokens", "e2e_latency_s"),
                               ("Total Sequence Length (Tokens)", "End-to-End Latency (Seconds)"), strict=True):
        for percentile, color, marker, style in ((50, ROLLOUT_COLOR, "o", "-"), (90, ROLLOUT_P90_COLOR, "s", "--")):
            ax.plot(steps, [r[f"{key}_p{percentile}"] for r in records], color=color,
                    marker=marker, markersize=4, linestyle=style, linewidth=2, label=f"P{percentile}")
        ax.set(xlabel="Training Step", ylabel=ylabel, ylim=(0, None),
               xlim=(min(steps) - 0.5, max(steps) + 0.5))
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.tick_params(axis="both", length=0, pad=8)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].set_ylabel("Total Sequence Length (K Tokens)")
    axes[0].set_ylim(0, 25000)
    axes[0].yaxis.set_major_locator(MultipleLocator(2500))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:g}"))
    axes[1].yaxis.set_major_locator(MultipleLocator(10))
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=False)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.87, wspace=0.30)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"rollout-lengths-latency.{extension}", dpi=180)
    pdf.savefig(fig)
    plt.close(fig)
    print(output / "rollout-lengths-latency.png")


def plot_generation_effective_throughput(rollout, output, pdf):
    rows = sorted(rollout, key=lambda r: int(r["step"]))
    steps = [int(r["step"]) for r in rows]
    rates = [float(r["effective_output_tokens"]) / float(r["generation_s"]) for r in rows]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(steps, rates, width=0.65, color=ROLLOUT_COLOR, alpha=0.8)
    ax.set(xlabel="Training Step", ylabel="Effective Generation Throughput (tok/s)",
           xlim=(min(steps) - 0.7, max(steps) + 0.7), ylim=(0, max(rates) * 1.10))
    ax.set_xticks(steps)
    ax.yaxis.set_major_locator(MultipleLocator(1000))
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(axis="both", length=0, pad=8)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.14, top=0.97)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"generation-effective-throughput.{extension}", dpi=180)
    pdf.savefig(fig)
    plt.close(fig)
    print(output / "generation-effective-throughput.png")


def plot_producer_consumer_throughput(train, rollout, output, pdf):
    training = {int(r["step"]): r for r in train}
    records = []
    for row in sorted(rollout, key=lambda r: int(r["step"])):
        step = int(row["step"])
        # This run accepts every generated sequence; both phases count input + output.
        if int(row["accepted"]) != int(row["generated"]):
            raise ValueError("Input + output comparison requires accepted-request token accounting when filtering occurs")
        tokens = int(row["input_tokens"]) + int(row["output_tokens"])
        generation_s = float(row["generation_s"])
        training_s = float(training[step]["trainer/train_s"])
        if min(generation_s, training_s) <= 0:
            raise ValueError(f"Nonpositive phase duration at step {step}")
        records.append({"step": step, "input_plus_output_tokens": tokens,
                        "generation_s": generation_s, "training_s": training_s,
                        "generation_tok_s": tokens / generation_s,
                        "training_tok_s": tokens / training_s})
    with (output / "producer-consumer-throughput.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    steps = [r["step"] for r in records]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(steps, [r["generation_tok_s"] for r in records], color=ROLLOUT_COLOR,
            marker="o", markersize=5, linewidth=2)
    ax.plot(steps, [r["training_tok_s"] for r in records], color=TRAINING_COLOR,
            marker="s", markersize=5, linewidth=2, linestyle="--")
    ax.legend(["Generation", "Training"], loc="lower center",
              bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    ax.set(xlabel="Training Step", ylabel="Throughput (tok/s)", ylim=(0, None),
           xlim=(min(steps) - 0.5, max(steps) + 0.5))
    ax.set_xticks(steps)
    ax.yaxis.set_major_locator(MultipleLocator(5000))
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(axis="both", length=0, pad=8)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.13, top=0.87)
    for extension in ("png", "pdf"):
        fig.savefig(output / f"producer-consumer-throughput.{extension}", dpi=180)
    pdf.savefig(fig)
    plt.close(fig)
    print(output / "producer-consumer-throughput.png")


def plot_teacher_system_performance(teacher, lengths, output, pdf):
    """Per-teacher scoring-window rates and request median input + output length."""
    length_by_key = {(r["step"], r["teacher"]): r["sequence_tokens_p50"] for r in lengths}
    dark_colors = {"caesar_cipher": "#3b0764", "simple_geometry": "#042f2e"}
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    exported = []
    for ax, (domain, color) in zip(axes, COLORS.items(), strict=True):
        rows = sorted((r for r in teacher if r["teacher"] == domain), key=lambda r: int(r["step"]))
        steps = [int(r["step"]) for r in rows]
        throughput = [float(r["scored_tok_s_gpu"]) for r in rows]
        mfu = [100 * float(r["mfu_estimate"]) for r in rows]
        sequence = [length_by_key[(step, domain)] / 1000 for step in steps]
        bars = ax.bar(steps, throughput, width=0.72, color=color,
                      label="Throughput (Bar Labels: Step Time MFU)", zorder=2)
        for bar, percent in zip(bars, mfu, strict=True):
            ax.annotate(f"{percent:.2f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                        color=dark_colors[domain], fontsize=8, zorder=5)
        right = ax.twinx()
        line, = right.plot(steps, sequence, color=dark_colors[domain], marker="o",
                           markersize=5, markeredgecolor="white", linewidth=2,
                           label="P50 Sequence Length", zorder=3)
        ax.set(xlabel="Training Step", ylabel="Throughput (tok/s/GPU)",
               xlim=(0.3, 20.7), ylim=(0, 2500))
        ax.set_xticks(steps)
        ax.yaxis.set_major_locator(MultipleLocator(500))
        ax.tick_params(axis="y", colors=color)
        ax.yaxis.label.set_color(color)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.18)
        right.set(ylabel="P50 Sequence Length (K Tokens)", ylim=(0, 20))
        right.yaxis.set_major_locator(MultipleLocator(2.5))
        right.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        right.spines["right"].set_visible(True)
        right.spines["right"].set_color(dark_colors[domain])
        right.spines["right"].set_linewidth(1)
        right.tick_params(axis="y", colors=dark_colors[domain], direction="in", pad=8)
        right.yaxis.label.set_color(dark_colors[domain])
        ax.legend(handles=[bars, line], loc="upper left", frameon=False, ncol=2,
                  title=NAMES[domain], alignment="left")
        for step, rate, percent, seq in zip(steps, throughput, mfu, sequence, strict=True):
            exported.append(dict(step=step, teacher=domain, throughput_tok_s_gpu=rate,
                                 mfu_percent=percent, sequence_tokens_p50=seq * 1000))
    fig.tight_layout(h_pad=3)
    fig.savefig(output / "teacher-system-performance.png", dpi=180)
    fig.savefig(output / "teacher-system-performance.pdf")
    pdf.savefig(fig)
    plt.close(fig)
    with (output / "teacher-system-performance.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Defaults to the run directory")
    args = parser.parse_args()
    root = args.run
    output = args.output_dir or root
    output.mkdir(parents=True, exist_ok=True)
    train = read_rows(root, "trainer")
    rollout = read_rows(root, "rollout_batches")
    teacher = read_rows(root, "teacher_batches")
    teacher_lengths = teacher_sequence_lengths(root, teacher)
    evaluation = read_rows(root, "evaluation")
    if not train:
        raise ValueError("No completed training steps to plot")
    last_step = max(int(r["step"]) for r in train)
    suffix = f" — through completed step {last_step}"
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    pdf = PdfPages(output / "mopd-metrics-report.pdf")
    plot_evaluation_snapshots(evaluation, output, pdf)
    plot_trainer_step_breakdown(train, output, pdf)
    plot_trainer_throughput_mfu(train, output, pdf)
    plot_rollout_lengths_latency(root, rollout, output, pdf)
    plot_generation_effective_throughput(rollout, output, pdf)
    plot_producer_consumer_throughput(train, rollout, output, pdf)
    plot_teacher_system_performance(teacher, teacher_lengths, output, pdf)
    fig = plt.figure(figsize=(12, 12))
    grid = fig.add_gridspec(3, 2)
    axes = np.array([
        fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[2, 0]), fig.add_subplot(grid[2, 1]),
    ])
    ax = axes[0]
    for domain, color in COLORS.items():
        series(
            ax, evaluation, f"eval/student/{domain}/avg_at_4", f"{NAMES[domain]} student", 100, color=color, marker="o"
        )
        ref = next(float(r[f"eval/teacher_reported/{domain}/avg_at_4"]) for r in evaluation)
        ax.plot([0, last_step], [100 * ref] * 2, "--", color=color, label=f"{NAMES[domain]} teacher (published)")
    ax.set(title="Student evaluation and teacher references", ylabel="Exact-match avg@4 (%)", ylim=(0, 105))
    ax = axes[1]
    bottoms = [0.0] * len(train)
    for key, label, color in (
        ("wait_s", "Wait", "#94a3b8"),
        ("train_s", "Actor update", "#2563eb"),
        ("other_s", "Other step work", "#f59e0b"),
        ("eval_s", "Evaluation", "#16a34a"),
    ):
        values = [float(r[f"trainer/{key}"]) for r in train]
        ax.bar([int(r["step"]) for r in train], values, bottom=bottoms, label=label, color=color)
        bottoms = [a + b for a, b in zip(bottoms, values, strict=True)]
    ax.set(title="Step time, with evaluation shown separately", ylabel="Seconds")
    ax = axes[2]
    series(ax, train, "trainer/train_s", "Actor update per step", marker=".")
    series(ax, train, "trainer/train_s_mean", "Cumulative mean", linestyle="--")
    mean_train = sum(float(r["trainer/train_s"]) for r in train) / len(train)
    ax.axhline(mean_train, color="#64748b", linestyle=":", label=f"Run mean: {mean_train:.2f} s")
    ax.set(title="Actor training time, including first-update startup", ylabel="Seconds")
    ax = axes[3]
    series(ax, train, "trainer/train_tok_s_gpu", "Per step", marker=".")
    series(ax, train, "trainer/train_tok_s_gpu_run", "Cumulative", linestyle="--")
    ax.set(title="Actor-update throughput", ylabel="Input + output tokens/s/GPU")
    ax = axes[4]
    series(ax, train, "trainer/train_mfu_estimate", "Actor RPC interval estimate", 100, marker=".")
    series(ax, train, "perf/mfu/actor", "Native engine MFU", 100, linestyle=":")
    series(ax, train, "trainer/train_mfu_estimate_run", "Cumulative RPC estimate", 100, linestyle="--")
    ax.set(title="Actor MFU estimates", ylabel="MFU (%)")
    save(fig, axes, output, "training-summary", "Training and evaluation" + suffix, pdf,
         "Evaluations measured only at steps 0, 10 and 20; dashed teacher scores are published references, not a controlled rerun.\n"
         "Actor rates: 8 GPUs, input + output tokens; cumulative rates are time-weighted. MFU is estimated.\n"
         "Step bars exclude baseline evaluation at step 0; other work includes checkpointing and weight synchronization.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    series(ax, rollout, "request_e2e_latency_s_p50", "Request end-to-end P50", marker=".")
    series(ax, rollout, "ready_s", "Logical batch end-to-end", linestyle="--")
    ax.set(title="Rollout latency", ylabel="Seconds")
    ax = axes[0, 1]
    series(ax, rollout, "request_input_tokens_p50", "Input P50", marker=".")
    series(ax, rollout, "request_output_tokens_p50", "Output P50", marker=".")
    ax.set(title="Sequence lengths within each batch", ylabel="Tokens (log scale)", yscale="log")
    ax = axes[1, 0]
    series(ax, rollout, "output_tok_s_gpu", "Generated output", linewidth=2)
    series(ax, rollout, "effective_output_tok_s_gpu", "Accepted loss-mask output", linestyle="--", marker=".")
    ax.set(title="Rollout throughput and effective throughput", ylabel="Output tokens/s/GPU")
    ax = axes[1, 1]
    series(ax, rollout, "filtered_ratio", "Filtered generated responses", 100, linewidth=2)
    series(ax, rollout, "failure_ratio", "Failed requests", 100, linestyle="--", marker=".")
    ax.set(title="Filtering and failures", ylabel="Percent", ylim=(-1, 101))
    save(fig, axes, output, "rollout-summary", "Logical rollout batches (one per step)" + suffix, pdf,
         "One logical batch per step: batch P50 equals that batch's value; request P50 is within the batch. Rates use 8 GPUs.\n"
         "End-to-end includes teacher scoring and reward. Throughput uses generation time; effective throughput uses accepted loss-mask tokens.\n"
         "All responses accepted: generated and effective throughput overlap; filtering and failure rates are both zero.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = (
        ("window_s", "Logical scoring batch window", "Seconds", 1),
        ("request_latency_s_p50", "Teacher request latency P50", "Seconds", 1),
        ("mfu_estimate", "Teacher MFU estimate over scoring window", "MFU (%)", 100),
        ("scored_tok_s_gpu", "Teacher scoring throughput", "Input + output tokens/s/GPU", 1),
    )
    for ax, (key, title, ylabel, scale) in zip(axes.flat, panels, strict=True):
        for domain, color in COLORS.items():
            series(
                ax, [r for r in teacher if r["teacher"] == domain], key, NAMES[domain], scale, color=color, marker="."
            )
        ax.set(title=title, ylabel=ylabel)
    save(fig, axes, output, "teacher-summary", "Teacher scoring (critic)" + suffix, pdf,
         "One scoring batch per teacher per step: batch P50 equals the batch value; request latency P50 is within the batch.\n"
         "Batch windows include waiting for rollout supply. Each teacher uses 4 GPUs; throughput counts input + output tokens.\n"
         "MFU is a model-FLOP estimate using 2,250 TFLOP/s/GPU, not a hardware-counter measurement.")
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for domain, color in COLORS.items():
        series(ax, [r for r in teacher_lengths if r["teacher"] == domain],
               "sequence_tokens_p50", NAMES[domain], color=color, marker="o")
    ax.set(ylabel="Median scored sequence length (tokens)", ylim=(0, None))
    save(fig, np.array([ax]), output, "teacher-sequence-length",
         "Teacher median scored sequence length" + suffix, pdf,
         "Per-step median of prompt + student-response tokens scored by each teacher.\n"
         "Derived from completed scoring requests; excludes the teacher API's auxiliary generated token.")
    with (output / "teacher-sequence-length.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "teacher", "requests", "sequence_tokens_p50"])
        writer.writeheader()
        writer.writerows(teacher_lengths)
    pdf.close()
    print(output / "mopd-metrics-report.pdf")


if __name__ == "__main__":
    main()
