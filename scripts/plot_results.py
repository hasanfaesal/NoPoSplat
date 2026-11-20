#!/usr/bin/env python3
"""Generate publication-quality plots and Markdown tables from eval & training logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

STEP_LOSS_RE = re.compile(r"train step (\d+);.*loss = ([0-9.]+)")


def parse_train_log(path: Path) -> tuple[np.ndarray, np.ndarray]:
    steps, losses = [], []
    with path.open("r", errors="ignore") as f:
        for line in f:
            m = STEP_LOSS_RE.search(line)
            if m:
                steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
    return np.array(steps), np.array(losses)


def plot_training_curve(log1: Path, log2: Path, out_path: Path) -> None:
    s1, l1 = parse_train_log(log1)
    s2, l2 = parse_train_log(log2)
    # Keep stage 2 strictly after step 80,000 (ignores any interleaved pre-resume lines)
    mask2 = s2 >= 80000
    steps = np.concatenate([s1, s2[mask2]])
    losses = np.concatenate([l1, l2[mask2]])
    order = np.argsort(steps)
    steps, losses = steps[order], losses[order]

    win = max(1, min(100, len(losses) // 20))
    smooth = np.convolve(losses, np.ones(win) / win, mode="valid")
    smooth_steps = steps[win - 1 :]

    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=180)
    ax.plot(steps, losses, alpha=0.18, color="#4C78A8", linewidth=0.8, label="Raw step loss (MSE + LPIPS)")
    ax.plot(smooth_steps, smooth, color="#1F4E79", linewidth=2.0, label=f"Rolling mean ({win}-log window)")
    ax.axvline(80000, color="#E45756", linestyle="--", linewidth=1.3, label="Stage 2 resume (80k ckpt)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Total Loss (MSE + LPIPS)")
    ax.set_title("NoPoSplat TartanAir Fine-Tuning Convergence (1x NVIDIA H100 80GB)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_gpu_telemetry(csv_path: Path, out_path: Path) -> None:
    vram_gb, util, power = [], [], []
    with csv_path.open("r") as f:
        for row in csv.DictReader(f):
            try:
                vram_gb.append(float(row["vram_used_mib"]) / 1024.0)
                util.append(float(row["gpu_util_pct"]))
                power.append(float(row["power_w"]))
            except (KeyError, ValueError):
                continue
    hrs = np.arange(len(vram_gb)) / 60.0  # 1 sample/min

    fig, axes = plt.subplots(3, 1, figsize=(9, 6.2), sharex=True, dpi=180)
    axes[0].plot(hrs, vram_gb, color="#4C78A8", linewidth=1.3)
    axes[0].set_ylabel("VRAM (GB)")
    axes[0].set_title(f"H100 80GB Hardware Telemetry (Peak VRAM: {max(vram_gb):.1f} GB)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(hrs, util, color="#54A24B", linewidth=1.0, alpha=0.85)
    axes[1].set_ylabel("GPU Util (%)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(hrs, power, color="#F58518", linewidth=1.0, alpha=0.85)
    axes[2].set_ylabel("Power (W)")
    axes[2].set_xlabel("Elapsed Wall Time (hours)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_benchmark_bars(summary: dict, out_path: Path) -> None:
    models = list(summary.keys())
    x = np.arange(len(models))
    psnr_a = [summary[m]["overall"]["nvs_aligned"]["psnr"] for m in models]
    ssim_a = [summary[m]["overall"]["nvs_aligned"]["ssim"] for m in models]
    lpips_a = [summary[m]["overall"]["nvs_aligned"]["lpips"] for m in models]
    auc10 = [summary[m]["overall"]["pose"]["auc_10"] for m in models]

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), dpi=180)
    specs = [
        (psnr_a, "PSNR (dB) ↑", "#4C78A8"),
        (ssim_a, "SSIM ↑", "#54A24B"),
        (lpips_a, "LPIPS ↓", "#E45756"),
        (auc10, "Pose AUC@10° (%) ↑", "#B279A2"),
    ]
    for ax, (vals, title, color) in zip(axes, specs):
        bars = ax.bar(x, vals, color=color, width=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=8.5)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_qualitative_grid(renders_dir: Path, models: list[str], out_path: Path) -> None:
    pair_dirs = sorted([p for p in renders_dir.iterdir() if p.is_dir()])
    if not pair_dirs:
        return
    cols = [("ctx0.png", "Context 1"), ("ctx1.png", "Context 2"), ("gt.png", "GT Target")] + [
        (f"{m}_aligned.png", m) for m in models
    ]
    tile = 256
    header = 28
    canvas = Image.new("RGB", (tile * len(cols), header + tile * len(pair_dirs)), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for ci, (_, label) in enumerate(cols):
        draw.text((ci * tile + 8, 6), label, fill=(0, 0, 0))
    for ri, pdir in enumerate(pair_dirs):
        for ci, (fname, _) in enumerate(cols):
            fpath = pdir / fname
            if fpath.exists():
                im = Image.open(fpath).resize((tile, tile))
                canvas.paste(im, (ci * tile, header + ri * tile))
    canvas.save(out_path)


def print_markdown_table(summary: dict) -> str:
    lines = [
        "| Model | Aligned PSNR ↑ | Aligned SSIM ↑ | Aligned LPIPS ↓ | Unaligned PSNR ↑ | e_R (°) ↓ | e_t (°) ↓ | AUC@5° ↑ | AUC@10° ↑ | AUC@20° ↑ | Latency (ms) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for m, d in summary.items():
        ov = d["overall"]
        na, nu, po = ov["nvs_aligned"], ov["nvs_unaligned"], ov["pose"]
        lines.append(
            f"| `{m}` | {na['psnr']:.2f} | {na['ssim']:.3f} | {na['lpips']:.3f} | "
            f"{nu['psnr']:.2f} | {po['e_R_deg_median']:.2f} | {po['e_t_deg_median']:.2f} | "
            f"{po['auc_5']:.1f} | {po['auc_10']:.1f} | {po['auc_20']:.1f} | {ov['latency_ms_mean']:.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if (args.logs_dir / "train.log").exists() and (args.logs_dir / "train_extended.log").exists():
        plot_training_curve(
            args.logs_dir / "train.log",
            args.logs_dir / "train_extended.log",
            args.out_dir / "training_loss_curve.png",
        )
    if (args.logs_dir / "gpu_telemetry.csv").exists():
        plot_gpu_telemetry(args.logs_dir / "gpu_telemetry.csv", args.out_dir / "h100_telemetry.png")

    summary_path = args.eval_dir / "eval_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        plot_benchmark_bars(summary, args.out_dir / "benchmark_comparison.png")
        if (args.eval_dir / "renders").exists():
            build_qualitative_grid(args.eval_dir / "renders", list(summary.keys()), args.out_dir / "qualitative_grid.png")
        md = print_markdown_table(summary)
        (args.out_dir / "benchmark_table.md").write_text(md + "\n")
        print(md)


if __name__ == "__main__":
    main()
