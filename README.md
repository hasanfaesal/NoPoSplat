<h1 align="center">NoPoSplat-TartanAir: Pose-Free Feed-Forward 3D Gaussian Splatting for 6-DoF Robotics Trajectories</h1>

<p align="center">
  <a href="https://huggingface.co/hasanfaesal/NoPoSplat-TartanAir"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights%20%26%20Splats-yellow.svg" alt="Hugging Face"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="https://arxiv.org/abs/2410.24207"><img src="https://img.shields.io/badge/Base%20Paper-NoPoSplat%20(ICLR%20'25%20Oral)-red.svg" alt="NoPoSplat Paper"></a>
  <a href="https://theairlab.org/tartanair-dataset/"><img src="https://img.shields.io/badge/Dataset-TartanAir-purple.svg" alt="TartanAir"></a>
</p>

<p align="center">
  Fine-tuning and benchmarking <a href="https://github.com/cvg/NoPoSplat"><strong>NoPoSplat</strong></a> (Ye et al., ICLR 2025 Oral) on 6-DoF aerial &amp; industrial robotics trajectories from <strong>TartanAir</strong> — achieving <strong>+2.02 dB PSNR</strong>, <strong>+34.1 pts Pose AUC@5°</strong>, and <strong>+30.9 pts Pose AUC@20°</strong> over zero-shot RealEstate10K weights, including strong zero-shot transfer to completely unseen indoor environments.
</p>

---

## Qualitative 3DGS Orbit Previews (Zero-Shot RE10K vs. Fine-Tuned TartanAir)

Each preview renders a smooth circular camera orbit directly from the predicted 3D Gaussians given **two unposed 256×256 input frames** (left: zero-shot `re10k.ckpt`, right: fine-tuned `tartanair_step5000`):

| In-Domain Held-Out (`office/Easy/P001`) | Aggressive 6-DoF Motion (`office/Hard/P000`) |
| :---: | :---: |
| ![office Easy orbit](https://huggingface.co/hasanfaesal/NoPoSplat-TartanAir/resolve/main/v2/orbits/orbit_office_Easy_P001.gif) | ![office Hard orbit](https://huggingface.co/hasanfaesal/NoPoSplat-TartanAir/resolve/main/v2/orbits/orbit_office_Hard_P000.gif) |

| Industrial Unseen Trajectory (`carwelding/Easy/P001`) | **100% Unseen Environment** (`office2/Easy/P000`) |
| :---: | :---: |
| ![carwelding Easy orbit](https://huggingface.co/hasanfaesal/NoPoSplat-TartanAir/resolve/main/v2/orbits/orbit_carwelding_Easy_P001.gif) | ![office2 Easy orbit](https://huggingface.co/hasanfaesal/NoPoSplat-TartanAir/resolve/main/v2/orbits/orbit_office2_Easy_P000.gif) |

**Interactive 3D `.ply` Splat Viewer (PlayCanvas SuperSplat):**
- [`office_Easy_P001_frame_120_145.ply`](https://playcanvas.com/supersplat/editor?load=https://huggingface.co/hasanfaesal/NoPoSplat-TartanAir/resolve/main/v2/splats/office_Easy_P001_frame_120_145.ply)
- [`office_Hard_P000_frame_120_145.ply`](https://playcanvas.com/supersplat/editor?load=https://huggingface.co/hasanfaesal/NoPoSplat-TartanAir/resolve/main/v2/splats/office_Hard_P000_frame_120_145.ply)
- [`carwelding_Easy_P001_frame_120_145.ply`](https://playcanvas.com/supersplat/editor?load=https://huggingface.co/hasanfaesal/NoPoSplat-TartanAir/resolve/main/v2/splats/carwelding_Easy_P001_frame_120_145.ply)
- [`office2_Easy_P000_frame_94_119.ply`](https://playcanvas.com/supersplat/editor?load=https://huggingface.co/hasanfaesal/NoPoSplat-TartanAir/resolve/main/v2/splats/office2_Easy_P000_frame_94_119.ply) *(100% unseen environment)*

---

## Motivation

Feed-forward generalizable 3D Gaussian Splatting models like **NoPoSplat** reconstruct 3D scenes and relative camera poses directly from sparse unposed image pairs without COLMAP. However, foundation checkpoints pretrained on forward-facing real-estate videos (**RealEstate10K**) degrade sharply on robotics trajectories featuring rapid 6-DoF yaw/pitch/roll rotations, low-light industrial geometry, and wide baselines.

This repository adapts NoPoSplat to **TartanAir** with:
1. A clean, leak-free TartanAir-to-NoPoSplat data pipeline (`NED → OpenCV` similarity transform + canonical view-0 pose normalization).
2. Fine-tuning on 22 diverse TartanAir trajectories (`18,136` frames across `office/{Easy,Hard}` and `carwelding/{Easy,Hard}`) warm-started from `re10k.ckpt`.
3. A deterministic, reproducible 60-pair evaluation suite spanning **Novel View Synthesis** (pose-aligned & unaligned), **Relative Pose Estimation** (PnP-RANSAC + 200-step photometric/SSIM refinement), **cross-environment zero-shot transfer**, and **inference efficiency**.
4. Modern stack compatibility fixes for **PyTorch 2.4–2.14**, **CUDA 12/13 (C++20)**, **NumPy 2.0+**, and **Matplotlib 3.8+**.

---

## Quantitative Benchmark Results

All models are evaluated identically on **60 deterministic context pairs (180 target views)** across **4 held-out TartanAir trajectories** (`assets/evaluation_index_tartanair.json`), evenly stratified into **Small ($\Delta=15$)**, **Medium ($\Delta=35$)**, and **Wide ($\Delta=65$)** frame baselines on an NVIDIA A100-SXM4-40GB.

### 1. Overall Performance (60 Pairs, 180 Target Views)

| Model | Aligned PSNR ↑ | Aligned SSIM ↑ | Aligned LPIPS ↓ | Unaligned PSNR ↑ | Median $e_R$ (°) ↓ | Median $e_t$ (°) ↓ | AUC@5° ↑ | AUC@10° ↑ | AUC@20° ↑ | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `re10k_zeroshot` (baseline) | 15.38 | 0.520 | 0.500 | 11.70 | 3.81° | 9.87° | 6.1% | 21.6% | 40.0% | 91.7 |
| `tartanair_step2000` | 17.34 | 0.593 | 0.456 | 13.15 | **1.62°** | 2.43° | 38.8% | 57.4% | 69.3% | 86.1 |
| `tartanair_step4000` *(best pose)* | 17.37 | 0.592 | **0.451** | 13.36 | 1.96° | **1.75°** | **40.2%** | **57.8%** | **70.9%** | **85.4** |
| `tartanair_step5000` *(best NVS)* | **17.40** | **0.599** | **0.451** | **13.47** | 1.85° | 2.14° | 38.3% | 54.8% | 68.3% | **85.4** |

![Benchmark Comparison](https://huggingface.co/hasanfaesal/NoPoSplat-TartanAir/resolve/main/v2/plots/benchmark_comparison.png)

### 2. Per-Scene Breakdown (Including 100% Unseen-Environment Transfer)

| Scene (Regime) | Model | Aligned PSNR ↑ | Aligned SSIM ↑ | Aligned LPIPS ↓ | AUC@5° ↑ | AUC@10° ↑ | AUC@20° ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `office/Easy/P001` *(in-domain Easy)* | `re10k_zeroshot` | 18.45 | 0.717 | 0.371 | 3.5% | 11.8% | 27.3% |
| | `tartanair_step4000` | **22.11** (+3.66) | **0.779** | **0.293** | 46.9% | **66.1%** | **76.4%** |
| | `tartanair_step5000` | 21.64 (+3.19) | 0.767 | 0.298 | **49.6%** | 64.8% | 76.3% |
| `office/Hard/P000` *(in-domain Hard 6-DoF)* | `re10k_zeroshot` | 11.74 | 0.463 | 0.619 | 12.7% | 32.3% | 55.1% |
| | `tartanair_step2000` | 12.43 (+0.69) | 0.490 | 0.594 | **39.9%** | **58.6%** | **71.7%** |
| | `tartanair_step5000` | **13.13** (+1.39) | **0.541** | **0.593** | 26.7% | 42.2% | 58.0% |
| `carwelding/Easy/P001` *(industrial unseen traj.)* | `re10k_zeroshot` | 14.36 | 0.342 | 0.532 | 6.4% | 15.4% | 33.6% |
| | `tartanair_step4000` | **15.61** (+1.25) | **0.423** | **0.481** | **31.3%** | **51.2%** | **67.0%** |
| | `tartanair_step5000` | 15.40 (+1.04) | 0.419 | 0.484 | 26.9% | 44.8% | 62.2% |
| **`office2/Easy/P000`** ***(100% unseen env)*** | `re10k_zeroshot` | 16.95 | 0.559 | 0.478 | 10.4% | 35.1% | 50.9% |
| | `tartanair_step4000` | 19.33 (+2.38) | 0.661 | 0.432 | **63.5%** | **74.4%** | **83.2%** |
| | `tartanair_step5000` | **19.45** (+2.50) | **0.667** | **0.426** | 57.3% | 72.8% | 83.0% |

### 3. Breakdown by Baseline Width

| Baseline Gap | Model | Aligned PSNR ↑ | Aligned SSIM ↑ | Aligned LPIPS ↓ | Median $e_R$ (°) ↓ | Median $e_t$ (°) ↓ | AUC@20° ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Small** ($\Delta=15$) | `re10k_zeroshot` | 18.90 | 0.631 | 0.377 | 2.64° | 9.08° | 44.8% |
| | `tartanair_step4000` | **22.61** (+3.71) | 0.747 | **0.291** | **0.00°** | **0.73°** | **84.1%** |
| | `tartanair_step5000` | 22.42 (+3.52) | **0.751** | **0.291** | 0.09° | 1.87° | 79.6% |
| **Medium** ($\Delta=35$) | `re10k_zeroshot` | 14.96 | 0.508 | 0.529 | 4.11° | 7.31° | 49.6% |
| | `tartanair_step4000` | 17.06 (+2.10) | 0.592 | 0.478 | 2.33° | 1.68° | **75.1%** |
| | `tartanair_step5000` | **17.17** (+2.21) | **0.594** | **0.474** | **1.76°** | **1.32°** | 73.9% |
| **Wide** ($\Delta=65$) | `re10k_zeroshot` | 12.27 | 0.422 | 0.593 | 5.33° | 19.72° | 30.0% |
| | `tartanair_step4000` | 12.45 (+0.18) | 0.438 | **0.584** | **2.76°** | **5.50°** | **56.6%** |
| | `tartanair_step5000` | **12.62** (+0.35) | **0.451** | 0.586 | 4.17° | 6.98° | 53.9% |

---

## Qualitative Render Comparison & Training Diagnostics

<p align="center">
  <img src="https://huggingface.co/hasanfaesal/NoPoSplat-TartanAir/resolve/main/v2/plots/qualitative_grid.png" alt="Qualitative Render Comparison" width="100%">
</p>

| Training Convergence (`train.log`) | A100-40GB Hardware Telemetry (`gpu_telemetry.csv`) |
| :---: | :---: |
| ![Training Loss Curve](https://huggingface.co/hasanfaesal/NoPoSplat-TartanAir/resolve/main/v2/plots/training_loss_curve.png) | ![A100 Telemetry](https://huggingface.co/hasanfaesal/NoPoSplat-TartanAir/resolve/main/v2/plots/a100_telemetry.png) |

---

## Key Engineering Fixes & Diagnostic Findings

During adaptation from RealEstate10K to TartanAir, we identified and resolved four critical failure modes (documented in commits `c10ee83`–`b3f992c`):

1. **NED → OpenCV Similarity Transform (`src/scripts/convert_tartanair.py`):** Converting TartanAir's NED poses requires a full similarity transform $R_{\text{c2w}} = R_{\text{ned2cam}} R_{\text{ned}} R_{\text{ned2cam}}^\top$ alongside $t_{\text{c2w}} = R_{\text{ned2cam}} t_{\text{ned}}$. Omitting the left-multiplication leaves relative rotation invariant while cyclically permuting relative translation by ~90° ($e_t \approx 76^\circ\text{–}89^\circ$).
2. **Canonical View-0 Pose Normalization (`config/dataset/tartanair.yaml`):** NoPoSplat (`pose_free: true`) predicts 3D Gaussians in view-0 canonical space. Setting `relative_pose: true` and `make_baseline_1: true` is mandatory; passing raw world-frame coordinates causes target cameras to render empty space, triggering silent NaN divergence masked by finite black-image MSE/LPIPS loss (~0.45).
3. **Training Stability (`config/experiment/tartanair_h100.yaml`):** Added `gradient_clip_val: 1.0`, reduced fine-tuning LR to `5e-5` (`backbone_lr_multiplier: 0.1`), and verified `0 / 1,145` NaN tensors across every saved checkpoint.
4. **Modern PyTorch / CUDA / NumPy Compatibility:** Fixed `curope/kernels.cu` (`scalar_type()`), PyTorch 2.4+ `weights_only=False` checkpoint loading, `eval_pose.py` encoder-only key filtering, NumPy 2.0 `np.trapezoid` fallback in `cam_utils.pose_auc`, and Matplotlib 3.8+ `matplotlib.colormaps`.

---

## Pretrained Checkpoints

Verified 0-NaN checkpoints are hosted on Hugging Face at [**`hasanfaesal/NoPoSplat-TartanAir`**](https://huggingface.co/hasanfaesal/NoPoSplat-TartanAir):

| Checkpoint | Steps | Aligned PSNR ↑ | Pose AUC@10° ↑ | Pose AUC@20° ↑ | Recommended Use |
| :--- | :---: | :---: | :---: | :---: | :--- |
| [`v2/checkpoints/epoch_249-step_4000.ckpt`](https://huggingface.co/hasanfaesal/NoPoSplat-TartanAir/resolve/main/v2/checkpoints/epoch_249-step_4000.ckpt) | 4,000 | 17.37 | **57.8%** | **70.9%** | Best 6-DoF relative pose accuracy |
| [`v2/checkpoints/epoch_312-step_5000.ckpt`](https://huggingface.co/hasanfaesal/NoPoSplat-TartanAir/resolve/main/v2/checkpoints/epoch_312-step_5000.ckpt) | 5,000 | **17.40** | 54.8% | 68.3% | Best novel view synthesis quality |

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/hasanfaesal/NoPoSplat.git
cd NoPoSplat
pip install -r requirements.txt

# Compile CroCo RoPE CUDA kernel
cd src/model/encoder/backbone/croco/curope
python setup.py build_ext --inplace
cd ../../../../../..
```

*(On PyTorch 2.14+ / GCC 13+ / C++20, install `diff-gaussian-rasterization-w-pose` with `--no-build-isolation` after guarding `lerp(float,float,float)` in `helper_math.h` and adding `#include <cstdint>` to `rasterizer_impl.h`.)*

### 2. Convert TartanAir Trajectories

```bash
python -m src.scripts.convert_tartanair \
  --input_dir /path/to/tartanair_raw \
  --output_dir datasets/tartanair
```

### 3. Fine-Tune from `re10k.ckpt`

```bash
# Download base RE10K weights into pretrained_weights/re10k.ckpt
python -m src.main +experiment=tartanair_h100 \
  data_loader.train.batch_size=6 \
  trainer.max_steps=6000 \
  trainer.gradient_clip_val=1.0
```

### 4. Reproduce the 60-Pair TartanAir Benchmark

```bash
python scripts/eval_tartanair.py \
  --repo-dir . \
  --data-dir datasets/tartanair/test \
  --out-dir outputs/tartanair_eval \
  --ckpt re10k_zeroshot=pretrained_weights/re10k.ckpt \
  --ckpt tartanair_step4000=pretrained_weights/epoch_249-step_4000.ckpt \
  --ckpt tartanair_step5000=pretrained_weights/epoch_312-step_5000.ckpt \
  --pairs-per-bucket 5
```

---

## Acknowledgements & Citation

This project builds directly upon the official [**NoPoSplat**](https://github.com/cvg/NoPoSplat) codebase (ETH Zürich / NVIDIA, ICLR 2025 Oral) and the [**TartanAir**](https://theairlab.org/tartanair-dataset/) dataset (CMU AirLab, IROS 2020).

```bibtex
@inproceedings{ye2025noposplat,
  title     = {No Pose, No Problem: Surprisingly Simple 3D Gaussian Splats from Sparse Unposed Images},
  author    = {Ye, Botao and Liu, Sifei and Xu, Haofei and Li, Xueting and Pollefeys, Marc and Yang, Ming-Hsuan and Peng, Songyou},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2025}
}

@inproceedings{wang2020tartanair,
  title     = {TartanAir: A Dataset to Push the Limits of Visual SLAM},
  author    = {Wang, Wenshan and Zhu, Delong and Wang, Xiangwei and Hu, Yaoyu and Qiu, Yuheng and Wang, Chen and Hu, Yafei and Kapoor, Ashish and Scherer, Sebastian},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2020}
}
```
