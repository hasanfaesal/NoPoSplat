#!/usr/bin/env python3
"""Deterministic NVS + Pose + Efficiency evaluation for NoPoSplat on TartanAir.

Directly reuses NoPoSplat's own encoder, decoder, data shims, pose alignment,
PnP + photometric pose refinement, and metric functions so numbers are 100%
consistent with the upstream ICLR 2025 codebase.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # NumPy 2.0+ compatibility for NoPoSplat cam_utils.pose_auc
import torch
import torch.nn as nn
import torchvision.transforms as tf
from einops import rearrange, repeat
from hydra import compose, initialize_config_dir
from PIL import Image


def add_repo_to_path(repo_dir: Path) -> None:
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))


def decode_chunk_example(example: dict) -> tuple[str, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Decode cameras -> (extrinsics c2w [N,4,4], intrinsics [N,3,3]) matching DatasetRE10k."""
    poses = example["cameras"]
    n = poses.shape[0]
    intrinsics = repeat(torch.eye(3, dtype=torch.float32), "h w -> b h w", b=n).clone()
    fx, fy, cx, cy = poses[:, :4].T
    intrinsics[:, 0, 0], intrinsics[:, 1, 1] = fx, fy
    intrinsics[:, 0, 2], intrinsics[:, 1, 2] = cx, cy

    w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=n).clone()
    w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
    return example["key"], w2c.inverse(), intrinsics, example["images"]


def decode_images(raw_imgs: list[torch.Tensor], indices: list[int]) -> torch.Tensor:
    to_tensor = tf.ToTensor()
    return torch.stack([to_tensor(Image.open(BytesIO(raw_imgs[i].numpy().tobytes()))) for i in indices])


def build_eval_pairs(
    scenes: dict[str, int],
    buckets: dict[str, int],
    pairs_per_bucket: int = 5,
    num_targets: int = 3,
) -> list[dict]:
    """Build deterministic (scene, context, target, bucket) pairs spread evenly along each trajectory."""
    pairs = []
    for scene, n_frames in sorted(scenes.items()):
        for bucket, gap in buckets.items():
            max_start = n_frames - gap - 1
            if max_start <= 10:
                continue
            starts = np.linspace(10, max_start, pairs_per_bucket, dtype=int)
            for s in starts:
                c0, c1 = int(s), int(s + gap)
                # Interpolated targets strictly between c0 and c1
                targets = np.linspace(c0, c1, num_targets + 2, dtype=int)[1:-1].tolist()
                pairs.append(
                    {
                        "scene": scene,
                        "bucket": bucket,
                        "gap": gap,
                        "context": [c0, c1],
                        "target": targets,
                    }
                )
    return pairs


def load_model(repo_dir: Path, ckpt_path: Path, device: torch.device):
    """Instantiate EncoderNoPoSplat, DecoderSplattingCUDA, losses, and data_shim from Hydra config."""
    from src.config import load_typed_root_config
    from src.dataset.data_module import get_data_shim
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder

    with initialize_config_dir(version_base=None, config_dir=str(repo_dir / "config")):
        cfg_dict = compose(
            config_name="main",
            overrides=[
                "+experiment=tartanair_h100",
                "model.encoder.pretrained_weights=''",
                "checkpointing.load=null",
            ],
        )
    set_cfg(cfg_dict)
    cfg = load_typed_root_config(cfg_dict)

    encoder, _ = get_encoder(cfg.model.encoder)
    decoder = get_decoder(cfg.model.decoder)
    losses = nn.ModuleList(get_losses(cfg.loss)).to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    enc_sd = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    missing, unexpected = encoder.load_state_dict(enc_sd, strict=False)
    print(f"Loaded {ckpt_path.name}: missing={len(missing)}, unexpected={len(unexpected)}")

    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    decoder = decoder.to(device).eval()
    data_shim = get_data_shim(encoder)
    return encoder, decoder, losses, data_shim


def make_batch(
    scene: str,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    raw_imgs: list[torch.Tensor],
    context_idx: list[int],
    target_idx: list[int],
    data_shim,
    device: torch.device,
    relative_pose: bool = True,
) -> dict:
    from src.dataset.shims.crop_shim import apply_crop_shim
    from src.misc.cam_utils import camera_normalization

    c_idx = torch.tensor(context_idx, dtype=torch.int64)
    t_idx = torch.tensor(target_idx, dtype=torch.int64)
    near, far = 0.5, 100.0

    # Normalize camera poses to view-0 canonical frame (matching NoPoSplat base_dataset.yaml)
    ext = extrinsics.clone()
    if relative_pose:
        ext = camera_normalization(ext[c_idx][0:1], ext)

    example = {
        "context": {
            "extrinsics": ext[c_idx],
            "intrinsics": intrinsics[c_idx],
            "image": decode_images(raw_imgs, context_idx),
            "near": torch.full((len(c_idx),), near, dtype=torch.float32),
            "far": torch.full((len(c_idx),), far, dtype=torch.float32),
            "index": c_idx,
            "overlap": torch.tensor([0.5], dtype=torch.float32),
        },
        "target": {
            "extrinsics": ext[t_idx],
            "intrinsics": intrinsics[t_idx],
            "image": decode_images(raw_imgs, target_idx),
            "near": torch.full((len(t_idx),), near, dtype=torch.float32),
            "far": torch.full((len(t_idx),), far, dtype=torch.float32),
            "index": t_idx,
        },
        "scene": scene,
    }
    example = apply_crop_shim(example, (256, 256))
    batch = {
        "context": {k: v.unsqueeze(0).to(device) for k, v in example["context"].items()},
        "target": {k: v.unsqueeze(0).to(device) for k, v in example["target"].items()},
        "scene": [scene],
    }
    return data_shim(batch)


def render_with_pose_alignment(
    decoder, losses, gaussians, batch, steps: int = 100, rot_lr: float = 0.005, trans_lr: float = 0.005
):
    """Exact replica of ModelWrapper.test_step_align."""
    from src.misc.cam_utils import update_pose

    b, v, _, h, w = batch["target"]["image"].shape
    device = batch["target"]["image"].device
    cam_rot_delta = nn.Parameter(torch.zeros([b, v, 3], device=device))
    cam_trans_delta = nn.Parameter(torch.zeros([b, v, 3], device=device))
    opt = torch.optim.Adam(
        [{"params": [cam_rot_delta], "lr": rot_lr}, {"params": [cam_trans_delta], "lr": trans_lr}]
    )
    extrinsics = batch["target"]["extrinsics"].clone()

    with torch.set_grad_enabled(True):
        for _ in range(steps):
            opt.zero_grad()
            out = decoder.forward(
                gaussians,
                extrinsics,
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                cam_rot_delta=cam_rot_delta,
                cam_trans_delta=cam_trans_delta,
            )
            loss = sum(fn.forward(out, batch, gaussians, 0) for fn in losses)
            loss.backward()
            with torch.no_grad():
                opt.step()
                new_ext = update_pose(
                    cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i"),
                    cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i"),
                    extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j"),
                )
                cam_rot_delta.data.fill_(0)
                cam_trans_delta.data.fill_(0)
                extrinsics = rearrange(new_ext, "(b v) i j -> b v i j", b=b, v=v)

    with torch.no_grad():
        return decoder.forward(
            gaussians,
            extrinsics,
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
        )


def estimate_relative_pose(decoder, losses, gaussians, vis_dump, batch, steps: int = 200):
    """Exact replica of PoseEvaluator.test_step (PnP-RANSAC + 200 Adam steps)."""
    from src.evaluation.metrics import compute_pose_error
    from src.loss.loss_ssim import ssim as ssim_fn
    from src.misc.cam_utils import get_pnp_pose, update_pose

    b, _, _, h, w = batch["context"]["image"].shape
    device = batch["context"]["image"].device
    view2_rgb = batch["context"]["image"][:, 1:2].clone() * 0.5 + 0.5

    try:
        pose_opt = get_pnp_pose(
            vis_dump["means"][0, 1].squeeze(),
            vis_dump["opacities"][0, 1].squeeze(),
            batch["context"]["intrinsics"][0, 1],
            h,
            w,
        ).to(device)
    except Exception:
        # Fallback to identity (view-0 canonical frame) if too few opaque Gaussians for PnP
        pose_opt = torch.eye(4, dtype=torch.float32, device=device)

    cam_rot_delta = nn.Parameter(torch.zeros([b, 1, 3], device=device))
    cam_trans_delta = nn.Parameter(torch.zeros([b, 1, 3], device=device))
    opt = torch.optim.Adam(
        [{"params": [cam_rot_delta], "lr": 0.005}, {"params": [cam_trans_delta], "lr": 0.005}]
    )
    extrinsics = pose_opt.unsqueeze(0).unsqueeze(0)
    target_proxy = {"target": {"image": view2_rgb}}

    with torch.set_grad_enabled(True):
        for _ in range(steps):
            opt.zero_grad()
            out = decoder.forward(
                gaussians,
                extrinsics,
                batch["context"]["intrinsics"][:, 1:2],
                batch["context"]["near"][:, 1:2],
                batch["context"]["far"][:, 1:2],
                (h, w),
                cam_rot_delta=cam_rot_delta,
                cam_trans_delta=cam_trans_delta,
            )
            total_loss = sum(fn.forward(out, target_proxy, gaussians, 0) for fn in losses)
            _, _, _, structure = ssim_fn(
                rearrange(view2_rgb, "b v c h w -> (b v) c h w"),
                rearrange(out.color, "b v c h w -> (b v) c h w"),
                size_average=True,
                data_range=1.0,
                retrun_seprate=True,
                win_size=11,
            )
            (total_loss + (1.0 - structure)).backward()
            with torch.no_grad():
                opt.step()
                new_ext = update_pose(
                    cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i"),
                    cam_trans_delta=rearrange(cam_trans_delta, "b v i -> (b v) i"),
                    extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j"),
                )
                cam_rot_delta.data.fill_(0)
                cam_trans_delta.data.fill_(0)
                extrinsics = rearrange(new_ext, "(b v) i j -> b v i j", b=b, v=1)

    # Normalize GT view-2 pose relative to view-1 (canonical frame of view-1)
    c2w_0 = batch["context"]["extrinsics"][0, 0]
    c2w_1 = batch["context"]["extrinsics"][0, 1]
    gt_rel_pose = torch.linalg.inv(c2w_0) @ c2w_1
    e_t, _, e_R = compute_pose_error(gt_rel_pose, extrinsics[0, 0])
    e_pose = torch.max(e_t, e_R)
    return float(e_R.item()), float(e_t.item()), float(e_pose.item())


def save_tensor_png(tensor_chw: torch.Tensor, path: Path) -> None:
    arr = (tensor_chw.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def aggregate_metrics(records: list[dict]) -> dict:
    from src.misc.cam_utils import pose_auc

    def summarize(rows: list[dict]) -> dict:
        if not rows:
            return {}
        e_pose = np.array([r["e_pose"] for r in rows])
        auc = pose_auc(e_pose, [5, 10, 20])
        return {
            "n_pairs": len(rows),
            "nvs_aligned": {
                "psnr": float(np.mean([r["psnr_aligned"] for r in rows])),
                "ssim": float(np.mean([r["ssim_aligned"] for r in rows])),
                "lpips": float(np.mean([r["lpips_aligned"] for r in rows])),
            },
            "nvs_unaligned": {
                "psnr": float(np.mean([r["psnr_unaligned"] for r in rows])),
                "ssim": float(np.mean([r["ssim_unaligned"] for r in rows])),
                "lpips": float(np.mean([r["lpips_unaligned"] for r in rows])),
            },
            "pose": {
                "e_R_deg_mean": float(np.mean([r["e_R"] for r in rows])),
                "e_R_deg_median": float(np.median([r["e_R"] for r in rows])),
                "e_t_deg_mean": float(np.mean([r["e_t"] for r in rows])),
                "e_t_deg_median": float(np.median([r["e_t"] for r in rows])),
                "auc_5": float(auc[0] * 100.0),
                "auc_10": float(auc[1] * 100.0),
                "auc_20": float(auc[2] * 100.0),
            },
            "latency_ms_mean": float(np.mean([r["enc_ms"] + r["dec_ms"] for r in rows])),
        }

    return {
        "overall": summarize(records),
        "by_bucket": {
            b: summarize([r for r in records if r["bucket"] == b])
            for b in sorted({r["bucket"] for r in records})
        },
        "by_scene": {
            s: summarize([r for r in records if r["scene"] == s])
            for s in sorted({r["scene"] for r in records})
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, default=None, help="Optional Google Drive mirror dir")
    parser.add_argument("--ckpt", action="append", required=True, help="name=path/to/ckpt.ckpt")
    parser.add_argument("--pairs-per-bucket", type=int, default=5)
    parser.add_argument("--align-steps", type=int, default=100)
    parser.add_argument("--pose-steps", type=int, default=200)
    parser.add_argument("--save-renders", type=int, default=6, help="Number of pairs to save side-by-side PNGs")
    args = parser.parse_args()

    add_repo_to_path(args.repo_dir)
    from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load all .torch chunks in data_dir
    scene_store = {}
    for chunk_path in sorted(args.data_dir.glob("*.torch")):
        for ex in torch.load(chunk_path, map_location="cpu", weights_only=False):
            key, extrinsics, intrinsics, imgs = decode_chunk_example(ex)
            scene_store[key] = (extrinsics, intrinsics, imgs)
    print(f"Loaded {len(scene_store)} scenes from {args.data_dir}: {list(scene_store.keys())}")

    buckets = {"small": 15, "medium": 35, "wide": 65}
    index_path = args.out_dir / "evaluation_index_tartanair.json"
    if index_path.exists():
        pairs = json.loads(index_path.read_text())
    else:
        pairs = build_eval_pairs(
            {k: len(v[2]) for k, v in scene_store.items()},
            buckets=buckets,
            pairs_per_bucket=args.pairs_per_bucket,
        )
        index_path.write_text(json.dumps(pairs, indent=2))
    print(f"Evaluating {len(pairs)} deterministic pairs per model.")

    render_indices = set(np.linspace(0, len(pairs) - 1, min(args.save_renders, len(pairs)), dtype=int).tolist())
    summary = {}

    for spec in args.ckpt:
        name, ckpt_str = spec.split("=", 1)
        ckpt_path = Path(ckpt_str)
        model_out = args.out_dir / f"metrics_{name}.json"

        if model_out.exists():
            print(f"\n========== Skipping {name} (cached at {model_out.name}) ==========")
            cached = json.loads(model_out.read_text())
            summary[name] = cached["summary"]
            continue

        print(f"\n========== Evaluating {name} ({ckpt_path.name}) ==========")
        torch.cuda.reset_peak_memory_stats(device)
        encoder, decoder, losses, data_shim = load_model(args.repo_dir, ckpt_path, device)
        records = []

        for idx, pair in enumerate(pairs):
            scene = pair["scene"]
            extrinsics, intrinsics, imgs = scene_store[scene]
            batch = make_batch(scene, extrinsics, intrinsics, imgs, pair["context"], pair["target"], data_shim, device)
            rgb_gt = batch["target"]["image"][0]

            # 1. Forward encoder (timed)
            vis_dump = {}
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                gaussians = encoder(batch["context"], 0, visualization_dump=vis_dump)
            torch.cuda.synchronize()
            enc_ms = (time.perf_counter() - t0) * 1000.0

            # 2. Unaligned NVS render (timed)
            _, _, _, h, w = batch["target"]["image"].shape
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            with torch.no_grad():
                out_unaligned = decoder.forward(
                    gaussians,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                )
            torch.cuda.synchronize()
            dec_ms = (time.perf_counter() - t1) * 1000.0

            # 3. Pose-aligned NVS render (NoPoSplat standard protocol)
            out_aligned = render_with_pose_alignment(decoder, losses, gaussians, batch, steps=args.align_steps)

            # 4. Relative pose estimation (PnP + 200 Adam steps)
            e_R, e_t, e_pose = estimate_relative_pose(decoder, losses, gaussians, vis_dump, batch, steps=args.pose_steps)

            rec = {
                **pair,
                "psnr_unaligned": float(compute_psnr(rgb_gt, out_unaligned.color[0]).mean().item()),
                "ssim_unaligned": float(compute_ssim(rgb_gt, out_unaligned.color[0]).mean().item()),
                "lpips_unaligned": float(compute_lpips(rgb_gt, out_unaligned.color[0]).mean().item()),
                "psnr_aligned": float(compute_psnr(rgb_gt, out_aligned.color[0]).mean().item()),
                "ssim_aligned": float(compute_ssim(rgb_gt, out_aligned.color[0]).mean().item()),
                "lpips_aligned": float(compute_lpips(rgb_gt, out_aligned.color[0]).mean().item()),
                "e_R": e_R,
                "e_t": e_t,
                "e_pose": e_pose,
                "enc_ms": enc_ms,
                "dec_ms": dec_ms,
            }
            records.append(rec)

            if idx in render_indices:
                rdir = args.out_dir / "renders" / f"pair_{idx:03d}_{scene.replace('/', '_')}_{pair['bucket']}"
                # Save context & GT once, and each model's middle target prediction
                mid = len(pair["target"]) // 2
                if not (rdir / "gt.png").exists():
                    save_tensor_png(batch["context"]["image"][0, 0] * 0.5 + 0.5, rdir / "ctx0.png")
                    save_tensor_png(batch["context"]["image"][0, 1] * 0.5 + 0.5, rdir / "ctx1.png")
                    save_tensor_png(rgb_gt[mid], rdir / "gt.png")
                save_tensor_png(out_aligned.color[0, mid], rdir / f"{name}_aligned.png")
                save_tensor_png(out_unaligned.color[0, mid], rdir / f"{name}_unaligned.png")

            if (idx + 1) % 10 == 0 or (idx + 1) == len(pairs):
                print(
                    f"[{name}] {idx+1}/{len(pairs)} | "
                    f"Aligned PSNR={rec['psnr_aligned']:.2f} | "
                    f"Unaligned PSNR={rec['psnr_unaligned']:.2f} | "
                    f"e_R={e_R:.2f}° e_t={e_t:.2f}°"
                )

        peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
        agg = aggregate_metrics(records)
        agg["peak_vram_gb"] = float(peak_vram_gb)
        payload = {"model": name, "checkpoint": ckpt_path.name, "summary": agg, "records": records}
        model_out.write_text(json.dumps(payload, indent=2))
        summary[name] = agg

        if args.backup_dir is not None:
            args.backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model_out, args.backup_dir / model_out.name)
            shutil.copy2(index_path, args.backup_dir / index_path.name)

        del encoder, decoder, losses
        torch.cuda.empty_cache()

    summary_path = args.out_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    if args.backup_dir is not None:
        shutil.copy2(summary_path, args.backup_dir / summary_path.name)
        if (args.out_dir / "renders").exists():
            shutil.copytree(args.out_dir / "renders", args.backup_dir / "renders", dirs_exist_ok=True)
    print(f"\nAll evaluations complete. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
