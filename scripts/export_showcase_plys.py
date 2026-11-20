#!/usr/bin/env python3
"""Export verified 0-NaN 3D Gaussian Splat (.ply) showcase files from a trained NoPoSplat checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from eval_tartanair import add_repo_to_path, decode_chunk_example, load_model, make_batch


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--pairs-per-scene", type=int, default=2)
    args = parser.parse_args()

    add_repo_to_path(args.repo_dir)
    from src.model.ply_export import export_ply

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    encoder, _, _, data_shim = load_model(args.repo_dir, args.ckpt, device)

    for chunk_path in sorted(args.data_dir.glob("*.torch")):
        for ex in torch.load(chunk_path, map_location="cpu", weights_only=False):
            scene, ext, intr, imgs = decode_chunk_example(ex)
            n = len(imgs)
            starts = [min(120, n // 4), min(max(200, n // 2), n - 35)]
            for s in starts[: args.pairs_per_scene]:
                c0, c1 = int(s), int(min(s + 25, n - 1))
                batch = make_batch(scene, ext, intr, imgs, [c0, c1], [(c0 + c1) // 2], data_shim, device)
                vis_dump = {}
                gaussians = encoder(batch["context"], 0, visualization_dump=vis_dump)
                assert torch.isfinite(gaussians.means).all(), f"Non-finite means in {scene}"
                assert torch.isfinite(gaussians.opacities).all(), f"Non-finite opacities in {scene}"
                out_ply = args.out_dir / f"{scene.replace('/', '_')}_frame_{c0}_{c1}.ply"
                export_ply(
                    gaussians.means[0],
                    vis_dump["scales"][0],
                    vis_dump["rotations"][0],
                    gaussians.harmonics[0],
                    gaussians.opacities[0],
                    out_ply,
                    shift_and_scale=False,
                    save_sh_dc_only=True,
                )
                print(f"Exported verified 0-NaN PLY: {out_ply.name} ({out_ply.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
