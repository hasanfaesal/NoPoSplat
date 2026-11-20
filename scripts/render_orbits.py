#!/usr/bin/env python3
"""Render smooth inline orbit/wobble GIF previews comparing Zero-Shot vs Fine-Tuned 3DGS."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from eval_tartanair import add_repo_to_path, decode_chunk_example, load_model, make_batch


@torch.no_grad()
def render_wobble_frames(encoder, decoder, batch, num_frames: int = 36, radius_scale: float = 0.12) -> list[Image.Image]:
    from src.visualization.camera_trajectory.wobble import generate_wobble_transformation

    device = batch["context"]["image"].device
    gaussians = encoder(batch["context"], 0)
    # Canonical frame of view 0 is identity in NoPoSplat
    base_ext = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)  # [1, 4, 4]
    t = torch.linspace(0, 1, num_frames + 1, device=device)[:-1]
    # Smooth circular wobble in the camera XY plane
    radius = torch.tensor([radius_scale], dtype=torch.float32, device=device)
    tf = generate_wobble_transformation(radius, t, num_rotations=1, scale_radius_with_t=False)  # [1, T, 4, 4]
    exts = base_ext.unsqueeze(1) @ tf  # [1, T, 4, 4]

    intr = batch["context"]["intrinsics"][:, 0:1].expand(1, num_frames, 3, 3)
    near = batch["context"]["near"][:, 0:1].expand(1, num_frames)
    far = batch["context"]["far"][:, 0:1].expand(1, num_frames)
    out = decoder.forward(gaussians, exts, intr, near, far, (256, 256))

    frames = []
    for i in range(num_frames):
        arr = (out.color[0, i].detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        frames.append(Image.fromarray(arr))
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--ckpt", action="append", required=True, help="name=path")
    parser.add_argument("--num-frames", type=int, default=36)
    args = parser.parse_args()

    add_repo_to_path(args.repo_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scene_store = {}
    for chunk_path in sorted(args.data_dir.glob("*.torch")):
        for ex in torch.load(chunk_path, map_location="cpu", weights_only=False):
            key, ext, intr, imgs = decode_chunk_example(ex)
            scene_store[key] = (ext, intr, imgs)

    # Pick 1 representative medium-baseline pair per scene
    showcase = []
    for scene, (ext, intr, imgs) in sorted(scene_store.items()):
        c0 = min(150, max(10, len(imgs) // 4))
        c1 = min(c0 + 30, len(imgs) - 1)
        showcase.append((scene, [c0, c1], [(c0 + c1) // 2]))

    # Render frames per model
    per_scene_frames: dict[str, dict[str, list[Image.Image]]] = {s: {} for s, _, _ in showcase}
    model_names = []
    for spec in args.ckpt:
        name, ckpt_str = spec.split("=", 1)
        model_names.append(name)
        encoder, decoder, _, data_shim = load_model(args.repo_dir, Path(ckpt_str), device)
        for scene, ctx, tgt in showcase:
            ext, intr, imgs = scene_store[scene]
            batch = make_batch(scene, ext, intr, imgs, ctx, tgt, data_shim, device)
            per_scene_frames[scene][name] = render_wobble_frames(encoder, decoder, batch, num_frames=args.num_frames)
        del encoder, decoder
        torch.cuda.empty_cache()

    # Stitch side-by-side labeled GIFs
    for scene, _, _ in showcase:
        combined = []
        w, h, hdr = 256, 256, 26
        for fi in range(args.num_frames):
            canvas = Image.new("RGB", (w * len(model_names), h + hdr), (20, 20, 20))
            draw = ImageDraw.Draw(canvas)
            for mi, name in enumerate(model_names):
                draw.text((mi * w + 8, 6), name, fill=(240, 240, 240))
                canvas.paste(per_scene_frames[scene][name][fi], (mi * w, hdr))
            combined.append(canvas)
        gif_path = args.out_dir / f"orbit_{scene.replace('/', '_')}.gif"
        combined[0].save(gif_path, save_all=True, append_images=combined[1:], duration=65, loop=0, optimize=True)
        print(f"Saved {gif_path}")
        if args.backup_dir is not None:
            args.backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(gif_path, args.backup_dir / gif_path.name)


if __name__ == "__main__":
    main()
