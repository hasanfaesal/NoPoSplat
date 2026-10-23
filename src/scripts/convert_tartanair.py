#!/usr/bin/env python3
"""
Convert TartanAir sequences into NoPoSplat-compatible .torch chunk files and index.json.

TartanAir structure:
  root/
    ENV_NAME/
      Easy (or Hard)/
        P000/
          image_left/
            000000_left.png, 000001_left.png, ...
          pose_left.txt

TartanAir intrinsics:
  w=640, h=480, fx=320.0, fy=320.0, cx=320.0, cy=240.0
Camera poses in pose_left.txt: [tx, ty, tz, qx, qy, qz, qw] in NED frame.
Mapped to OpenCV camera-to-world (c2w), then inverted to w2c (3x4 flattened).
"""

import argparse
import io
import json
import os
from pathlib import Path
from typing import List, TypedDict

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torch import Tensor
from tqdm import tqdm

# TartanAir pinhole camera specifications
IMG_W = 640
IMG_H = 480
FX = 320.0
FY = 320.0
CX = 320.0
CY = 240.0

# Target ~100MB per chunk file to match NoPoSplat standards
TARGET_BYTES_PER_CHUNK = int(1e8)


class Metadata(TypedDict):
    url: str
    timestamps: Tensor
    cameras: Tensor


class Example(Metadata):
    key: str
    images: List[Tensor]


def ned_pose_to_opencv_w2c(pose: np.ndarray) -> np.ndarray:
    """
    Convert a single TartanAir NED pose [tx, ty, tz, qx, qy, qz, qw]
    to an OpenCV world-to-camera (w2c) 3x4 matrix.
    """
    tx, ty, tz, qx, qy, qz, qw = pose

    # Rotate from NED orientation to OpenCV camera frame (x=right, y=down, z=forward)
    rot_ned = R.from_quat([qx, qy, qz, qw]).as_matrix()

    # Transformation from NED frame to OpenCV camera frame:
    # NED: x=forward, y=right, z=down
    # OpenCV: x=right, y=down, z=forward
    R_ned2cam = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0]
    ], dtype=np.float32)

    # c2w rotation and translation in camera frame
    R_c2w = rot_ned @ R_ned2cam.T
    t_c2w = np.array([ty, tz, tx], dtype=np.float32)

    # Construct 4x4 c2w matrix
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_c2w
    c2w[:3, 3] = t_c2w

    # Invert to obtain w2c
    w2c = np.linalg.inv(c2w)
    return w2c[:3, :]  # return 3x4


def encode_image_as_jpeg_bytes(img_path: Path, quality: int = 95) -> Tensor:
    """Read image and encode to JPEG bytes in a uint8 tensor."""
    with Image.open(img_path) as img:
        img_rgb = img.convert("RGB")
        buf = io.BytesIO()
        img_rgb.save(buf, format="JPEG", quality=quality)
        raw_bytes = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        return torch.from_numpy(raw_bytes.copy())


def process_sequence(seq_dir: Path) -> Example:
    """Parse one trajectory folder (P000)."""
    img_dir = seq_dir / "image_left"
    pose_file = seq_dir / "pose_left.txt"

    if not img_dir.exists() or not pose_file.exists():
        return None

    poses = np.loadtxt(pose_file, delimiter=" ")
    img_files = sorted(list(img_dir.glob("*.png")))
    num_frames = min(len(poses), len(img_files))

    if num_frames < 10:
        return None

    # Normalized camera intrinsics: [fx/w, fy/h, cx/w, cy/h, 0, 0]
    intrinsic = np.array([FX / IMG_W, FY / IMG_H, CX / IMG_W, CY / IMG_H, 0.0, 0.0], dtype=np.float32)

    timestamps = []
    cameras = []
    images = []

    for idx in range(num_frames):
        w2c_3x4 = ned_pose_to_opencv_w2c(poses[idx])
        cam_entry = np.concatenate([intrinsic, w2c_3x4.flatten()])
        cameras.append(cam_entry)
        timestamps.append(idx)
        images.append(encode_image_as_jpeg_bytes(img_files[idx]))

    return {
        "key": str(seq_dir.relative_to(seq_dir.parents[2])),
        "url": "",
        "timestamps": torch.tensor(timestamps, dtype=torch.int64),
        "cameras": torch.tensor(np.stack(cameras), dtype=torch.float32),
        "images": images,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert TartanAir to NoPoSplat chunk format.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to TartanAir root")
    parser.add_argument("--output_dir", type=str, required=True, help="Destination dataset path")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation sequence ratio")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Collect all trajectory folders (P000, P001, etc.)
    all_seqs = sorted(list(input_dir.glob("*/*/*/image_left")))
    seq_dirs = [p.parent for p in all_seqs if (p.parent / "pose_left.txt").exists()]
    print(f"Discovered {len(seq_dirs)} valid TartanAir trajectories.")

    if len(seq_dirs) == 0:
        print("No valid trajectories found. Check --input_dir structure.")
        return

    # Deterministic train/test split
    np.random.seed(42)
    indices = np.random.permutation(len(seq_dirs))
    num_val = max(1, int(len(seq_dirs) * args.val_ratio))
    val_indices = set(indices[:num_val])

    train_seqs = [seq_dirs[i] for i in range(len(seq_dirs)) if i not in val_indices]
    val_seqs = [seq_dirs[i] for i in range(len(seq_dirs)) if i in val_indices]

    for stage, seq_list in [("train", train_seqs), ("test", val_seqs)]:
        stage_dir = output_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nProcessing {len(seq_list)} sequences for {stage} split...")

        chunk_idx = 0
        current_chunk = []
        current_bytes = 0

        for seq_path in tqdm(seq_list, desc=f"Writing {stage} chunks"):
            example = process_sequence(seq_path)
            if example is None:
                continue

            # Estimate payload bytes
            seq_bytes = sum(len(img) for img in example["images"])
            current_chunk.append(example)
            current_bytes += seq_bytes

            if current_bytes >= TARGET_BYTES_PER_CHUNK:
                torch.save(current_chunk, stage_dir / f"{chunk_idx:06d}.torch")
                chunk_idx += 1
                current_chunk = []
                current_bytes = 0

        if len(current_chunk) > 0:
            torch.save(current_chunk, stage_dir / f"{chunk_idx:06d}.torch")

        # Generate index.json
        print(f"Generating index.json for {stage}...")
        index = {}
        for chunk_file in stage_dir.glob("*.torch"):
            chunk_data = torch.load(chunk_file)
            for ex in chunk_data:
                index[ex["key"]] = chunk_file.name

        with open(stage_dir / "index.json", "w") as f:
            json.dump(index, f, indent=2)

    print("\nDataset conversion completed successfully.")


if __name__ == "__main__":
    main()
