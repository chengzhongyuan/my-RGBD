#!/usr/bin/env python3
"""
Pre-compute semantic segmentation masks for TUM dataset sequences.
Used by modified ORB-SLAM2 for dynamic feature point filtering.

Paper Section 3.1: BiSeNetV2 semantic segmentation for person detection.
Output: binary mask PNG files (0=static, 255=dynamic/person)

Usage:
    python precompute_masks.py --dataset /path/to/tum/sequence --output masks/
"""

import argparse, os, sys, time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dynamic_slam.bisenetv2 import SemanticSegmenter


def main():
    parser = argparse.ArgumentParser(description='Pre-compute dynamic masks for TUM dataset')
    parser.add_argument('--dataset', required=True, help='Path to TUM dataset')
    parser.add_argument('--output', default=None, help='Output directory (default: dataset/masks/)')
    parser.add_argument('--model', default=None, help='Path to BiSeNetV2 weights')
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()

    ds = args.dataset
    out_dir = args.output or os.path.join(ds, 'masks')
    os.makedirs(out_dir, exist_ok=True)

    # Load RGB file list
    rgb_txt = os.path.join(ds, 'rgb.txt')
    with open(rgb_txt) as f:
        rgb_lines = [l.strip().split() for l in f if l.strip() and not l.startswith('#')]

    print(f"Loading semantic segmenter...")
    seg = SemanticSegmenter(model_path=args.model)
    print(f"Model: {seg.model_type}, device: {seg.device}")

    print(f"Processing {min(len(rgb_lines), args.max_frames or len(rgb_lines))} frames...")
    t_start = time.time()

    for idx, parts in enumerate(rgb_lines):
        if args.max_frames and idx >= args.max_frames:
            break

        ts = float(parts[0])
        rgb_rel = parts[1]
        rgb_path = os.path.join(ds, rgb_rel)

        rgb = cv2.imread(rgb_path)
        if rgb is None:
            print(f"  Skip: {rgb_path}")
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Semantic segmentation
        mask = seg.segment(rgb)

        # Save mask as PNG
        mask_name = f"{ts:.6f}.png"
        cv2.imwrite(os.path.join(out_dir, mask_name), mask * 255)

        if idx % 100 == 0 and idx > 0:
            elapsed = time.time() - t_start
            print(f"  Frame {idx}: {elapsed/idx*1000:.0f}ms/frame")

    elapsed = time.time() - t_start
    n = min(len(rgb_lines), args.max_frames or len(rgb_lines))
    print(f"Done: {n} masks in {elapsed:.1f}s ({elapsed/n*1000:.0f}ms/frame)")
    print(f"Masks saved to: {out_dir}")


if __name__ == '__main__':
    main()
