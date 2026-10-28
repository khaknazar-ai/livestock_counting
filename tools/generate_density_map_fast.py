"""
tools/generate_density_map_fast.py

Fast version: generates density maps directly at output resolution (512x512)
using local Gaussian patches instead of full-image blur.

~10-20x faster than the original.

Usage:
    python tools/generate_density_map_fast.py
"""

import os
import json
import argparse
import numpy as np
from scipy.spatial import KDTree
import yaml
from tqdm import tqdm


def gaussian_kernel(size, sigma):
    """Create a 2D Gaussian kernel of given size and sigma."""
    x = np.arange(0, size, dtype=np.float32) - size // 2
    y = x[:, np.newaxis]
    kernel = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel


def generate_density_map_fast(out_h, out_w, points, img_h, img_w,
                               k=3, min_sigma=4.0, fixed_sigma=15.0):
    """Generate density map directly at output resolution using local patches."""

    density = np.zeros((out_h, out_w), dtype=np.float32)

    if len(points) == 0:
        return density

    pts = np.array(points, dtype=np.float32)
    N = len(pts)

    # Scale points to output resolution
    scale_x = out_w / img_w
    scale_y = out_h / img_h
    pts_scaled = pts.copy()
    pts_scaled[:, 0] *= scale_x
    pts_scaled[:, 1] *= scale_y

    # Compute adaptive sigma
    if N >= 4:
        tree = KDTree(pts_scaled)
        k_actual = min(k + 1, N)
        distances, _ = tree.query(pts_scaled, k=k_actual)
        sigmas = 0.3 * np.mean(distances[:, 1:], axis=1)
        sigmas = np.maximum(sigmas, min_sigma)
    else:
        sigmas = np.full(N, fixed_sigma)

    # Place local Gaussian patches
    for i in range(N):
        cx = pts_scaled[i, 0]
        cy = pts_scaled[i, 1]
        sigma = float(sigmas[i])

        # Kernel size = 6*sigma (covers 99.7% of distribution)
        ksize = int(np.ceil(sigma * 6)) | 1  # ensure odd
        half = ksize // 2

        kernel = gaussian_kernel(ksize, sigma)

        # Bounds in density map
        y_start = int(round(cy)) - half
        x_start = int(round(cx)) - half
        y_end = y_start + ksize
        x_end = x_start + ksize

        # Clip to image bounds
        ky_start = max(0, -y_start)
        kx_start = max(0, -x_start)
        ky_end = ksize - max(0, y_end - out_h)
        kx_end = ksize - max(0, x_end - out_w)

        y_start = max(0, y_start)
        x_start = max(0, x_start)
        y_end = min(out_h, y_end)
        x_end = min(out_w, x_end)

        if y_end > y_start and x_end > x_start:
            patch = kernel[ky_start:ky_end, kx_start:kx_end]
            density[y_start:y_end, x_start:x_end] += patch

    return density


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    ann_dir = cfg['paths']['annotation_dir']
    img_dir = cfg['paths']['image_dir']
    out_dir = cfg['paths']['density_map_dir']
    out_h, out_w = cfg['density_map']['output_size']
    k = cfg['density_map']['k_nearest']
    min_sigma = cfg['density_map']['min_sigma']
    fixed_sigma = cfg['density_map']['fixed_sigma']

    os.makedirs(out_dir, exist_ok=True)

    ann_files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.json')])
    if not ann_files:
        print(f"No annotations in {ann_dir}")
        return

    print(f"Processing {len(ann_files)} files -> {out_dir}")
    skipped = 0
    total_err = 0.0
    n_with_pts = 0

    for ann_file in tqdm(ann_files, desc="Density maps"):
        with open(os.path.join(ann_dir, ann_file), 'r', encoding='utf-8') as f:
            ann = json.load(f)

        points = ann.get('points', [])
        count = ann.get('count', len(points))
        image_file = ann.get('image_file', '')

        # Get image dimensions
        img_path = None
        stem = ann_file.replace('.json', '')
        for ext in ['.jpg', '.jpeg', '.png']:
            candidate = os.path.join(img_dir, stem + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if not img_path and image_file:
            candidate = os.path.join(img_dir, image_file)
            if os.path.exists(candidate):
                img_path = candidate

        if img_path is None:
            skipped += 1
            continue

        from PIL import Image
        img = Image.open(img_path)
        img_w, img_h = img.size

        density = generate_density_map_fast(
            out_h, out_w, points, img_h, img_w,
            k=k, min_sigma=min_sigma, fixed_sigma=fixed_sigma
        )

        # Sanity check
        if len(points) > 0:
            total_err += abs(density.sum() - count)
            n_with_pts += 1

        out_path = os.path.join(out_dir, stem + '.npy')
        np.save(out_path, density)

    print(f"\nDone. Processed: {len(ann_files) - skipped} | Skipped: {skipped}")
    if n_with_pts > 0:
        print(f"Avg count error: {total_err / n_with_pts:.4f}")


if __name__ == '__main__':
    main()
