"""
tools/visualize_density.py

Overlay a .npy density map on its source image and display/save the result.
Use this AFTER running generate_density_map.py to verify the density maps
look correct before training.

Usage:
    python tools/visualize_density.py --image data/livestock/images/img001.jpg \
                                       --density data/livestock/density_maps/img001.npy

    # Batch: check all density maps in a directory
    python tools/visualize_density.py --batch \
        --img_dir data/livestock/images \
        --den_dir data/livestock/density_maps \
        --out_dir experiments/density_checks
"""

import os
import argparse
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')   # No GUI needed; works in headless environments
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_density(npy_path: str) -> np.ndarray:
    d = np.load(npy_path)
    return d.astype(np.float32)


def overlay_density_on_image(img_bgr: np.ndarray,
                              density: np.ndarray,
                              alpha: float = 0.55,
                              colormap: str = 'jet') -> np.ndarray:
    """
    Blend a heatmap of the density map over the original image.

    Args:
        img_bgr:   Original image in BGR (OpenCV format)
        density:   (H, W) density map
        alpha:     Heatmap opacity (0=invisible, 1=fully opaque)
        colormap:  Matplotlib colormap name

    Returns:
        blended:   (H, W, 3) uint8 BGR image
    """
    H, W = img_bgr.shape[:2]

    # Resize density to match image if needed
    if density.shape != (H, W):
        density = cv2.resize(density, (W, H), interpolation=cv2.INTER_CUBIC)
        if density.sum() > 0:
            density = density / density.max()   # normalize 0-1 for display only

    # Normalize to 0-1 for colormap
    d_norm = density.copy()
    if d_norm.max() > 0:
        d_norm = d_norm / d_norm.max()

    # Apply colormap: float → RGBA (matplotlib) → uint8 BGR (OpenCV)
    cmap = cm.get_cmap(colormap)
    heatmap_rgba = cmap(d_norm)                          # (H, W, 4) float [0,1]
    heatmap_rgb  = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    heatmap_bgr  = cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR)

    img_float    = img_bgr.astype(np.float32)
    heat_float   = heatmap_bgr.astype(np.float32)
    blended      = cv2.addWeighted(img_float, 1 - alpha, heat_float, alpha, 0)

    return blended.astype(np.uint8)


def make_summary_figure(img_bgr: np.ndarray,
                         density: np.ndarray,
                         gt_count: float,
                         image_name: str,
                         save_path: str):
    """
    Create a 3-panel figure: original | density heatmap | overlay.
    Saves to save_path as PNG.
    """
    H, W = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize density for display
    den_display = cv2.resize(density, (W, H), interpolation=cv2.INTER_CUBIC)
    overlay = overlay_density_on_image(img_bgr, density)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"{image_name}  |  GT count: {gt_count:.1f}", fontsize=13)

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    im = axes[1].imshow(den_display, cmap='jet')
    axes[1].set_title(f"Density Map  (sum={density.sum():.1f})")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay_rgb)
    axes[2].set_title("Overlay")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


def visualize_single(img_path: str, den_path: str, save_path: str = None):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    density = load_density(den_path)
    gt_count = density.sum()
    image_name = os.path.basename(img_path)

    out = save_path or img_path.replace('.jpg', '_density_check.png') \
                                .replace('.jpeg', '_density_check.png') \
                                .replace('.png', '_density_check.png')

    make_summary_figure(img, density, gt_count, image_name, out)
    print(f"✓ Saved: {out}  (GT count from density: {gt_count:.1f})")


def visualize_batch(img_dir: str, den_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    den_files = sorted([f for f in os.listdir(den_dir) if f.endswith('.npy')])
    if not den_files:
        print(f"No .npy files in {den_dir}")
        return

    print(f"Visualizing {len(den_files)} density maps...")
    errors = []

    for den_file in den_files:
        stem = den_file.replace('.npy', '')
        den_path = os.path.join(den_dir, den_file)

        # Try common image extensions
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
            candidate = os.path.join(img_dir, stem + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            print(f"  [SKIP] No image found for: {stem}")
            errors.append(stem)
            continue

        save_path = os.path.join(out_dir, stem + '_check.png')
        try:
            visualize_single(img_path, den_path, save_path)
        except Exception as e:
            print(f"  [ERROR] {stem}: {e}")
            errors.append(stem)

    print(f"\n✓ Done. Output: {out_dir}")
    if errors:
        print(f"  Failed: {errors}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--image',   type=str, default=None)
    parser.add_argument('--density', type=str, default=None)
    parser.add_argument('--save',    type=str, default=None)
    parser.add_argument('--img_dir', type=str, default='data/livestock/images')
    parser.add_argument('--den_dir', type=str, default='data/livestock/density_maps')
    parser.add_argument('--out_dir', type=str, default='experiments/density_checks')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch:
        visualize_batch(args.img_dir, args.den_dir, args.out_dir)
    else:
        if not args.image or not args.density:
            print("For single-image mode: provide --image and --density")
            print("For batch mode: use --batch flag")
            return
        visualize_single(args.image, args.density, args.save)


if __name__ == '__main__':
    main()
