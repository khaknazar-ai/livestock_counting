"""
inference.py

Single-image inference: image → count estimate + density heatmap PNG.

This is the module called by the Web prototype (webapp/app.py).
It is also useful for manual testing during development.

Inference flow: preprocess image → predict density map → sum the density map to estimate the livestock count.

Usage:
    python inference.py --image path/to/pen_image.jpg
    python inference.py --image path/to/image.jpg \
                        --checkpoint experiments/logs/checkpoint_best.pth \
                        --out_dir results/
"""

import os
import argparse
import time
import yaml
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch

from models.csrnet import CSRNet
from models.msa_csrnet import MSACSRNet


# ─────────────────────────────────────────────
# Preprocessing (matches ValTransforms, no torch dependency for web use)
# ─────────────────────────────────────────────

# ImageNet normalization constants (VGG-16 trained with these)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_path: str,
               input_size: tuple = (512, 512)) -> tuple:
    """
    Load and preprocess a single image for inference.

    Returns:
        tensor: torch.Tensor [1, 3, H, W]  — ready for model
        orig:   np.ndarray   (H_orig, W_orig, 3) BGR — for visualization
    """
    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    H, W = input_size
    img_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (W, H)).astype(np.float32) / 255.0

    # ImageNet normalization
    img_norm = (img_resized - IMAGENET_MEAN) / IMAGENET_STD   # (H, W, 3)
    img_chw  = img_norm.transpose(2, 0, 1)                    # (3, H, W)
    tensor   = torch.from_numpy(img_chw).unsqueeze(0)         # (1, 3, H, W)

    return tensor, orig


# ─────────────────────────────────────────────
# Heatmap visualization
# ─────────────────────────────────────────────

def density_to_heatmap(density: np.ndarray,
                        target_size: tuple,
                        colormap: str = 'jet') -> np.ndarray:
    """
    Convert a raw density map to a colored heatmap BGR image.

    Args:
        density:     (H_d, W_d) float32 density map from model
        target_size: (H, W) of original image for resize
        colormap:    matplotlib colormap name

    Returns:
        heatmap: (H, W, 3) uint8 BGR image
    """
    H, W = target_size

    # Resize density map to original image size
    density_up = cv2.resize(density, (W, H), interpolation=cv2.INTER_CUBIC)
    density_up = np.maximum(density_up, 0)   # clip negatives (model artifact)

    # Normalize to 0–1 for colormap
    if density_up.max() > 0:
        density_norm = density_up / density_up.max()
    else:
        density_norm = density_up

    # Apply colormap
    cmap       = cm.get_cmap(colormap)
    heatmap    = (cmap(density_norm)[:, :, :3] * 255).astype(np.uint8)
    heatmap_bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    return heatmap_bgr


def make_overlay(orig_bgr: np.ndarray,
                 density: np.ndarray,
                 alpha: float = 0.5) -> np.ndarray:
    """Blend density heatmap over original image."""
    H, W = orig_bgr.shape[:2]
    heatmap = density_to_heatmap(density, (H, W))
    overlay = cv2.addWeighted(orig_bgr, 1 - alpha, heatmap, alpha, 0)
    return overlay


def save_result_figure(orig_bgr: np.ndarray,
                        density: np.ndarray,
                        count: float,
                        out_path: str,
                        inference_ms: float = None):
    """
    Save a 3-panel result figure: original | heatmap | overlay.
    Used by webapp and for manual inspection.
    """
    H, W = orig_bgr.shape[:2]
    overlay = make_overlay(orig_bgr, density, alpha=0.5)

    orig_rgb    = cv2.cvtColor(orig_bgr,  cv2.COLOR_BGR2RGB)
    overlay_rgb = cv2.cvtColor(overlay,   cv2.COLOR_BGR2RGB)

    title = f"Estimated Count: {count:.1f}"
    if inference_ms:
        title += f"  |  Inference: {inference_ms:.0f} ms"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    axes[0].imshow(orig_rgb)
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    # Show density map with matplotlib colormap for proper colorbar
    den_up = cv2.resize(density, (W, H), interpolation=cv2.INTER_CUBIC)
    den_up = np.maximum(den_up, 0)   # clip negatives
    im2 = axes[1].imshow(den_up, cmap='jet')
    axes[1].set_title(f"Density Map  (sum={density.sum():.1f})")
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(overlay_rgb)
    axes[2].set_title("Overlay")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────
# Core inference function (called by webapp)
# ─────────────────────────────────────────────

def predict(image_path: str,
            model: torch.nn.Module,
            device: torch.device,
            input_size: tuple = (512, 512)) -> dict:
    """
    Run inference on a single image.

    Args:
        image_path: path to input image
        model:      loaded CSRNet or MSACSRNet (already on device, eval mode)
        device:     torch device
        input_size: (H, W) for model input

    Returns dict with:
        count:          float  — estimated animal count
        density_map:    np.ndarray (H_d, W_d) — raw density map
        heatmap_bgr:    np.ndarray (H, W, 3)  — colored heatmap (BGR)
        overlay_bgr:    np.ndarray (H, W, 3)  — heatmap overlay on image (BGR)
        inference_ms:   float  — inference time in milliseconds
        image_shape:    tuple  — (H_orig, W_orig) of input image
    """
    tensor, orig_bgr = preprocess(image_path, input_size)
    tensor = tensor.to(device)

    model.eval()
    t0 = time.time()
    with torch.no_grad():
        density_tensor = model(tensor)                    # [1, 1, H/8, W/8]
    inference_ms = (time.time() - t0) * 1000

    density = density_tensor.squeeze().cpu().numpy()      # (H/8, W/8)
    density = np.maximum(density, 0)                      # clip negatives

    count = float(density.sum())

    H_orig, W_orig = orig_bgr.shape[:2]
    heatmap_bgr = density_to_heatmap(density, (H_orig, W_orig))
    overlay_bgr = make_overlay(orig_bgr, density, alpha=0.5)

    return {
        'count':        count,
        'density_map':  density,
        'heatmap_bgr':  heatmap_bgr,
        'overlay_bgr':  overlay_bgr,
        'inference_ms': inference_ms,
        'image_shape':  (H_orig, W_orig),
    }


# ─────────────────────────────────────────────
# Model loader helper (used by both inference.py and webapp)
# ─────────────────────────────────────────────

def load_model(checkpoint_path: str,
               device: torch.device,
               model_name: str = None) -> torch.nn.Module:
    """
    Load a trained density model from checkpoint.
    Returns model in eval mode on device.

    Auto-detects model type from checkpoint metadata.
    Can be overridden with model_name='csrnet' or 'msa_csrnet'.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Detect model type
    if model_name is None:
        if isinstance(ckpt, dict):
            model_name = ckpt.get('model_name', None)
        if model_name is None:
            # Heuristic: check state dict keys for MSA-specific layers
            state = ckpt.get('model_state', ckpt) if isinstance(ckpt, dict) else ckpt
            if isinstance(state, dict) and any('branch1' in k or 'attention' in k for k in state.keys()):
                model_name = 'msa_csrnet'
            else:
                model_name = 'csrnet'

    if model_name == 'msa_csrnet':
        model = MSACSRNet().to(device)
    else:
        model = CSRNet().to(device)

    # Load weights
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        model.load_state_dict(ckpt['model_state'])
    elif isinstance(ckpt, dict):
        model.load_state_dict(ckpt)
    else:
        model.load_state_dict(ckpt)

    model.eval()
    return model


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run CSRNet inference on a single livestock image.'
    )
    parser.add_argument('--image',      type=str, required=True,
                        help='Path to input image (.jpg/.png)')
    parser.add_argument('--config',     type=str,
                        default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str,
                        default='experiments/logs/checkpoint_best.pth')
    parser.add_argument('--out_dir',    type=str,
                        default='results',
                        help='Directory to save output figures')
    parser.add_argument('--no_save',    action='store_true',
                        help='Print count only, do not save figures')
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        cfg['hardware']['device'] if torch.cuda.is_available() else 'cpu'
    )
    input_size = tuple(cfg['train']['input_size'])

    # Load model
    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        return

    print(f"[Inference] Loading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    # Run inference
    print(f"[Inference] Image: {args.image}")
    result = predict(args.image, model, device, input_size)

    print(f"\n  Estimated count : {result['count']:.1f}")
    print(f"  Inference time  : {result['inference_ms']:.1f} ms")
    print(f"  Image size      : {result['image_shape']}")

    if not args.no_save:
        os.makedirs(args.out_dir, exist_ok=True)
        stem     = os.path.splitext(os.path.basename(args.image))[0]
        fig_path = os.path.join(args.out_dir, f'{stem}_result.png')
        orig_bgr = cv2.imread(args.image)
        save_result_figure(orig_bgr,
                           result['density_map'],
                           result['count'],
                           fig_path,
                           result['inference_ms'])
        print(f"  Result figure   : {fig_path}")


if __name__ == '__main__':
    main()
