"""
datasets/transforms.py

Data augmentation for livestock density estimation training.

Design constraints (from experiment design):
  - Input: image + density map PAIR — transforms must be applied IDENTICALLY to both
  - Spatial transforms (flip, crop) shift point locations → must warp density map too
  - Color transforms (jitter, normalize) apply to IMAGE ONLY — density map is unaffected
  - After any spatial transform, density map integral must still equal animal count

Two transform pipelines are defined:
  - TrainTransforms: full augmentation (spatial + color) for training set
  - ValTransforms:   resize + normalize only (no augmentation) for val/test

All transforms operate on (image: PIL.Image, density: np.ndarray) pairs.
"""

import random
import numpy as np
import cv2
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch
from PIL import Image


# ─────────────────────────────────────────────
# Paired spatial transforms (image + density)
# ─────────────────────────────────────────────

class PairedResize:
    """
    Resize both image and density map to target_size (H, W).
    Rescales density values so the integral (count) is preserved after resize.
    """
    def __init__(self, target_size: tuple):
        self.H, self.W = target_size

    def __call__(self, image: Image.Image, density: np.ndarray):
        # Resize image
        image = image.resize((self.W, self.H), Image.BILINEAR)

        # Resize density map and restore count
        count_before = density.sum()
        density = cv2.resize(density.astype(np.float32),
                             (self.W, self.H),
                             interpolation=cv2.INTER_CUBIC)
        if density.sum() > 0 and count_before > 0:
            density *= count_before / density.sum()   # preserve integral

        return image, density


class PairedRandomHorizontalFlip:
    """
    Randomly flip image and density map left-right with probability p.
    Spatial flip: density map pixel at (x,y) moves to (W-1-x, y) — handled by np.fliplr.
    """
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: Image.Image, density: np.ndarray):
        if random.random() < self.p:
            image   = TF.hflip(image)
            density = np.fliplr(density).copy()   # .copy() to fix negative strides
        return image, density


class PairedRandomVerticalFlip:
    """
    Randomly flip image and density map top-bottom with probability p.
    Useful for top-down pen images where orientation is arbitrary.
    """
    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, image: Image.Image, density: np.ndarray):
        if random.random() < self.p:
            image   = TF.vflip(image)
            density = np.flipud(density).copy()
        return image, density


class PairedRandomCrop:
    """
    Random crop of size crop_size from both image and density map.
    The crop region is identical for both, preserving spatial correspondence.
    Density map integral changes after crop (animals outside crop are excluded) —
    this is CORRECT behavior: we only count animals in the cropped region.
    """
    def __init__(self, crop_size: tuple):
        self.crop_H, self.crop_W = crop_size

    def __call__(self, image: Image.Image, density: np.ndarray):
        W, H = image.size   # PIL: (W, H)

        if H < self.crop_H or W < self.crop_W:
            # Image smaller than crop: skip crop, just resize
            image   = image.resize((self.crop_W, self.crop_H), Image.BILINEAR)
            count_before = density.sum()
            density = cv2.resize(density.astype(np.float32),
                                 (self.crop_W, self.crop_H),
                                 interpolation=cv2.INTER_CUBIC)
            if density.sum() > 0 and count_before > 0:
                density *= count_before / density.sum()   # preserve integral
            return image, density

        top  = random.randint(0, H - self.crop_H)
        left = random.randint(0, W - self.crop_W)

        image   = TF.crop(image, top, left, self.crop_H, self.crop_W)
        density = density[top:top+self.crop_H, left:left+self.crop_W].copy()

        return image, density


# ─────────────────────────────────────────────
# Image-only transforms (density map unchanged)
# ─────────────────────────────────────────────

class ImageColorJitter:
    """
    Random brightness, contrast, saturation, hue jitter — image only.
    Simulates different lighting conditions in pens (experiment note: "亮度增强").
    """
    def __init__(self, brightness=0.3, contrast=0.3,
                 saturation=0.2, hue=0.05):
        self.jitter = T.ColorJitter(brightness=brightness,
                                    contrast=contrast,
                                    saturation=saturation,
                                    hue=hue)

    def __call__(self, image: Image.Image, density: np.ndarray):
        image = self.jitter(image)
        return image, density


class ImageNormalize:
    """
    Normalize image to ImageNet mean/std (required for VGG-16 frontend).
    Converts PIL Image → torch.Tensor [C, H, W].
    Density map is converted to torch.Tensor [1, H, W] float32.
    """
    def __init__(self):
        self.to_tensor  = T.ToTensor()
        # ImageNet stats — VGG-16 was trained with these
        self.normalize  = T.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    def __call__(self, image: Image.Image, density: np.ndarray):
        image   = self.normalize(self.to_tensor(image))        # [3, H, W]
        density = torch.from_numpy(
            density.astype(np.float32)
        ).unsqueeze(0)                                          # [1, H, W]
        return image, density


# ─────────────────────────────────────────────
# Composed pipelines
# ─────────────────────────────────────────────

class TrainTransforms:
    """
    Full augmentation pipeline for training set.
    Order: Resize → RandomCrop → HFlip → VFlip → ColorJitter → Normalize

    Applies data augmentation for a small-scale livestock dataset.
    """
    def __init__(self, input_size: tuple = (512, 512)):
        H, W = input_size
        # Resize slightly larger than crop target, then crop
        # This introduces random position variation (common CSRNet practice)
        resize_to = (int(H * 1.1), int(W * 1.1))

        self.transforms = [
            PairedResize(resize_to),
            PairedRandomCrop(input_size),
            PairedRandomHorizontalFlip(p=0.5),
            PairedRandomVerticalFlip(p=0.3),
            ImageColorJitter(brightness=0.3, contrast=0.3,
                             saturation=0.2, hue=0.05),
            ImageNormalize(),
        ]

    def __call__(self, image, density):
        for t in self.transforms:
            image, density = t(image, density)
        return image, density


class ValTransforms:
    """
    Minimal pipeline for validation and test sets.
    Only resize + normalize — no randomness, fully reproducible.
    """
    def __init__(self, input_size: tuple = (512, 512)):
        self.transforms = [
            PairedResize(input_size),
            ImageNormalize(),
        ]

    def __call__(self, image, density):
        for t in self.transforms:
            image, density = t(image, density)
        return image, density
