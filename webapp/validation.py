"""
webapp/validation.py

Simple input quality check + confidence from TTA stability.

Design philosophy:
  - Model ALWAYS runs and shows results
  - No algorithmic rejection — the human correction workflow IS the quality gate
  - Quality warnings help users know when to pay extra attention
  - Confidence = TTA count stability (is the model's answer consistent?)
  - Heatmap visualization lets the user judge result quality visually
"""

import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import logging


class DomainGate:
    """Optional VGG similarity — soft signal only, used for experiment analysis."""

    def __init__(self, reference_path=None):
        self.enabled = False
        self.mean_feature = None
        self.sim_mean = 0.7
        self.sim_std = 0.1

        if reference_path and os.path.exists(reference_path):
            try:
                data = np.load(reference_path)
                self.mean_feature = data['mean_feature']
                self.sim_mean = float(data['sim_mean'])
                self.sim_std = float(data['sim_std'])
                self.enabled = True
                logging.info(f"[DomainGate] Loaded reference embeddings")
            except Exception as e:
                logging.warning(f"[DomainGate] Not loaded: {e}")

    def extract_features(self, model, tensor, device):
        tensor = tensor.to(device)
        with torch.no_grad():
            feat = model.frontend(tensor)
            feat = F.adaptive_avg_pool2d(feat, 1).squeeze().cpu().numpy()
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 0 else feat

    def compute_similarity(self, features):
        if self.mean_feature is None:
            return None
        return float(np.dot(features, self.mean_feature))


def run_tta(model, tensor, device, n_augments=3):
    """Test-time augmentation for count stability estimation."""
    model.eval()
    counts = []
    augs = [lambda x: x, lambda x: torch.flip(x, [3]), lambda x: torch.flip(x, [2])]
    for aug in augs[:n_augments]:
        with torch.no_grad():
            d = model(aug(tensor).to(device))
            counts.append(max(float(d.sum().cpu()), 0))
    m, s = np.mean(counts), np.std(counts)
    return dict(counts=[round(c, 2) for c in counts],
                mean=round(m, 2), std=round(s, 2),
                cv=round(float(s / (m + 1e-8)), 4))


def check_input_quality(image_path):
    """Check image quality. Warns but NEVER blocks."""
    img = cv2.imread(image_path)
    if img is None:
        return dict(valid=False, quality='error',
                    warnings=['无法读取图片文件，请检查格式。'])

    h, w = img.shape[:2]
    warnings = []

    # Resolution
    if h * w < 100 * 100:
        warnings.append('图片分辨率较低，可能影响准确性。')

    # Blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if blur < 15:
        warnings.append('图片较模糊，建议使用清晰图片。')

    # Brightness
    bright = float(np.mean(gray))
    if bright < 25:
        warnings.append('图片过暗，可能影响检测效果。')
    elif bright > 245:
        warnings.append('图片过亮。')

    # Quality grade
    if blur >= 80 and 25 <= bright <= 245 and h * w >= 200 * 200:
        quality = 'good'
    elif blur >= 15 and 20 <= bright <= 250:
        quality = 'acceptable'
    else:
        quality = 'poor'

    return dict(valid=True, quality=quality, warnings=warnings)


def compute_confidence(predicted_count, input_quality, tta_result,
                       domain_similarity=None):
    """
    Simple confidence based on TTA stability + input quality.

    No count-magnitude penalties.
    No rejection logic.
    Model always shows its result — user decides with heatmap + correction.
    """
    warnings = []
    needs_review = False

    # TTA stability is the primary confidence signal
    # If the model gives the same answer under augmentation, it's confident
    tta_cv = 1.0
    if tta_result:
        tta_cv = tta_result.get('cv', 1.0)

    if tta_cv < 0.05:
        level, label, color = 'high', '高置信度', '#059669'
        hint = '模型预测稳定，结果可信。建议结合热力图确认。'
    elif tta_cv < 0.12:
        level, label, color = 'high', '高置信度', '#059669'
        hint = '模型预测较为稳定。'
    elif tta_cv < 0.25:
        level, label, color = 'medium', '中置信度', '#d97706'
        hint = '模型预测基本稳定，建议参考热力图。'
    elif tta_cv < 0.40:
        level, label, color = 'medium', '中置信度', '#d97706'
        hint = '预测存在一定波动，建议人工复核。'
        warnings.append(f'TTA变异系数={tta_cv:.2f}，计数有一定波动。')
    else:
        level, label, color = 'low', '低置信度', '#dc2626'
        hint = '模型预测不稳定，建议人工核查并校正。'
        needs_review = True
        warnings.append(f'TTA变异系数={tta_cv:.2f}，计数波动较大。')

    # Input quality modifier
    if input_quality == 'poor':
        if level == 'high':
            level, label, color = 'medium', '中置信度', '#d97706'
            hint = '图片质量偏低，建议关注热力图。'
        warnings.append('图片质量不佳，可能影响检测精度。')

    # Near-zero count note
    if predicted_count < 1:
        level, label, color = 'low', '低置信度', '#dc2626'
        hint = '未检测到明显目标，请确认图片内容。'
        needs_review = True

    return dict(
        level=level, label=label, color=color, hint=hint,
        needs_review=needs_review, warnings=warnings,
        domain_similarity=round(domain_similarity, 4) if domain_similarity is not None else None,
    )


def format_count(raw_count):
    """Integer for display, float for records."""
    if raw_count < 0:
        return 0, 0.0
    return round(raw_count), round(raw_count, 2)
