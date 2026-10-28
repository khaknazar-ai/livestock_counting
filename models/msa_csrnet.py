"""
models/msa_csrnet.py

MSA-CSRNet: Multi-Scale Attention CSRNet

An enhanced variant of CSRNet for the model comparison experiment.
Same VGG-16 frontend (for fair transfer-learning comparison), but the
backend replaces the single-scale dilated CNN with:

  1. Multi-Scale Feature Extraction (MSFE):
     Three parallel branches with dilation rates 1, 2, 4.
     Each branch captures livestock features at different scales —
     critical for aerial/surveillance images where animal size varies.

  2. Channel Attention Module (CAM):
     Squeeze-and-Excitation block that learns to weight feature channels.
     Helps the model focus on density-relevant features and suppress
     background noise (grass, fences, shadows).

  3. Fusion + Regression Head:
     Concatenated multi-scale features → channel attention → 1×1 conv
     regression to single-channel density map.

Architecture diagram:
                        ┌─── Branch(d=1) ──┐
  VGG-16 frontend  ──→  ├─── Branch(d=2) ──┼──→ Concat ──→ CAM ──→ 1×1 → density
  (frozen/unfrozen)      └─── Branch(d=4) ──┘

Why this design:
  - Same frontend ensures fair comparison with CSRNet
  - Multi-scale captures size variation in livestock images
  - Attention suppresses irrelevant channels (background, noise)
  - Modest parameter reduction (~10M vs ~16M) — more efficient than CSRNet

Transfer learning: identical to CSRNet.
  - Phase 1: frontend frozen, MSA backend trains
  - Phase 2: frontend unfrozen at lower LR
  - Can load CSRNet crowd-counting weights for frontend (same VGG-16)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional

from models.csrnet import FRONTEND_END_IDX


# ─────────────────────────────────────────────
# Channel Attention Module (Squeeze-and-Excitation)
# ─────────────────────────────────────────────

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation channel attention.

    Learns per-channel importance weights via:
      GAP → FC → ReLU → FC → Sigmoid → scale

    Reduction ratio controls bottleneck size.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.squeeze(x)       # [B, C, 1, 1]
        w = self.excite(w)        # [B, C, 1, 1]
        return x * w              # channel-wise scaling


# ─────────────────────────────────────────────
# Multi-Scale Dilated Branch
# ─────────────────────────────────────────────

class DilatedBranch(nn.Module):
    """
    A single dilated convolution branch.

    Conv(dilation=d) → BN → ReLU → Conv → BN → ReLU

    BN is added (vs original CSRNet) for more stable training
    with the multi-branch architecture.
    """

    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.branch(x)


# ─────────────────────────────────────────────
# MSA Backend
# ─────────────────────────────────────────────

class MSABackend(nn.Module):
    """
    Multi-Scale Attention backend.

    Three parallel branches (d=1,2,4) → concat → reduce →
    channel attention → regression head.
    """

    def __init__(self, in_ch: int = 512, branch_ch: int = 128):
        super().__init__()

        # Multi-scale branches
        self.branch1 = DilatedBranch(in_ch, branch_ch, dilation=1)
        self.branch2 = DilatedBranch(in_ch, branch_ch, dilation=2)
        self.branch4 = DilatedBranch(in_ch, branch_ch, dilation=4)

        # Fusion: 3*branch_ch → 256
        fused_ch = branch_ch * 3
        self.fusion = nn.Sequential(
            nn.Conv2d(fused_ch, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Channel attention on fused features
        self.attention = ChannelAttention(256, reduction=16)

        # Regression head: 256 → 64 → 1
        self.head = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, x):
        # Multi-scale extraction
        b1 = self.branch1(x)      # [B, 128, H/8, W/8]
        b2 = self.branch2(x)      # [B, 128, H/8, W/8]
        b4 = self.branch4(x)      # [B, 128, H/8, W/8]

        # Concatenate
        cat = torch.cat([b1, b2, b4], dim=1)  # [B, 384, H/8, W/8]

        # Fuse + attend
        fused = self.fusion(cat)               # [B, 256, H/8, W/8]
        attended = self.attention(fused)        # [B, 256, H/8, W/8]

        # Regress density
        out = self.head(attended)              # [B, 1, H/8, W/8]
        return out


# ─────────────────────────────────────────────
# MSA-CSRNet
# ─────────────────────────────────────────────

class MSACSRNet(nn.Module):
    """
    MSA-CSRNet: Multi-Scale Attention CSRNet.

    Same interface as CSRNet for drop-in replacement:
      - forward(x) → density map [B, 1, H/8, W/8]
      - predict_count(x) → count [B]
      - freeze_frontend() / unfreeze_frontend()
      - load_vgg16_weights() / load_csrnet_weights()
    """

    def __init__(self, branch_ch: int = 128):
        super().__init__()

        # ── Frontend: same VGG-16 as CSRNet ──
        try:
            vgg16 = models.vgg16(weights=None)
        except TypeError:
            vgg16 = models.vgg16(pretrained=False)
        self.frontend = nn.Sequential(
            *list(vgg16.features.children())[:FRONTEND_END_IDX]
        )

        # ── Backend: MSA (replaces CSRNet's dilated backend) ──
        self.backend = MSABackend(in_ch=512, branch_ch=branch_ch)

        # ── Initialize backend weights ──
        self._init_backend_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.frontend(x)
        x = self.backend(x)
        return x

    def predict_count(self, x: torch.Tensor) -> torch.Tensor:
        density = self.forward(x)
        return density.sum(dim=[1, 2, 3])

    # ── Weight loading (same as CSRNet) ──

    def load_vgg16_weights(self, weights_path: str):
        """Load ImageNet VGG-16 weights into frontend."""
        if not torch.cuda.is_available():
            state = torch.load(weights_path, map_location='cpu')
        else:
            state = torch.load(weights_path)

        frontend_state = {}
        for k, v in state.items():
            if k.startswith('features.'):
                idx = int(k.split('.')[1])
                if idx < FRONTEND_END_IDX:
                    new_key = k[len('features.'):]
                    frontend_state[new_key] = v

        missing, unexpected = self.frontend.load_state_dict(
            frontend_state, strict=False
        )
        loaded = len(frontend_state) - len(missing)
        print(f"[MSA-CSRNet] VGG-16 frontend: loaded {loaded} tensors")

    def load_csrnet_weights(self, weights_path: str):
        """
        Load CSRNet crowd-counting weights.
        Only the FRONTEND weights are transferred (backend architecture differs).
        """
        if not torch.cuda.is_available():
            ckpt = torch.load(weights_path, map_location='cpu')
        else:
            ckpt = torch.load(weights_path)

        if isinstance(ckpt, dict):
            for key in ['state_dict', 'model_state', 'model']:
                if key in ckpt:
                    ckpt = ckpt[key]
                    break

        # Extract frontend weights only
        frontend_state = {}
        for k, v in ckpt.items():
            k_clean = k.replace('module.', '')
            if k_clean.startswith('frontend.'):
                frontend_state[k_clean[len('frontend.'):]] = v

        if not frontend_state:
            # Try 'features.' prefix
            for k, v in ckpt.items():
                k_clean = k.replace('module.', '')
                if k_clean.startswith('features.'):
                    idx = int(k_clean.split('.')[1])
                    if idx < FRONTEND_END_IDX:
                        frontend_state[k_clean[len('features.'):]] = v

        if frontend_state:
            missing, _ = self.frontend.load_state_dict(frontend_state, strict=False)
            loaded = len(frontend_state) - len(missing)
            print(f"[MSA-CSRNet] Crowd-counting frontend: loaded {loaded} tensors")
            print(f"[MSA-CSRNet] MSA backend initialized from scratch (architecture differs)")
        else:
            print(f"[MSA-CSRNet] No frontend weights found in checkpoint")

    # ── Transfer learning controls ──

    def freeze_frontend(self):
        for param in self.frontend.parameters():
            param.requires_grad = False
        print("[MSA-CSRNet] Frontend FROZEN")

    def unfreeze_frontend(self):
        for param in self.frontend.parameters():
            param.requires_grad = True
        print("[MSA-CSRNet] Frontend UNFROZEN")

    def frontend_is_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.frontend.parameters())

    def get_optimizer_param_groups(self,
                                   lr_backend: float = 1e-5,
                                   lr_frontend: float = 1e-6) -> list:
        frontend_params = [p for p in self.frontend.parameters()
                           if p.requires_grad]
        backend_params = list(self.backend.parameters())

        groups = [{'params': backend_params, 'lr': lr_backend}]
        if frontend_params:
            groups.append({'params': frontend_params, 'lr': lr_frontend})
        return groups

    # ── Info ──

    def count_parameters(self) -> dict:
        def n_params(module):
            return sum(p.numel() for p in module.parameters())
        def n_trainable(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            'frontend_total':     n_params(self.frontend),
            'frontend_trainable': n_trainable(self.frontend),
            'backend_total':      n_params(self.backend),
            'backend_trainable':  n_trainable(self.backend),
            'total':              n_params(self),
            'total_trainable':    n_trainable(self),
        }

    def print_summary(self):
        p = self.count_parameters()
        frozen = self.frontend_is_frozen()
        print("=" * 50)
        print(" MSA-CSRNet Architecture Summary")
        print("=" * 50)
        print(f"  Frontend (VGG-16 conv1_1–conv4_3)")
        print(f"    Parameters : {p['frontend_total']:,}")
        print(f"    Trainable  : {p['frontend_trainable']:,}"
              f"  {'[FROZEN]' if frozen else '[ACTIVE]'}")
        print(f"  Backend  (MSA: 3-branch dilated + channel attention)")
        print(f"    Parameters : {p['backend_total']:,}")
        print(f"    Trainable  : {p['backend_trainable']:,}  [ACTIVE]")
        print(f"  Total      : {p['total']:,}")
        print(f"  Trainable  : {p['total_trainable']:,}")
        print("=" * 50)

    def _init_backend_weights(self):
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
