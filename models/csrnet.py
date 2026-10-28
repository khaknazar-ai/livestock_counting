"""
models/csrnet.py

CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly
Congested Scenes. Li et al., CVPR 2018.

Architecture (matches original paper exactly):
  Frontend: VGG-16 layers 0-22 (conv1_1 → conv4_3, NO pool4/pool5/FC layers)
            → Extracts rich spatial features at 1/8 input resolution
  Backend:  6 dilated conv layers (dilation rates 2,2,2,2,2,2)
            → Enlarges receptive field without losing spatial resolution
            → Final 1×1 conv → single-channel density map

Why this architecture for the project:
  - Frontend uses ImageNet-pretrained VGG-16: strong general feature extractor
    that already understands texture/shape → ideal for transfer learning
  - Dilated backend avoids pooling → density map at 1/8 resolution (not 1/32)
    → higher spatial precision for counting occluded, dense livestock
  - The clean frontend/backend separation directly enables the transfer-learning
    strategy: freeze frontend, fine-tune backend on livestock data

Transfer learning strategy (from experiment design):
  - Phase 1 (epochs 0→freeze_epoch):  frontend FROZEN, backend trained freely
  - Phase 2 (epochs freeze_epoch→end): frontend unfrozen at 10× lower LR
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


# ─────────────────────────────────────────────
# VGG-16 Frontend
# ─────────────────────────────────────────────

# VGG-16 conv layers used as CSRNet frontend.
# Original CSRNet paper uses the first 10 conv layers (conv1_1 through conv4_3).
# This gives output stride of 8 (3 max-pools, pool4 is NOT included).
#
# Layer indices into vgg16.features:
#   0  Conv2d(3,   64, 3, pad=1)   conv1_1
#   1  ReLU
#   2  Conv2d(64,  64, 3, pad=1)   conv1_2
#   3  ReLU
#   4  MaxPool2d(2,2)              pool1  stride x2
#   5  Conv2d(64, 128, 3, pad=1)   conv2_1
#   6  ReLU
#   7  Conv2d(128,128, 3, pad=1)   conv2_2
#   8  ReLU
#   9  MaxPool2d(2,2)              pool2  stride x4
#  10  Conv2d(128,256, 3, pad=1)   conv3_1
#  11  ReLU
#  12  Conv2d(256,256, 3, pad=1)   conv3_2
#  13  ReLU
#  14  Conv2d(256,256, 3, pad=1)   conv3_3
#  15  ReLU
#  16  MaxPool2d(2,2)              pool3  stride x8
#  17  Conv2d(256,512, 3, pad=1)   conv4_1
#  18  ReLU
#  19  Conv2d(512,512, 3, pad=1)   conv4_2
#  20  ReLU
#  21  Conv2d(512,512, 3, pad=1)   conv4_3
#  22  ReLU
#
# We take layers 0-22 (inclusive), giving 512-channel output at 1/8 resolution.

FRONTEND_END_IDX = 23   # features[:23] = layers 0-22 inclusive


# ─────────────────────────────────────────────
# Dilated Backend
# ─────────────────────────────────────────────

def make_backend() -> nn.Sequential:
    """
    Build CSRNet backend: 6 dilated conv layers + 1x1 output conv.

    Input:  [B, 512, H/8, W/8]
    Output: [B,   1, H/8, W/8]  (density map at 1/8 resolution)
    """
    return nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 128, kernel_size=3, padding=2, dilation=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(128,  64, kernel_size=3, padding=2, dilation=2),
        nn.ReLU(inplace=True),
        nn.Conv2d( 64,   1, kernel_size=1),
    )


# ─────────────────────────────────────────────
# CSRNet
# ─────────────────────────────────────────────

class CSRNet(nn.Module):
    """
    CSRNet for dense livestock counting via density map regression.

    Forward pass:
        image  [B, 3, H, W]  →  density_map  [B, 1, H/8, W/8]

    The predicted density map D satisfies:
        estimated_count = D.sum(dim=[1,2,3])   (sum over spatial dims)

    Usage:
        model = CSRNet()
        model.load_vgg16_weights('models/pretrained/vgg16-397923af.pth')
        # Phase 1: freeze frontend
        model.freeze_frontend()
        # ... train backend only ...
        # Phase 2: unfreeze for full fine-tuning
        model.unfreeze_frontend()
    """

    def __init__(self):
        super().__init__()

        # ── Frontend: VGG-16 layers 0-22 (conv1_1–conv4_3) ──
        # weights=None: we load pretrained weights separately via load_vgg16_weights()
        try:
            vgg16 = models.vgg16(weights=None)    # PyTorch >= 0.13
        except TypeError:
            vgg16 = models.vgg16(pretrained=False) # PyTorch 1.12 compat
        self.frontend = nn.Sequential(
            *list(vgg16.features.children())[:FRONTEND_END_IDX]
        )

        # ── Backend: dilated CNN ───────────────────────
        self.backend = make_backend()

        # ── Weight initialization for backend ─────────
        self._init_backend_weights()

    # ─────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]  normalized input image (ImageNet stats)

        Returns:
            density: [B, 1, H/8, W/8]  predicted density map
                     sum over spatial dims = estimated count
        """
        x = self.frontend(x)    # [B, 512, H/8, W/8]
        x = self.backend(x)     # [B,   1, H/8, W/8]
        return x

    # ─────────────────────────────────────────────
    # Weight loading
    # ─────────────────────────────────────────────

    def load_vgg16_weights(self, weights_path: str):
        """
        Load ImageNet-pretrained VGG-16 weights into the frontend.

        Only the convolutional layer weights (features.0 → features.16)
        are loaded. VGG-16 FC layer weights are discarded (not used).

        Args:
            weights_path: path to vgg16-397923af.pth
                          (download: https://download.pytorch.org/models/vgg16-397923af.pth)
        """
        if not torch.cuda.is_available():
            state = torch.load(weights_path, map_location='cpu')
        else:
            state = torch.load(weights_path)

        # VGG-16 state dict uses keys like "features.0.weight", "features.0.bias", ...
        # We need to remap to our self.frontend.0.weight, etc.
        frontend_state = {}
        for k, v in state.items():
            if k.startswith('features.'):
                idx_str = k.split('.')[1]   # e.g. "0", "2", ...
                idx = int(idx_str)
                if idx < FRONTEND_END_IDX:
                    # Remap: "features.N.weight" → "N.weight"
                    new_key = k[len('features.'):]
                    frontend_state[new_key] = v

        missing, unexpected = self.frontend.load_state_dict(
            frontend_state, strict=False
        )
        loaded = len(frontend_state) - len(missing)
        print(f"[CSRNet] VGG-16 frontend: loaded {loaded} tensors "
              f"| missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"  Missing keys: {missing[:5]}")

    def load_csrnet_weights(self, weights_path: str):
        """
        Load a full CSRNet checkpoint (frontend + backend).
        Handles multiple formats from different repos/sources.
        """
        if not torch.cuda.is_available():
            ckpt = torch.load(weights_path, map_location='cpu')
        else:
            ckpt = torch.load(weights_path)

        # Handle different checkpoint formats
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state = ckpt['state_dict']
        elif isinstance(ckpt, dict) and 'model_state' in ckpt:
            state = ckpt['model_state']
        elif isinstance(ckpt, dict) and 'model' in ckpt:
            state = ckpt['model']
        elif isinstance(ckpt, dict):
            # Check if keys look like model params (have .weight / .bias)
            sample_keys = list(ckpt.keys())[:3]
            if any('.weight' in k or '.bias' in k for k in sample_keys):
                state = ckpt
            else:
                # Unknown dict format, try as-is
                state = ckpt
        else:
            # Might be a full model object (legacy torch.save(model))
            try:
                state = ckpt.state_dict()
            except AttributeError:
                state = ckpt

        # Strip 'module.' prefix if saved with DataParallel
        state = {k.replace('module.', ''): v for k, v in state.items()}

        # Try direct load first
        missing, unexpected = self.load_state_dict(state, strict=False)

        # If too many missing keys, try remapping common key patterns
        my_keys = set(self.state_dict().keys())
        if len(missing) > len(my_keys) * 0.5:
            # Try remapping: some repos use different naming
            remapped = {}
            for k, v in state.items():
                new_k = k
                # "features.N.xxx" → "frontend.N.xxx"
                if k.startswith('features.'):
                    new_k = k.replace('features.', 'frontend.', 1)
                remapped[new_k] = v

            missing2, unexpected2 = self.load_state_dict(remapped, strict=False)
            if len(missing2) < len(missing):
                missing, unexpected = missing2, unexpected2
                print(f"[CSRNet] Remapped keys (features→frontend)")

        loaded = len(my_keys) - len(missing)
        print(f"[CSRNet] Full checkpoint: loaded {loaded}/{len(my_keys)} params "
              f"| missing={len(missing)} unexpected={len(unexpected)}")
        if missing and len(missing) <= 10:
            print(f"  Missing: {missing}")

    # ─────────────────────────────────────────────
    # Transfer learning: freeze / unfreeze frontend
    # ─────────────────────────────────────────────

    def freeze_frontend(self):
        """
        Freeze all VGG-16 frontend parameters.

        Called at the start of training (Phase 1).
        Experiment note: "冻结部分前端特征提取层" — freeze low-level feature extractor
        so that early training only updates the density regression backend.
        This prevents overwriting the pretrained crowd-counting feature maps
        before the backend has learned to work with livestock features.
        """
        for param in self.frontend.parameters():
            param.requires_grad = False
        print("[CSRNet] Frontend FROZEN — only backend will be trained.")

    def unfreeze_frontend(self):
        """
        Unfreeze VGG-16 frontend for full fine-tuning (Phase 2).

        Called after Phase 1 convergence. The frontend is trained at a
        lower learning rate (lr_frontend << lr_backend) to gently adapt
        VGG-16 features to livestock texture/shape without catastrophic
        forgetting of the crowd-counting representations.
        """
        for param in self.frontend.parameters():
            param.requires_grad = True
        print("[CSRNet] Frontend UNFROZEN — full fine-tuning active.")

    def frontend_is_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.frontend.parameters())

    # ─────────────────────────────────────────────
    # Parameter groups for optimizer
    # ─────────────────────────────────────────────

    def get_optimizer_param_groups(self,
                                   lr_backend:  float = 1e-5,
                                   lr_frontend: float = 1e-6) -> list:
        """
        Return parameter groups for optimizer with different learning rates.

        In Phase 2 (unfrozen), the frontend uses a 10× lower LR than the
        backend to avoid disrupting pretrained VGG-16 features.

        Usage in train.py:
            param_groups = model.get_optimizer_param_groups(lr_backend, lr_frontend)
            optimizer = torch.optim.Adam(param_groups)
        """
        frontend_params = [p for p in self.frontend.parameters()
                           if p.requires_grad]
        backend_params  = list(self.backend.parameters())

        groups = [{'params': backend_params, 'lr': lr_backend}]
        if frontend_params:
            groups.append({'params': frontend_params, 'lr': lr_frontend})

        return groups

    # ─────────────────────────────────────────────
    # Count prediction helper
    # ─────────────────────────────────────────────

    def predict_count(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run forward pass and return scalar count estimates.

        Args:
            x: [B, 3, H, W]

        Returns:
            counts: [B]  predicted animal counts (one per image)
        """
        density = self.forward(x)                        # [B, 1, H/8, W/8]
        counts  = density.sum(dim=[1, 2, 3])             # [B]
        return counts

    # ─────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────

    def _init_backend_weights(self):
        """
        Initialize backend conv layers with Kaiming normal.
        Biases initialized to 0.
        This gives stable gradients before any pretrained weights are loaded.
        """
        for m in self.backend.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # ─────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────

    def count_parameters(self) -> dict:
        """Return parameter counts for frontend, backend, and total."""
        def n_params(module):
            return sum(p.numel() for p in module.parameters())

        def n_trainable(module):
            return sum(p.numel() for p in module.parameters()
                       if p.requires_grad)

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
        print(" CSRNet Architecture Summary")
        print("=" * 50)
        print(f"  Frontend (VGG-16 conv1_1–conv4_3)")
        print(f"    Parameters : {p['frontend_total']:,}")
        print(f"    Trainable  : {p['frontend_trainable']:,}"
              f"  {'[FROZEN]' if frozen else '[ACTIVE]'}")
        print(f"  Backend  (dilated CNN × 6 + 1×1 output)")
        print(f"    Parameters : {p['backend_total']:,}")
        print(f"    Trainable  : {p['backend_trainable']:,}  [ACTIVE]")
        print(f"  Total      : {p['total']:,}")
        print(f"  Trainable  : {p['total_trainable']:,}")
        print("=" * 50)
        print(f"  Input  : [B, 3, H, W]")
        print(f"  Output : [B, 1, H/8, W/8]  (density map)")
        print(f"  Count  : sum(output, dim=[1,2,3])")
        print("=" * 50)
