"""
train.py

Transfer learning training script for CSRNet livestock density estimation.

Implements the two-phase transfer strategy from experiment design:
  Phase 1 (epochs 0 → cfg.frontend_unfreeze_epoch):
      VGG-16 frontend FROZEN, only backend trains at lr=1e-5.
      Goal: teach the backend to produce density maps from crowd features
            before adapting the frontend to livestock appearance.

  Phase 2 (epochs cfg.frontend_unfreeze_epoch → num_epochs):
      Frontend UNFROZEN at 10× lower LR (1e-6).
      Goal: gently adapt VGG-16 features to livestock texture/shape
            without catastrophically forgetting crowd-counting representations.

Loss: MSE between predicted density map and GT density map (standard CSRNet).
      Applied only to samples that have point annotations (has_density=True).
      Count-only samples are excluded from density loss but logged for MAE.

Checkpointing: saves best model by val MAE (not val loss) — MAE is the
               primary evaluation metric for the experiment.

Usage:
    python train.py
    python train.py --config configs/config.yaml
    python train.py --resume experiments/logs/checkpoint_epoch_30.pth
"""

import os
import time
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.csrnet import CSRNet
from models.msa_csrnet import MSACSRNet
from datasets.livestock_dataset import build_dataloaders
import csv as csv_mod


# ─────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────

class DensityMSELoss(nn.Module):
    """
    MSE loss between predicted and GT density maps.

    Only computed on samples with point annotations (has_density=True).
    Count-only samples (has_density=False) are excluded — their density
    maps are all-zero placeholders, not real supervision signals.

    The GT density map is at original input resolution [B,1,H,W].
    The prediction is at 1/8 resolution [B,1,H/8,W/8].
    We downsample GT to match prediction before computing loss.
    This is the standard approach in CSRNet implementations.
    """

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self,
                pred:        torch.Tensor,   # [B, 1, H/8, W/8]
                gt_density:  torch.Tensor,   # [B, 1, H,   W  ]
                has_density: torch.Tensor    # [B] bool mask
                ) -> torch.Tensor:

        # Filter to only point-annotated samples
        mask = has_density.bool()
        if mask.sum() == 0:
            # Return zero loss that stays connected to the computation graph.
            # Using pred.sum() * 0 ensures gradients can still flow (no detached tensor).
            return (pred * 0.0).sum()

        pred_masked = pred[mask]           # [N, 1, H/8, W/8]
        gt_masked   = gt_density[mask]     # [N, 1, H,   W  ]

        # Downsample GT density map to match prediction resolution
        # Use avg_pool so the integral (count) is preserved after downsampling
        scale_factor = pred_masked.shape[2] / gt_masked.shape[2]   # e.g. 0.125
        gt_down = nn.functional.interpolate(
            gt_masked,
            size=pred_masked.shape[2:],    # (H/8, W/8)
            mode='bilinear',
            align_corners=False
        )
        # Rescale values: avg_pool reduces sum by factor², restore it
        gt_down = gt_down / (scale_factor ** 2)

        loss = self.mse(pred_masked, gt_down) / mask.sum().float()
        return loss


# ─────────────────────────────────────────────
# Evaluation loop (shared by train + eval scripts)
# ─────────────────────────────────────────────

def evaluate(model: CSRNet,
             loader,
             device: torch.device) -> dict:
    """
    Compute MAE and RE on a data loader.

    MAE = (1/N) * sum(|pred_count - gt_count|)
    RE  = (1/N) * sum(|pred_count - gt_count| / max(gt_count, 1))

    Returns dict with keys: mae, re, predictions (list), ground_truths (list)
    """
    model.eval()
    abs_errors = []
    rel_errors = []
    predictions = []
    ground_truths = []

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)       # [B, 3, H, W]
            counts = batch['count'].to(device)        # [B]

            pred_density = model(images)              # [B, 1, H/8, W/8]
            pred_counts  = pred_density.sum(dim=[1, 2, 3])  # [B]

            for pred, gt in zip(pred_counts.cpu().numpy(),
                                counts.cpu().numpy()):
                ae = abs(pred - gt)
                re = ae / max(gt, 1.0)
                abs_errors.append(ae)
                rel_errors.append(re)
                predictions.append(float(pred))
                ground_truths.append(float(gt))

    return {
        'mae':           np.mean(abs_errors),
        're':            np.mean(rel_errors),
        'predictions':   predictions,
        'ground_truths': ground_truths,
    }


# ─────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


# ─────────────────────────────────────────────
# Scheduler builder
# ─────────────────────────────────────────────

def build_scheduler(optimizer, cfg_train: dict, remaining_epochs: int = None):
    """
    Build LR scheduler from config.
    Supports 'step' (legacy) and 'cosine' (recommended for MSA-CSRNet).
    """
    sched_type = cfg_train.get('lr_scheduler', 'step')
    if remaining_epochs is None:
        remaining_epochs = cfg_train['num_epochs']

    if sched_type == 'cosine':
        lr_min = cfg_train.get('lr_min', 1e-7)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining_epochs, eta_min=lr_min
        )
    else:
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg_train['lr_step_size'],
            gamma=cfg_train['lr_gamma']
        )


# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────

def train(cfg: dict, resume_path: str = None, model_name: str = 'csrnet'):

    # ── Ensure required directories exist ────
    for key in ['log_dir', 'webapp_upload']:
        os.makedirs(cfg['paths'][key], exist_ok=True)
    os.makedirs(os.path.dirname(cfg['csrnet']['pretrained_backbone']), exist_ok=True)

    # ── Setup ────────────────────────────────
    device = torch.device(cfg['hardware']['device']
                          if torch.cuda.is_available()
                          else 'cpu')
    print(f"[Train] Device: {device}")
    print(f"[Train] Model: {model_name}")

    log_dir  = cfg['paths']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    writer   = SummaryWriter(log_dir=log_dir)

    # ── Data ─────────────────────────────────
    train_loader, val_loader, _ = build_dataloaders(cfg)
    print(f"[Train] Batches per epoch: {len(train_loader)}")

    # ── Model ────────────────────────────────
    if model_name == 'msa_csrnet':
        model = MSACSRNet().to(device)
        model_tag = 'msa_csrnet'
    else:
        model = CSRNet().to(device)
        model_tag = 'csrnet'

    # Load pretrained weights — two-tier strategy:
    #   Tier 1 (best): Full CSRNet pretrained on ShanghaiTech crowd counting
    #   Tier 2 (fallback): VGG-16 ImageNet classification weights for frontend only
    # SKIP when resuming — checkpoint already contains trained weights.
    pretrain_cfg  = cfg['csrnet']
    crowd_path    = pretrain_cfg.get('crowd_pretrained', '')
    vgg_path      = pretrain_cfg['pretrained_backbone']
    freeze_epoch  = pretrain_cfg['frontend_unfreeze_epoch']

    if resume_path and os.path.exists(resume_path):
        print(f"[Transfer] Skipping pretrained loading — will restore from checkpoint")
    elif crowd_path and os.path.exists(crowd_path):
        # Tier 1: full CSRNet crowd-counting weights (frontend + backend)
        model.load_csrnet_weights(crowd_path)
        print(f"[Transfer] Loaded FULL CSRNet crowd-counting weights: {crowd_path}")
        print("[Transfer] Both frontend (VGG-16) and backend (dilated CNN) initialized")
        print("[Transfer] This is the transfer-learning path: crowd → livestock")
    elif os.path.exists(vgg_path):
        # Tier 2: VGG-16 ImageNet only (frontend)
        model.load_vgg16_weights(vgg_path)
        print(f"[Transfer] Loaded VGG-16 ImageNet weights (frontend only): {vgg_path}")
        print("[Transfer] Backend starts from random init — weaker transfer.")
        print("[Transfer] For best results, provide a ShanghaiTech CSRNet checkpoint.")
    else:
        print(f"[WARN] No pretrained weights found!")
        print(f"       crowd_pretrained : {crowd_path}")
        print(f"       VGG-16 backbone  : {vgg_path}")
        print("       Training from scratch (not recommended).")
        print("       Download VGG-16: https://download.pytorch.org/models/vgg16-397923af.pth")

    # Phase 1: freeze frontend (skip if resuming — resume block handles it)
    if not (resume_path and os.path.exists(resume_path)):
        model.freeze_frontend()
        model.print_summary()

    # ── Optimizer ─────────────────────────────
    train_cfg = cfg['train']
    lr_b      = train_cfg['learning_rate']
    lr_f      = train_cfg['lr_frontend']

    # Phase 1 optimizer: backend only
    optimizer = torch.optim.Adam(
        model.get_optimizer_param_groups(lr_backend=lr_b, lr_frontend=lr_f),
        weight_decay=train_cfg['weight_decay']
    )

    scheduler = build_scheduler(optimizer, train_cfg, remaining_epochs=train_cfg['num_epochs'])

    criterion = DensityMSELoss()

    # ── Resume ────────────────────────────────
    start_epoch    = 0
    best_val_mae   = float('inf')
    phase2_started = False

    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        start_epoch    = ckpt['epoch'] + 1
        best_val_mae   = ckpt.get('best_val_mae', float('inf'))
        phase2_started = ckpt.get('phase2_started', False)

        # Restore correct phase state BEFORE rebuilding optimizer
        if phase2_started:
            model.unfreeze_frontend()
        else:
            model.freeze_frontend()

        # Rebuild optimizer with matching param groups, then load state
        optimizer = torch.optim.Adam(
            model.get_optimizer_param_groups(lr_backend=lr_b, lr_frontend=lr_f),
            weight_decay=train_cfg['weight_decay']
        )
        optimizer.load_state_dict(ckpt['optimizer_state'])

        scheduler = build_scheduler(optimizer, train_cfg,
                                    remaining_epochs=train_cfg['num_epochs'] - start_epoch)
        if 'scheduler_state' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state'])

        print(f"[Resume] Loaded checkpoint from epoch {ckpt['epoch']}: "
              f"best_val_mae={best_val_mae:.4f}, "
              f"phase={'2' if phase2_started else '1'}")

    num_epochs = train_cfg['num_epochs']
    accum_steps = train_cfg.get('accumulation_steps', 1)
    patience = train_cfg.get('early_stopping_patience', 0)  # 0 = disabled
    epochs_no_improve = 0
    sched_type = train_cfg.get('lr_scheduler', 'step')

    print(f"\n[Train] Starting training: epochs {start_epoch}→{num_epochs-1}")
    print(f"[Train] Phase 2 (unfreeze frontend) at epoch {freeze_epoch}")
    print(f"[Train] Scheduler: {sched_type} | Accumulation steps: {accum_steps}"
          f"{f' | Early stopping patience: {patience}' if patience > 0 else ''}")
    print("=" * 60)

    # ── Training curve CSV ────────────────────
    curve_path = os.path.join(log_dir, f'training_curves_{model_tag}.csv')
    if start_epoch > 0 and os.path.exists(curve_path):
        # Resuming: append to existing CSV
        curve_file = open(curve_path, 'a', newline='')
        curve_writer = csv_mod.writer(curve_file)
    else:
        curve_file = open(curve_path, 'w', newline='')
        curve_writer = csv_mod.writer(curve_file)
        curve_writer.writerow(['epoch', 'train_loss', 'train_mae', 'val_mae', 'val_re', 'lr', 'phase'])

    # ── Epoch loop ────────────────────────────
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        # ── Phase transition ──────────────────
        if epoch == freeze_epoch and not phase2_started:
            print(f"\n[Phase 2] Epoch {epoch}: unfreezing frontend (lr={lr_f})")
            model.unfreeze_frontend()
            # Rebuild optimizer to include frontend params
            optimizer = torch.optim.Adam(
                model.get_optimizer_param_groups(lr_backend=lr_b,
                                                 lr_frontend=lr_f),
                weight_decay=train_cfg['weight_decay']
            )
            scheduler = build_scheduler(optimizer, train_cfg,
                                        remaining_epochs=num_epochs - epoch)
            phase2_started = True

        # ── Training pass ────────────────────
        model.train()
        train_losses = []
        train_maes   = []

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch:03d}/{num_epochs-1} [train]",
                    leave=False)

        optimizer.zero_grad()
        for step_i, batch in enumerate(pbar):
            images      = batch['image'].to(device)       # [B, 3, H, W]
            gt_density  = batch['density'].to(device)     # [B, 1, H, W]
            gt_counts   = batch['count'].to(device)        # [B]
            has_density = batch['has_density'].to(device)  # [B] bool

            pred_density = model(images)                  # [B, 1, H/8, W/8]
            loss = criterion(pred_density, gt_density, has_density)

            # Scale loss for gradient accumulation
            (loss / accum_steps).backward()

            # Step optimizer every accum_steps (or at end of epoch)
            if (step_i + 1) % accum_steps == 0 or (step_i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=10.0
                )
                optimizer.step()
                optimizer.zero_grad()

            # Batch MAE (for progress bar)
            with torch.no_grad():
                pred_counts = pred_density.sum(dim=[1, 2, 3])
                batch_mae   = (pred_counts - gt_counts).abs().mean().item()

            train_losses.append(loss.item())
            train_maes.append(batch_mae)
            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                              'MAE':  f'{batch_mae:.2f}'})

        scheduler.step()

        avg_train_loss = np.mean(train_losses)
        avg_train_mae  = np.mean(train_maes)

        # ── Validation pass ──────────────────
        val_metrics = evaluate(model, val_loader, device)
        val_mae     = val_metrics['mae']
        val_re      = val_metrics['re']

        # ── Logging ──────────────────────────
        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:03d} | "
              f"loss={avg_train_loss:.4f} | "
              f"train_MAE={avg_train_mae:.2f} | "
              f"val_MAE={val_mae:.2f} | "
              f"val_RE={val_re:.4f} | "
              f"lr={current_lr:.2e} | "
              f"phase={'2' if phase2_started else '1'} | "
              f"{elapsed:.1f}s")

        writer.add_scalar('Loss/train',    avg_train_loss, epoch)
        writer.add_scalar('MAE/train',     avg_train_mae,  epoch)
        writer.add_scalar('MAE/val',       val_mae,        epoch)
        writer.add_scalar('RE/val',        val_re,         epoch)
        writer.add_scalar('LR/backend',    current_lr,     epoch)

        # ── CSV curve export ──────────────────
        curve_writer.writerow([epoch, f'{avg_train_loss:.6f}', f'{avg_train_mae:.4f}',
                               f'{val_mae:.4f}', f'{val_re:.6f}', f'{current_lr:.2e}',
                               2 if phase2_started else 1])
        curve_file.flush()

        # ── Checkpointing ────────────────────
        state = {
            'epoch':          epoch,
            'model_name':     model_tag,
            'model_state':    model.state_dict(),
            'optimizer_state':optimizer.state_dict(),
            'scheduler_state':scheduler.state_dict(),
            'val_mae':        val_mae,
            'val_re':         val_re,
            'best_val_mae':   best_val_mae,
            'phase2_started': phase2_started,
            'config':         cfg,
        }

        # Always save latest
        save_checkpoint(state,
                        os.path.join(log_dir, f'checkpoint_{model_tag}_latest.pth'))

        # Save best model by val MAE
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            state['best_val_mae'] = best_val_mae
            save_checkpoint(state,
                            os.path.join(log_dir, f'checkpoint_{model_tag}_best.pth'))
            print(f"  ✓ New best val MAE: {best_val_mae:.4f} — saved.")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Save periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(state,
                            os.path.join(log_dir,
                                         f'checkpoint_{model_tag}_epoch_{epoch:03d}.pth'))

        # Early stopping
        if patience > 0 and epochs_no_improve >= patience:
            print(f"\n[Early Stop] No improvement for {patience} epochs. "
                  f"Best val MAE: {best_val_mae:.4f}")
            break

    curve_file.close()
    writer.close()
    print("=" * 60)
    print(f"[Train] Done. Model: {model_tag}  Best val MAE: {best_val_mae:.4f}")
    print(f"[Train] Best model: {os.path.join(log_dir, f'checkpoint_{model_tag}_best.pth')}")
    print(f"[Train] Curves CSV: {curve_path}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Train CSRNet/MSA-CSRNet for livestock counting.')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint .pth to resume from')
    parser.add_argument('--model', type=str, default='csrnet',
                        choices=['csrnet', 'msa_csrnet'],
                        help='Model architecture (default: csrnet)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    train(cfg, resume_path=args.resume, model_name=args.model)
