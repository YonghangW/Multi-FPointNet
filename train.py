"""
Training script for Muti-FPointNet (Multi-modal Frustum PointNet).

Multi-task Loss
---------------
The total loss is the weighted sum of five terms:

    L = w_seg * L_seg
      + w_center * L_center
      + w_heading_cls * L_heading_cls
      + w_heading_res * L_heading_res
      + w_size_cls * L_size_cls
      + w_size_res * L_size_res

where:
    L_seg         — cross-entropy for per-point instance segmentation.
    L_center      — smooth-L1 (Huber) loss for the 3-D centre prediction.
    L_heading_cls — cross-entropy for heading-bin classification.
    L_heading_res — smooth-L1 for the heading residual of the GT bin.
    L_size_cls    — cross-entropy for size-cluster classification.
    L_size_res    — smooth-L1 for the size residual of the GT cluster.

Usage
-----
    python train.py \\
        --train_pkl data/train_frustum.pkl \\
        --val_pkl   data/val_frustum.pkl   \\
        --img_root  data/kitti/images      \\
        --output_dir checkpoints/          \\
        --epochs 100 --batch_size 32
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import FrustumKittiDataset
from models import MutiFrustumPointNet
from models.fpointnet import NUM_HEADING_BIN, NUM_SIZE_CLUSTER
from utils.box_util import compute_box3d_iou


# --------------------------------------------------------------------------- #
# Loss functions                                                               #
# --------------------------------------------------------------------------- #

def get_seg_loss(seg_logits: torch.Tensor, seg_labels: torch.Tensor) -> torch.Tensor:
    """Per-point binary cross-entropy segmentation loss.

    Args:
        seg_logits: ``[B, N, 2]``
        seg_labels: ``[B, N]`` int64 labels in {0, 1}

    Returns:
        Scalar loss.
    """
    B, N, _ = seg_logits.shape
    logits_flat = seg_logits.reshape(B * N, 2)
    labels_flat = seg_labels.reshape(B * N)
    return nn.functional.cross_entropy(logits_flat, labels_flat)


def get_box_loss(
    pred: dict,
    center_gt: torch.Tensor,
    angle_class_gt: torch.Tensor,
    angle_residual_gt: torch.Tensor,
    size_class_gt: torch.Tensor,
    size_residual_gt: torch.Tensor,
    num_heading_bin: int = NUM_HEADING_BIN,
    num_size_cluster: int = NUM_SIZE_CLUSTER,
) -> dict:
    """Multi-task 3D box regression loss.

    Args:
        pred: Model output dictionary (keys: ``center``, ``heading_scores``,
              ``heading_residuals``, ``size_scores``, ``size_residuals``).
        center_gt: ``[B, 3]``
        angle_class_gt: ``[B]``  int64
        angle_residual_gt: ``[B]``  float32
        size_class_gt: ``[B]``  int64
        size_residual_gt: ``[B, 3]``  float32
        num_heading_bin: Number of heading bins.
        num_size_cluster: Number of size clusters.

    Returns:
        Dict with individual loss terms and ``total`` key.
    """
    huber = nn.SmoothL1Loss()

    # Centre loss
    center_loss = huber(pred["center"], center_gt)

    # Heading classification loss
    heading_cls_loss = nn.functional.cross_entropy(
        pred["heading_scores"], angle_class_gt
    )

    # Heading residual loss (only for the GT bin)
    B = pred["heading_residuals"].shape[0]
    gt_bin_idx = angle_class_gt.unsqueeze(1)  # [B, 1]
    heading_res_pred = pred["heading_residuals"].gather(1, gt_bin_idx).squeeze(1)
    heading_res_loss = huber(heading_res_pred, angle_residual_gt)

    # Size classification loss
    size_cls_loss = nn.functional.cross_entropy(
        pred["size_scores"], size_class_gt
    )

    # Size residual loss (only for the GT cluster)
    gt_cluster_idx = size_class_gt.unsqueeze(1).unsqueeze(2).expand(-1, 1, 3)
    size_res_pred = pred["size_residuals"].gather(1, gt_cluster_idx).squeeze(1)
    size_res_loss = huber(size_res_pred, size_residual_gt)

    total = (
        center_loss
        + heading_cls_loss
        + heading_res_loss
        + size_cls_loss
        + size_res_loss
    )

    return {
        "total": total,
        "center": center_loss,
        "heading_cls": heading_cls_loss,
        "heading_res": heading_res_loss,
        "size_cls": size_cls_loss,
        "size_res": size_res_loss,
    }


# --------------------------------------------------------------------------- #
# Training / validation loops                                                  #
# --------------------------------------------------------------------------- #

def train_one_epoch(
    model: MutiFrustumPointNet,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    seg_weight: float = 1.0,
    box_weight: float = 1.0,
) -> dict:
    model.train()
    stats = {k: 0.0 for k in [
        "loss", "seg_loss", "center_loss",
        "heading_cls_loss", "heading_res_loss",
        "size_cls_loss", "size_res_loss",
    ]}
    num_batches = 0

    for batch in loader:
        pts = batch["point_cloud"].to(device)          # [B, N, C]
        img_crop = batch["img_crop"].to(device)        # [B, 3, H, W]
        seg_labels = batch["seg_label"].to(device)     # [B, N]
        center_gt = batch["center"].to(device)         # [B, 3]
        angle_cls_gt = batch["angle_class"].to(device) # [B]
        angle_res_gt = batch["angle_residual"].to(device)  # [B]
        size_cls_gt = batch["size_class"].to(device)   # [B]
        size_res_gt = batch["size_residual"].to(device) # [B, 3]

        optimizer.zero_grad()

        pred = model(pts, img_crop, seg_labels=seg_labels)

        # Segmentation loss
        seg_loss = get_seg_loss(pred["seg_logits"], seg_labels)

        # Box loss
        box_losses = get_box_loss(
            pred,
            center_gt,
            angle_cls_gt,
            angle_res_gt,
            size_cls_gt,
            size_res_gt,
        )

        loss = seg_weight * seg_loss + box_weight * box_losses["total"]
        loss.backward()
        optimizer.step()

        stats["loss"] += loss.item()
        stats["seg_loss"] += seg_loss.item()
        stats["center_loss"] += box_losses["center"].item()
        stats["heading_cls_loss"] += box_losses["heading_cls"].item()
        stats["heading_res_loss"] += box_losses["heading_res"].item()
        stats["size_cls_loss"] += box_losses["size_cls"].item()
        stats["size_res_loss"] += box_losses["size_res"].item()
        num_batches += 1

    return {k: v / max(num_batches, 1) for k, v in stats.items()}


@torch.no_grad()
def validate(
    model: MutiFrustumPointNet,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    seg_correct = 0
    seg_total = 0
    iou3d_sum = 0.0
    iou2d_sum = 0.0
    num_samples = 0

    from models.fpointnet import MEAN_SIZES

    for batch in loader:
        pts = batch["point_cloud"].to(device)
        img_crop = batch["img_crop"].to(device)
        seg_labels = batch["seg_label"].to(device)
        center_gt = batch["center"].numpy()
        heading_gt = batch["heading_angle"].numpy()
        size_cls_gt = batch["size_class"].numpy()
        size_res_gt = batch["size_residual"].numpy()

        pred = model(pts, img_crop)

        # Segmentation accuracy
        pred_labels = pred["seg_logits"].argmax(dim=2)          # [B, N]
        seg_correct += (pred_labels == seg_labels).sum().item()
        seg_total += seg_labels.numel()

        # 3D IoU
        mean_sizes = MEAN_SIZES.numpy()
        size_gt = (
            mean_sizes[size_cls_gt] + size_res_gt
        )  # [B, 3]

        iou3d_list, iou2d_list = compute_box3d_iou(
            pred["center"].cpu().numpy(),
            pred["heading_scores"].cpu().numpy(),
            pred["heading_residuals"].cpu().numpy(),
            pred["size_scores"].cpu().numpy(),
            pred["size_residuals"].cpu().numpy(),
            center_gt,
            heading_gt,
            size_gt,
        )

        iou3d_sum += sum(iou3d_list)
        iou2d_sum += sum(iou2d_list)
        num_samples += len(iou3d_list)

    n = max(num_samples, 1)
    return {
        "seg_acc": seg_correct / max(seg_total, 1),
        "iou3d": iou3d_sum / n,
        "iou2d": iou2d_sum / n,
    }


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Train Muti-FPointNet (Multi-modal Frustum PointNet)")
    p.add_argument("--train_pkl", required=True, help="Path to training pickle")
    p.add_argument("--val_pkl", required=True, help="Path to validation pickle")
    p.add_argument("--img_root", default="", help="Root dir for image files")
    p.add_argument("--output_dir", default="checkpoints", help="Checkpoint directory")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr_decay_step", type=int, default=20,
                   help="Decay learning rate every N epochs")
    p.add_argument("--lr_decay_rate", type=float, default=0.7)
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--num_object_points", type=int, default=512)
    p.add_argument("--img_feat_dim", type=int, default=256)
    p.add_argument("--img_size", type=int, nargs=2, default=[224, 224])
    p.add_argument("--no_pretrain", action="store_true",
                   help="Do not use pretrained ResNet-50 weights")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seg_weight", type=float, default=1.0)
    p.add_argument("--box_weight", type=float, default=1.0)
    p.add_argument("--use_xyz_only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    img_size = tuple(args.img_size)

    # ------------------------------------------------------------------ #
    # Datasets & loaders                                                  #
    # ------------------------------------------------------------------ #
    train_set = FrustumKittiDataset(
        pickle_file=args.train_pkl,
        img_root=args.img_root,
        num_points=args.num_points,
        img_size=img_size,
        augment=True,
        split="train",
        use_xyz_only=args.use_xyz_only,
    )
    val_set = FrustumKittiDataset(
        pickle_file=args.val_pkl,
        img_root=args.img_root,
        num_points=args.num_points,
        img_size=img_size,
        augment=False,
        split="val",
        use_xyz_only=args.use_xyz_only,
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    print(f"Train samples: {len(train_set):,}  |  Val samples: {len(val_set):,}")

    # ------------------------------------------------------------------ #
    # Model                                                               #
    # ------------------------------------------------------------------ #
    input_channels = 3 if args.use_xyz_only else 4
    model = MutiFrustumPointNet(
        num_points=args.num_points,
        num_object_points=args.num_object_points,
        input_channels=input_channels,
        img_feat_dim=args.img_feat_dim,
        pretrained_backbone=not args.no_pretrain,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")

    # ------------------------------------------------------------------ #
    # Optimiser & scheduler                                               #
    # ------------------------------------------------------------------ #
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate
    )

    # ------------------------------------------------------------------ #
    # Training loop                                                       #
    # ------------------------------------------------------------------ #
    best_iou3d = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(
            model, train_loader, optimizer, device,
            seg_weight=args.seg_weight,
            box_weight=args.box_weight,
        )
        scheduler.step()

        val_stats = validate(model, val_loader, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss={train_stats['loss']:.4f}  "
            f"seg={train_stats['seg_loss']:.4f}  "
            f"center={train_stats['center_loss']:.4f}  "
            f"hcls={train_stats['heading_cls_loss']:.4f}  "
            f"hres={train_stats['heading_res_loss']:.4f}  "
            f"scls={train_stats['size_cls_loss']:.4f}  "
            f"sres={train_stats['size_res_loss']:.4f}  "
            f"| val_seg_acc={val_stats['seg_acc']:.4f}  "
            f"iou3d={val_stats['iou3d']:.4f}  "
            f"iou2d={val_stats['iou2d']:.4f}  "
            f"[{elapsed:.1f}s]"
        )

        # Save best checkpoint
        if val_stats["iou3d"] > best_iou3d:
            best_iou3d = val_stats["iou3d"]
            ckpt_path = Path(args.output_dir) / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_stats": val_stats,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  → Saved best model (iou3d={best_iou3d:.4f}) to {ckpt_path}")

        # Save latest checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = Path(args.output_dir) / f"epoch_{epoch:04d}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_stats": val_stats,
                    "args": vars(args),
                },
                ckpt_path,
            )


if __name__ == "__main__":
    main()
