"""
Evaluation script for Muti-FPointNet.

Computes:
    - Per-point instance segmentation accuracy
    - Mean 3D IoU (volumetric)
    - Mean BEV (bird's-eye-view) 2D IoU
    - 3D detection AP at IoU thresholds 0.25 and 0.50

Usage
-----
    python test.py \\
        --val_pkl   data/val_frustum.pkl   \\
        --img_root  data/kitti/images      \\
        --checkpoint checkpoints/best_model.pth
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FrustumKittiDataset
from models import MutiFrustumPointNet
from models.fpointnet import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, MEAN_SIZES
from utils.box_util import compute_box3d_iou


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #

def compute_ap(iou_list: list[float], iou_threshold: float) -> float:
    """Compute detection AP (fraction of samples with IoU ≥ threshold)."""
    if not iou_list:
        return 0.0
    return sum(v >= iou_threshold for v in iou_list) / len(iou_list)


# --------------------------------------------------------------------------- #
# Main evaluation loop                                                        #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")

    # ------------------------------------------------------------------ #
    # Dataset                                                             #
    # ------------------------------------------------------------------ #
    img_size = tuple(args.img_size)
    val_set = FrustumKittiDataset(
        pickle_file=args.val_pkl,
        img_root=args.img_root,
        num_points=args.num_points,
        img_size=img_size,
        augment=False,
        split="val",
        use_xyz_only=args.use_xyz_only,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    print(f"Val samples: {len(val_set):,}")

    # ------------------------------------------------------------------ #
    # Model                                                               #
    # ------------------------------------------------------------------ #
    input_channels = 3 if args.use_xyz_only else 4
    model = MutiFrustumPointNet(
        num_points=args.num_points,
        num_object_points=args.num_object_points,
        input_channels=input_channels,
        img_feat_dim=args.img_feat_dim,
        pretrained_backbone=False,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ------------------------------------------------------------------ #
    # Evaluation                                                          #
    # ------------------------------------------------------------------ #
    seg_correct = 0
    seg_total = 0
    iou3d_list: list[float] = []
    iou2d_list: list[float] = []

    mean_sizes = MEAN_SIZES.numpy()

    for batch in tqdm(val_loader, desc="Evaluating"):
        pts = batch["point_cloud"].to(device)
        img_crop = batch["img_crop"].to(device)
        seg_labels = batch["seg_label"].to(device)
        center_gt = batch["center"].numpy()
        heading_gt = batch["heading_angle"].numpy()
        size_cls_gt = batch["size_class"].numpy()
        size_res_gt = batch["size_residual"].numpy()

        pred = model(pts, img_crop)

        # Segmentation accuracy
        pred_labels = pred["seg_logits"].argmax(dim=2)
        seg_correct += (pred_labels == seg_labels).sum().item()
        seg_total += seg_labels.numel()

        # 3D IoU
        size_gt = mean_sizes[size_cls_gt] + size_res_gt  # [B, 3]

        batch_iou3d, batch_iou2d = compute_box3d_iou(
            pred["center"].cpu().numpy(),
            pred["heading_scores"].cpu().numpy(),
            pred["heading_residuals"].cpu().numpy(),
            pred["size_scores"].cpu().numpy(),
            pred["size_residuals"].cpu().numpy(),
            center_gt,
            heading_gt,
            size_gt,
        )
        iou3d_list.extend(batch_iou3d)
        iou2d_list.extend(batch_iou2d)

    # ------------------------------------------------------------------ #
    # Report                                                              #
    # ------------------------------------------------------------------ #
    results = {
        "seg_acc": seg_correct / max(seg_total, 1),
        "mean_iou3d": float(np.mean(iou3d_list)) if iou3d_list else 0.0,
        "mean_iou2d": float(np.mean(iou2d_list)) if iou2d_list else 0.0,
        "ap_3d_025": compute_ap(iou3d_list, 0.25),
        "ap_3d_050": compute_ap(iou3d_list, 0.50),
        "ap_bev_025": compute_ap(iou2d_list, 0.25),
        "ap_bev_050": compute_ap(iou2d_list, 0.50),
    }

    print("\n=== Evaluation Results ===")
    print(f"  Seg Accuracy : {results['seg_acc']:.4f}")
    print(f"  Mean IoU 3D  : {results['mean_iou3d']:.4f}")
    print(f"  Mean IoU BEV : {results['mean_iou2d']:.4f}")
    print(f"  AP 3D  @0.25 : {results['ap_3d_025']:.4f}")
    print(f"  AP 3D  @0.50 : {results['ap_3d_050']:.4f}")
    print(f"  AP BEV @0.25 : {results['ap_bev_025']:.4f}")
    print(f"  AP BEV @0.50 : {results['ap_bev_050']:.4f}")

    return results


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Muti-FPointNet (Multi-modal Frustum PointNet)")
    p.add_argument("--val_pkl", required=True)
    p.add_argument("--img_root", default="")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_points", type=int, default=1024)
    p.add_argument("--num_object_points", type=int, default=512)
    p.add_argument("--img_feat_dim", type=int, default=256)
    p.add_argument("--img_size", type=int, nargs=2, default=[224, 224])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--use_xyz_only", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
