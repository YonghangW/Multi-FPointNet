"""
Multi-modal Frustum PointNet (Muti-FPointNet).

Combines image features extracted from 2D bounding-box crops (ResNet-50 +
Transformer Encoder) with the point-cloud global feature vector inside the
3D box-estimation sub-network.  The fusion happens *after* global max-pooling
on the point features and *before* the regression head, so the box estimator
sees both modalities simultaneously.

Architecture overview
---------------------
Input:
    pts         — frustum point cloud ``[B, N, C]``
    img_crop    — 2D image crop aligned with the detection proposal ``[B, 3, H, W]``

Stage 1 — Instance Segmentation:
    PointNetInstanceSeg(pts) → seg_logits [B, N, 2], global_feat [B, 1024]

Stage 2 — Masked Point Sampling:
    Apply predicted mask to select object points → pts_obj [B, M, C]

Stage 3 — Center Estimation (T-Net):
    STNxyz(pts_obj) → center_delta [B, 3]
    pts_centred = pts_obj − center_delta  (translate to object centre)

Stage 4 — Image Feature Extraction:
    ImageFeatureExtractor(img_crop) → img_feat [B, img_feat_dim]

Stage 5 — 3D Box Estimation with Image Fusion:
    PointNetBoxEstimation(pts_centred, extra_feat=img_feat)
    → center_residual, heading_*, size_*

Final centre: center = center_delta + center_residual
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fpointnet import (
    PointNetInstanceSeg,
    STNxyz,
    PointNetBoxEstimation,
    NUM_HEADING_BIN,
    NUM_SIZE_CLUSTER,
    MEAN_SIZES,
)
from .image_feature_extractor import ImageFeatureExtractor


class MutiFrustumPointNet(nn.Module):
    """Multi-modal Frustum PointNet for 3D object detection.

    Args:
        num_points (int): Fixed number of frustum points (N).
        num_object_points (int): Number of object points used for box
            estimation after masking and T-Net centering (M).
        input_channels (int): Per-point input feature dimension (e.g. 3
            for xyz, 4 for xyz+intensity).
        num_heading_bin (int): Number of heading-angle discretisation bins.
        num_size_cluster (int): Number of canonical size clusters.
        img_feat_dim (int): Dimensionality of the image feature vector
            produced by :class:`ImageFeatureExtractor`.
        img_d_model (int): Transformer d_model inside the image encoder.
        img_nhead (int): Attention heads in the image Transformer.
        img_num_encoder_layers (int): Stacked layers in the image Transformer.
        pretrained_backbone (bool): Load ImageNet weights for ResNet-50.
    """

    def __init__(
        self,
        num_points: int = 1024,
        num_object_points: int = 512,
        input_channels: int = 3,
        num_heading_bin: int = NUM_HEADING_BIN,
        num_size_cluster: int = NUM_SIZE_CLUSTER,
        img_feat_dim: int = 256,
        img_d_model: int = 512,
        img_nhead: int = 8,
        img_num_encoder_layers: int = 2,
        pretrained_backbone: bool = True,
    ) -> None:
        super().__init__()

        self.num_points = num_points
        self.num_object_points = num_object_points
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.img_feat_dim = img_feat_dim

        # ------------------------------------------------------------------ #
        # Sub-networks                                                        #
        # ------------------------------------------------------------------ #

        # Stage 1: instance segmentation inside frustum
        self.instance_seg = PointNetInstanceSeg(input_channels=input_channels)

        # Stage 3: center estimation (T-Net)
        self.tnet = STNxyz(input_channels=input_channels)

        # Stage 4: image feature extractor (ResNet-50 + Transformer Encoder)
        self.image_extractor = ImageFeatureExtractor(
            output_dim=img_feat_dim,
            d_model=img_d_model,
            nhead=img_nhead,
            num_encoder_layers=img_num_encoder_layers,
            pretrained=pretrained_backbone,
        )

        # Stage 5: box estimation with image-feature fusion
        self.box_estimator = PointNetBoxEstimation(
            input_channels=input_channels,
            num_heading_bin=num_heading_bin,
            num_size_cluster=num_size_cluster,
            extra_feat_dim=img_feat_dim,   # ← fuse image features here
        )

        # Register mean sizes as a buffer so they move to GPU automatically.
        self.register_buffer("mean_sizes", MEAN_SIZES[:num_size_cluster])

    # ---------------------------------------------------------------------- #
    # Helper: gather object points from segmentation mask                    #
    # ---------------------------------------------------------------------- #

    def _gather_object_points(
        self,
        pts: torch.Tensor,
        seg_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Sample ``num_object_points`` points predicted as *object*.

        If the number of predicted object points is smaller than
        ``num_object_points``, the selection is padded by random re-sampling
        from the available object points.  If no point is classified as
        object, all frustum points are used.

        Args:
            pts: ``[B, N, C]``
            seg_logits: ``[B, N, 2]``

        Returns:
            ``[B, num_object_points, C]``
        """
        B, N, C = pts.shape
        M = self.num_object_points

        # Predicted object mask: 1 where argmax == 1
        mask = seg_logits.argmax(dim=2)  # [B, N]  values in {0, 1}

        gathered = []
        for b in range(B):
            obj_idx = mask[b].nonzero(as_tuple=False).squeeze(1)  # [K]
            K = obj_idx.numel()
            if K == 0:
                # Fall back to all points when no object point is predicted.
                obj_idx = torch.arange(N, device=pts.device)
                K = N

            if K >= M:
                # Random sub-sample without replacement.
                chosen = obj_idx[torch.randperm(K, device=pts.device)[:M]]
            else:
                # Oversample (with replacement) to reach M.
                repeat = (M // K) + 1
                idx_rep = obj_idx.repeat(repeat)[:M]
                chosen = idx_rep

            gathered.append(pts[b, chosen])  # [M, C]

        return torch.stack(gathered, dim=0)  # [B, M, C]

    # ---------------------------------------------------------------------- #
    # Forward pass                                                            #
    # ---------------------------------------------------------------------- #

    def forward(
        self,
        pts: torch.Tensor,
        img_crop: torch.Tensor,
        seg_labels: torch.Tensor | None = None,
    ):
        """
        Args:
            pts: Frustum point cloud ``[B, N, input_channels]``.
            img_crop: 2D bounding-box image crop ``[B, 3, H, W]``.
            seg_labels: Optional ground-truth segmentation labels ``[B, N]``
                (values 0/1).  When provided during training, the *true* mask
                is used to select object points so that box-estimation trains
                on clean inputs even with imperfect segmentation.

        Returns:
            Dictionary with keys:

            ``seg_logits``
                Per-point binary segmentation logits ``[B, N, 2]``.
            ``center``
                Predicted 3D object centre in frustum coordinates ``[B, 3]``.
            ``heading_scores``
                Heading-bin classification logits ``[B, num_heading_bin]``.
            ``heading_residuals``
                Per-bin heading residuals ``[B, num_heading_bin]``.
            ``size_scores``
                Size-cluster classification logits ``[B, num_size_cluster]``.
            ``size_residuals``
                Per-cluster size residuals ``[B, num_size_cluster, 3]``.
            ``img_feat``
                Image feature vector ``[B, img_feat_dim]``.
        """
        # ------------------------------------------------------------------ #
        # Stage 1: Instance segmentation                                      #
        # ------------------------------------------------------------------ #
        seg_logits, _global_feat = self.instance_seg(pts)  # [B,N,2], [B,1024]

        # ------------------------------------------------------------------ #
        # Stage 2: Select object points                                       #
        # ------------------------------------------------------------------ #
        if seg_labels is not None:
            # During training use ground-truth labels for clean box estimation.
            pts_obj = self._gather_object_points_from_labels(pts, seg_labels)
        else:
            pts_obj = self._gather_object_points(pts, seg_logits)
        # pts_obj: [B, M, C]

        # ------------------------------------------------------------------ #
        # Stage 3: T-Net center estimation                                    #
        # ------------------------------------------------------------------ #
        center_delta = self.tnet(pts_obj)               # [B, 3]
        pts_centred = pts_obj - center_delta.unsqueeze(1)  # translate

        # ------------------------------------------------------------------ #
        # Stage 4: Image feature extraction                                   #
        # ------------------------------------------------------------------ #
        img_feat = self.image_extractor(img_crop)       # [B, img_feat_dim]

        # ------------------------------------------------------------------ #
        # Stage 5: 3D box estimation with image-feature fusion               #
        # ------------------------------------------------------------------ #
        (
            center_residual,
            heading_scores,
            heading_residuals,
            size_scores,
            size_residuals,
        ) = self.box_estimator(pts_centred, extra_feat=img_feat)

        # Final predicted centre
        center = center_delta + center_residual         # [B, 3]

        return {
            "seg_logits": seg_logits,
            "center": center,
            "heading_scores": heading_scores,
            "heading_residuals": heading_residuals,
            "size_scores": size_scores,
            "size_residuals": size_residuals,
            "img_feat": img_feat,
        }

    def _gather_object_points_from_labels(
        self,
        pts: torch.Tensor,
        seg_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Like :meth:`_gather_object_points` but uses ground-truth labels.

        Args:
            pts: ``[B, N, C]``
            seg_labels: ``[B, N]``  (0 = background, 1 = object)

        Returns:
            ``[B, num_object_points, C]``
        """
        B, N, C = pts.shape
        M = self.num_object_points
        gathered = []
        for b in range(B):
            obj_idx = (seg_labels[b] == 1).nonzero(as_tuple=False).squeeze(1)
            K = obj_idx.numel()
            if K == 0:
                obj_idx = torch.arange(N, device=pts.device)
                K = N
            if K >= M:
                chosen = obj_idx[torch.randperm(K, device=pts.device)[:M]]
            else:
                repeat = (M // K) + 1
                chosen = obj_idx.repeat(repeat)[:M]
            gathered.append(pts[b, chosen])
        return torch.stack(gathered, dim=0)
