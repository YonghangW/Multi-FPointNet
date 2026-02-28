"""
Core F-PointNet (Frustum PointNets) components.

Implements the three sub-networks described in the paper:
  "Frustum PointNets for 3D Object Detection from RGB-D Data"
  (Qi et al., CVPR 2018, https://arxiv.org/abs/1711.08488)

  1. PointNetInstanceSeg — binary segmentation of frustum points.
  2. STNxyz (T-Net)      — lightweight PointNet for center estimation.
  3. PointNetBoxEstimation — 3D bounding-box parameter regression.

The ``PointNetBoxEstimation`` module is intentionally designed to accept an
optional external feature vector (``extra_feat``) so that the multi-modal
model can fuse image features at the global-feature stage before regression.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Hyper-parameters (KITTI / Waymo conventions)                                #
# --------------------------------------------------------------------------- #
NUM_HEADING_BIN = 12   # Heading angle discretised into 12 bins
NUM_SIZE_CLUSTER = 8   # Number of canonical size clusters (multi-class)

# Default mean sizes for KITTI categories (l, w, h) in metres.
# Order: Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc
MEAN_SIZES = torch.tensor([
    [3.88, 1.63, 1.53],   # Car
    [5.06, 1.91, 2.20],   # Van
    [10.71, 2.85, 3.15],  # Truck
    [0.73, 0.67, 1.74],   # Pedestrian
    [0.73, 0.67, 1.74],   # Person_sitting
    [1.76, 0.60, 1.73],   # Cyclist
    [16.17, 2.53, 3.53],  # Tram
    [3.64, 1.53, 1.92],   # Misc
], dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Shared utility                                                               #
# --------------------------------------------------------------------------- #

class SharedMLP(nn.Module):
    """1-D convolution-based shared MLP operating on point features.

    Args:
        channels: Sequence of channel widths, including the input width.
        bn: Whether to apply BatchNorm after each linear step.
        activation: Activation applied after (optional) BN.
    """

    def __init__(self, channels, bn: bool = True, activation=nn.ReLU):
        super().__init__()
        layers = []
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            layers.append(nn.Conv1d(in_ch, out_ch, 1))
            if bn:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(activation(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLP(nn.Module):
    """Fully-connected MLP for global feature processing.

    Args:
        channels: Sequence of widths, including input and output widths.
        bn: Whether to apply BatchNorm after each layer (except the last).
        last_activation: Whether to apply activation after the final layer.
    """

    def __init__(self, channels, bn: bool = True, last_activation: bool = True):
        super().__init__()
        layers = []
        for i, (in_ch, out_ch) in enumerate(zip(channels[:-1], channels[1:])):
            is_last = i == len(channels) - 2
            layers.append(nn.Linear(in_ch, out_ch))
            if bn and not is_last:
                layers.append(nn.BatchNorm1d(out_ch))
            if not is_last or last_activation:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# --------------------------------------------------------------------------- #
# 1. Instance Segmentation Network                                             #
# --------------------------------------------------------------------------- #

class PointNetInstanceSeg(nn.Module):
    """PointNet-based binary instance segmentation inside a frustum.

    Classifies each frustum point as *object* or *background*.

    Architecture (following the original paper):
        - Shared MLP: 64 → 64 → 64 → 128 → 1024  (local features)
        - Global max-pooling → 1024-D global feature
        - Concatenate local (64-D from 2nd conv) + global (1024-D) → 1088-D
        - Shared MLP: 512 → 256 → 128 → 2  (segmentation logits)

    Args:
        input_channels (int): Number of per-point input features (≥3 for xyz).
    """

    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()

        # Local feature extraction
        self.local_mlp = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1), nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, 1),             nn.BatchNorm1d(64), nn.ReLU(inplace=True),
        )
        # Global feature extraction (continues from local_mlp output)
        self.global_mlp = nn.Sequential(
            nn.Conv1d(64, 64, 1),   nn.BatchNorm1d(64),   nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),  nn.BatchNorm1d(128),  nn.ReLU(inplace=True),
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU(inplace=True),
        )

        # Segmentation head — input is local (64) concatenated with global (1024)
        self.seg_mlp = nn.Sequential(
            nn.Conv1d(1088, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, 256, 1),  nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, 1),  nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 2, 1),
        )

    def forward(self, pts: torch.Tensor):
        """
        Args:
            pts: Point cloud of shape ``[B, N, input_channels]``.

        Returns:
            Tuple of:
                - ``seg_logits``: Per-point segmentation logits ``[B, N, 2]``.
                - ``global_feat``: Global point cloud feature ``[B, 1024]``.
        """
        # Transpose to [B, C, N] for Conv1d
        x = pts.transpose(2, 1)
        B, _, N = x.shape

        # Local features: [B, 64, N]
        local_feat = self.local_mlp(x)

        # Global features: [B, 1024, N] → max-pool → [B, 1024]
        global_feat_map = self.global_mlp(local_feat)              # [B, 1024, N]
        global_feat = torch.max(global_feat_map, dim=2)[0]         # [B, 1024]

        # Concatenate local + expanded global → [B, 1088, N]
        global_expanded = global_feat.unsqueeze(2).expand(-1, -1, N)
        seg_in = torch.cat([local_feat, global_expanded], dim=1)   # [B, 1088, N]

        # Segmentation logits: [B, 2, N] → transpose → [B, N, 2]
        seg_logits = self.seg_mlp(seg_in).transpose(2, 1)

        return seg_logits, global_feat


# --------------------------------------------------------------------------- #
# 2. T-Net (Center Estimation)                                                #
# --------------------------------------------------------------------------- #

class STNxyz(nn.Module):
    """Lightweight PointNet (T-Net) for 3D translation estimation.

    Given the masked (object) points in the frustum, this network regresses
    a 3D translation vector that approximately centres the object at the
    origin, facilitating subsequent box estimation.

    Architecture:
        - Shared MLP: 128 → 128 → 256  (per-point features)
        - Global max-pooling → 256-D
        - FC: 256 → 128 → 3  (translation offset)

    Args:
        input_channels (int): Per-point input dimension (default 3 for xyz).
    """

    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()

        self.point_mlp = nn.Sequential(
            nn.Conv1d(input_channels, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 1),            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pts: Masked object points ``[B, M, 3]``.

        Returns:
            Estimated center translation ``[B, 3]``.
        """
        x = pts.transpose(2, 1)                     # [B, 3, M]
        feat = self.point_mlp(x)                    # [B, 256, M]
        global_feat = torch.max(feat, dim=2)[0]     # [B, 256]
        return self.fc(global_feat)                 # [B, 3]


# --------------------------------------------------------------------------- #
# 3. Box Estimation Network                                                    #
# --------------------------------------------------------------------------- #

class PointNetBoxEstimation(nn.Module):
    """PointNet-based 3D bounding-box parameter estimation.

    Regresses the 3D bounding-box parameters from the masked, centre-
    subtracted object points.  Optionally fuses an external feature
    vector (e.g. image features from :class:`ImageFeatureExtractor`)
    at the global-feature stage before regression.

    Output parameters per sample:
        - ``center_residual``: Residual offset from T-Net centre (3-D).
        - ``heading_scores``: Heading-bin classification logits
          (``NUM_HEADING_BIN``-D).
        - ``heading_residuals``: Per-bin angle residuals
          (``NUM_HEADING_BIN``-D).
        - ``size_scores``: Size-cluster classification logits
          (``NUM_SIZE_CLUSTER``-D).
        - ``size_residuals``: Per-cluster size residuals
          (``NUM_SIZE_CLUSTER``×3-D).

    Args:
        input_channels (int): Per-point input dimension.
        num_heading_bin (int): Number of heading angle bins.
        num_size_cluster (int): Number of size clusters.
        extra_feat_dim (int): Dimensionality of the optional external
            feature vector.  When 0 no fusion is performed.
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_heading_bin: int = NUM_HEADING_BIN,
        num_size_cluster: int = NUM_SIZE_CLUSTER,
        extra_feat_dim: int = 0,
    ) -> None:
        super().__init__()

        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.extra_feat_dim = extra_feat_dim

        # Per-point feature extraction
        self.point_mlp = nn.Sequential(
            nn.Conv1d(input_channels, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 1),            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),            nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Conv1d(256, 512, 1),            nn.BatchNorm1d(512), nn.ReLU(inplace=True),
        )

        # Global feature dimension after optional fusion
        global_dim = 512 + extra_feat_dim

        # Regression head
        output_dim = (
            3                          # center residual
            + num_heading_bin          # heading bin scores
            + num_heading_bin          # heading residuals
            + num_size_cluster         # size cluster scores
            + num_size_cluster * 3     # size residuals
        )
        self.regression_head = nn.Sequential(
            nn.Linear(global_dim, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, 256),        nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, output_dim),
        )

    def forward(
        self,
        pts: torch.Tensor,
        extra_feat: torch.Tensor | None = None,
    ):
        """
        Args:
            pts: Centred masked object points ``[B, M, input_channels]``.
            extra_feat: Optional external feature vector ``[B, extra_feat_dim]``.
                When provided, it is concatenated with the global point-cloud
                feature before regression — enabling image-feature fusion.

        Returns:
            Tuple of:
                - ``center_residual`` ``[B, 3]``
                - ``heading_scores``  ``[B, num_heading_bin]``
                - ``heading_residuals`` ``[B, num_heading_bin]``
                - ``size_scores``     ``[B, num_size_cluster]``
                - ``size_residuals``  ``[B, num_size_cluster, 3]``
        """
        x = pts.transpose(2, 1)                     # [B, C, M]
        feat = self.point_mlp(x)                    # [B, 512, M]
        global_feat = torch.max(feat, dim=2)[0]     # [B, 512]

        # Fuse with image (or other external) features
        if extra_feat is not None:
            global_feat = torch.cat([global_feat, extra_feat], dim=1)  # [B, 512+E]

        # Regression
        out = self.regression_head(global_feat)     # [B, output_dim]

        # --- Parse outputs ---
        idx = 0
        center_residual = out[:, idx:idx + 3]
        idx += 3

        heading_scores = out[:, idx:idx + self.num_heading_bin]
        idx += self.num_heading_bin

        heading_residuals = out[:, idx:idx + self.num_heading_bin]
        idx += self.num_heading_bin

        size_scores = out[:, idx:idx + self.num_size_cluster]
        idx += self.num_size_cluster

        size_residuals = out[:, idx:].view(
            -1, self.num_size_cluster, 3
        )  # [B, NS, 3]

        return (
            center_residual,
            heading_scores,
            heading_residuals,
            size_scores,
            size_residuals,
        )
