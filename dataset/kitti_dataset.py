"""
KITTI Frustum Dataset for multi-modal 3D object detection.

Loads pre-processed frustum data produced by the KITTI data preparation
pipeline (e.g. the ``kitti_util.py`` from the original F-PointNet repository).
Each sample contains:
    - A frustum point cloud (xyz + optional intensity).
    - A 2D image crop aligned with the 2D detection proposal.
    - Per-point binary segmentation labels.
    - 3D bounding-box ground-truth (centre, heading, size).

Expected pickle format
----------------------
The dataset reads a list of Python dicts (or a single dict of lists) stored
in a ``.pkl`` file.  Each entry must have:

    ``point_cloud``  (N, 4)   xyz + intensity in frustum frame
    ``box2d``        (4,)     2D box [x1, y1, x2, y2] in image pixels
    ``box3d_center`` (3,)     3D box centre in camera coordinates
    ``angle_class``  int      heading bin index
    ``angle_residual`` float  heading residual
    ``size_class``   int      size-cluster index
    ``size_residual`` (3,)    per-dimension size residual
    ``seg_label``    (N,)     binary 0/1 per-point segmentation mask
    ``image_path``   str      path to the corresponding full image

    (optional) ``frustum_angle`` float  yaw angle of the frustum itself

Usage
-----
::

    from dataset import FrustumKittiDataset
    from torch.utils.data import DataLoader

    train_set = FrustumKittiDataset(
        pickle_file="kitti_train.pkl",
        img_root="kitti/images",
        num_points=1024,
        img_size=(224, 224),
        augment=True,
        split="train",
    )
    loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=4)
"""

from __future__ import annotations

import os
import pickle
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class FrustumKittiDataset(Dataset):
    """PyTorch Dataset wrapping frustum-format KITTI data.

    Args:
        pickle_file (str | Path): Path to the ``.pkl`` file containing the
            pre-processed frustum samples.
        img_root (str | Path): Root directory under which image files
            (specified by ``image_path`` in each pickle sample) are located.
        num_points (int): Fixed point-cloud size.  Each frustum cloud is
            either randomly sub-sampled (if larger) or repeated (if smaller)
            to this length.
        img_size (Tuple[int, int]): ``(H, W)`` to resize every image crop
            before feeding it to the image encoder.
        augment (bool): If ``True``, apply random data augmentation (per-axis
            jitter and random Y-axis flip of the point cloud; random colour
            jitter of the image crop).
        split (str): ``"train"`` or ``"val"`` — used only for augmentation
            logic (augmentation is disabled for ``"val"``).
        use_xyz_only (bool): If ``True`` only xyz (3 channels) are returned
            for each point; otherwise xyz + intensity (4 channels).
    """

    _JITTER_SIGMA = 0.01
    _JITTER_CLIP = 0.05

    def __init__(
        self,
        pickle_file: str | Path,
        img_root: str | Path = "",
        num_points: int = 1024,
        img_size: Tuple[int, int] = (224, 224),
        augment: bool = True,
        split: str = "train",
        use_xyz_only: bool = False,
    ) -> None:
        super().__init__()

        self.img_root = Path(img_root)
        self.num_points = num_points
        self.img_size = img_size
        self.augment = augment and (split == "train")
        self.use_xyz_only = use_xyz_only

        # ------------------------------------------------------------------ #
        # Load data                                                           #
        # ------------------------------------------------------------------ #
        with open(pickle_file, "rb") as f:
            raw = pickle.load(f)

        # Accept both a list of dicts and a dict of lists.
        if isinstance(raw, list):
            self.samples = raw
        elif isinstance(raw, dict):
            keys = list(raw.keys())
            self.samples = [
                {k: raw[k][i] for k in keys}
                for i in range(len(raw[keys[0]]))
            ]
        else:
            raise ValueError(
                f"Unexpected pickle format: {type(raw)}.  Expected list or dict."
            )

        # ------------------------------------------------------------------ #
        # Image transforms                                                    #
        # ------------------------------------------------------------------ #
        color_jitter = transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
        )
        if self.augment:
            self.img_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.RandomHorizontalFlip(),
                color_jitter,
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

    # ---------------------------------------------------------------------- #
    # Dataset interface                                                       #
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # ------------------------------------------------------------------ #
        # Point cloud                                                         #
        # ------------------------------------------------------------------ #
        pc = np.array(sample["point_cloud"], dtype=np.float32)  # [K, 4]
        pc = self._resample(pc, self.num_points)                 # [N, 4]
        if self.use_xyz_only:
            pc = pc[:, :3]

        seg_label = np.array(sample["seg_label"], dtype=np.int64)
        seg_label = self._resample_labels(
            seg_label,
            np.array(sample["point_cloud"], dtype=np.float32).shape[0],
            self.num_points,
        )

        # ------------------------------------------------------------------ #
        # 3D ground-truth                                                     #
        # ------------------------------------------------------------------ #
        center = np.array(sample["box3d_center"], dtype=np.float32)  # [3]
        angle_class = int(sample["angle_class"])
        angle_residual = float(sample["angle_residual"])
        size_class = int(sample["size_class"])
        size_residual = np.array(sample["size_residual"], dtype=np.float32)  # [3]
        heading_angle = float(sample.get("heading_angle", 0.0))  # radians

        # ------------------------------------------------------------------ #
        # Image crop                                                          #
        # ------------------------------------------------------------------ #
        img_crop = self._load_image_crop(sample)

        # ------------------------------------------------------------------ #
        # Data augmentation                                                   #
        # ------------------------------------------------------------------ #
        if self.augment:
            pc = self._augment_pointcloud(pc)

        return {
            "point_cloud": torch.from_numpy(pc),                        # [N, C]
            "img_crop": img_crop,                                       # [3, H, W]
            "seg_label": torch.from_numpy(seg_label),                   # [N]
            "center": torch.from_numpy(center),                        # [3]
            "angle_class": torch.tensor(angle_class, dtype=torch.long),
            "angle_residual": torch.tensor(angle_residual, dtype=torch.float32),
            "size_class": torch.tensor(size_class, dtype=torch.long),
            "size_residual": torch.from_numpy(size_residual),           # [3]
            "heading_angle": torch.tensor(heading_angle, dtype=torch.float32),
        }

    # ---------------------------------------------------------------------- #
    # Helpers                                                                 #
    # ---------------------------------------------------------------------- #

    def _resample(self, pc: np.ndarray, n: int) -> np.ndarray:
        """Sub-sample or repeat rows to obtain exactly *n* points."""
        k = pc.shape[0]
        if k == n:
            return pc
        if k > n:
            idx = np.random.choice(k, n, replace=False)
        else:
            idx = np.concatenate([
                np.arange(k),
                np.random.choice(k, n - k, replace=True),
            ])
        return pc[idx]

    def _resample_labels(
        self, labels: np.ndarray, original_k: int, n: int
    ) -> np.ndarray:
        """Apply the same resampling as :meth:`_resample` to segmentation labels."""
        k = original_k
        if k == n:
            return labels
        if k > n:
            idx = np.random.choice(k, n, replace=False)
        else:
            idx = np.concatenate([
                np.arange(k),
                np.random.choice(k, n - k, replace=True),
            ])
        return labels[idx]

    def _load_image_crop(self, sample: dict) -> torch.Tensor:
        """Load the 2D image crop and apply the configured transforms.

        Falls back to a zero tensor if the image file is not accessible
        (e.g. when the dataset is used without the raw images).
        """
        try:
            img_path = self.img_root / sample["image_path"]
            img = Image.open(img_path).convert("RGB")
            # Optionally crop to box2d
            if "box2d" in sample:
                x1, y1, x2, y2 = [int(v) for v in sample["box2d"]]
                img = img.crop((x1, y1, x2, y2))
        except Exception:
            # Return a zeroed tensor of the expected shape
            h, w = self.img_size
            return torch.zeros(3, h, w)

        return self.img_transform(img)

    def _augment_pointcloud(self, pc: np.ndarray) -> np.ndarray:
        """Apply random jitter and random flip to the point cloud.

        Args:
            pc: ``[N, C]`` point cloud array.

        Returns:
            Augmented ``[N, C]`` array.
        """
        # Random Gaussian jitter on xyz
        jitter = np.clip(
            self._JITTER_SIGMA * np.random.randn(*pc[:, :3].shape),
            -self._JITTER_CLIP,
            self._JITTER_CLIP,
        ).astype(np.float32)
        pc = pc.copy()
        pc[:, :3] += jitter

        # Random Y-axis flip (mirror along Z=0 plane in frustum frame)
        if random.random() > 0.5:
            pc[:, 2] = -pc[:, 2]

        return pc
