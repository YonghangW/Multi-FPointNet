"""
3D Bounding-box utility functions for Frustum PointNet.

Covers:
    - Angle discretisation helpers (``angle2class``, ``class2angle``).
    - Size-cluster helpers (``size2class``, ``class2size``).
    - 3D box corner computation (``get_3d_box``).
    - Volumetric IoU between two sets of 3D boxes (``box3d_iou``,
      ``compute_box3d_iou``).
"""

from __future__ import annotations

import numpy as np
from typing import Sequence

from models.fpointnet import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, MEAN_SIZES


# --------------------------------------------------------------------------- #
# Angle helpers                                                               #
# --------------------------------------------------------------------------- #

def angle2class(angle: float, num_heading_bin: int = NUM_HEADING_BIN):
    """Discretise a continuous angle (radians, in [0, 2π)) into a bin index
    and the residual offset within that bin.

    Args:
        angle: Heading angle in radians, normalised to ``[0, 2π)``.
        num_heading_bin: Number of uniform bins.

    Returns:
        Tuple ``(bin_idx, residual)`` where ``bin_idx`` is an integer in
        ``[0, num_heading_bin)`` and ``residual`` is a float in
        ``(-π/num_heading_bin, π/num_heading_bin]``.
    """
    angle = angle % (2 * np.pi)
    bin_size = 2 * np.pi / num_heading_bin
    bin_idx = int(angle / bin_size)
    residual = angle - (bin_idx * bin_size + bin_size / 2)
    return bin_idx, residual


def class2angle(
    bin_idx: int,
    residual: float,
    num_heading_bin: int = NUM_HEADING_BIN,
    to_label_format: bool = True,
) -> float:
    """Reconstruct a continuous angle from its bin index and residual.

    Args:
        bin_idx: Heading bin index.
        residual: Residual within the bin.
        num_heading_bin: Number of uniform bins.
        to_label_format: If ``True`` the angle is returned in ``[0, 2π)``;
            otherwise it may be negative.

    Returns:
        Heading angle in radians.
    """
    bin_size = 2 * np.pi / num_heading_bin
    angle = bin_idx * bin_size + bin_size / 2 + residual
    if to_label_format and angle > np.pi:
        angle -= 2 * np.pi
    return float(angle)


# --------------------------------------------------------------------------- #
# Size-cluster helpers                                                        #
# --------------------------------------------------------------------------- #

def size2class(size: Sequence[float], num_size_cluster: int = NUM_SIZE_CLUSTER):
    """Find the closest size cluster and return its index and the residual.

    Args:
        size: Object size ``(l, w, h)`` in metres.
        num_size_cluster: Number of canonical size clusters.

    Returns:
        Tuple ``(cluster_idx, residual)`` where ``residual`` is a 3-element
        numpy array ``(dl, dw, dh)``.
    """
    size = np.array(size, dtype=np.float32)
    mean = MEAN_SIZES.numpy()[:num_size_cluster]          # [NS, 3]
    dists = np.sum((mean - size) ** 2, axis=1)            # [NS]
    cluster_idx = int(np.argmin(dists))
    residual = size - mean[cluster_idx]
    return cluster_idx, residual


def class2size(
    cluster_idx: int,
    residual: Sequence[float],
    num_size_cluster: int = NUM_SIZE_CLUSTER,
) -> np.ndarray:
    """Reconstruct a 3D size from its cluster index and residual.

    Args:
        cluster_idx: Size-cluster index.
        residual: Per-dimension residuals ``(dl, dw, dh)``.
        num_size_cluster: Number of canonical size clusters.

    Returns:
        Reconstructed size ``[l, w, h]`` as a numpy array.
    """
    mean = MEAN_SIZES.numpy()[:num_size_cluster]
    return mean[cluster_idx] + np.array(residual, dtype=np.float32)


# --------------------------------------------------------------------------- #
# 3-D box corner computation                                                  #
# --------------------------------------------------------------------------- #

def get_3d_box(
    box_size: Sequence[float],
    heading_angle: float,
    center: Sequence[float],
) -> np.ndarray:
    """Compute the 8 corners of a 3D axis-aligned bounding box after rotation.

    The box is axis-aligned in the object frame, then rotated around the
    Y-axis (heading), then translated to ``center``.

    Args:
        box_size: ``(l, w, h)`` in metres.
        heading_angle: Rotation around the Y-axis in radians.
        center: ``(x, y, z)`` of the box centre in metres.

    Returns:
        Numpy array of shape ``[8, 3]`` — the 8 corner coordinates.
    """
    l, w, h = box_size
    c, s = np.cos(heading_angle), np.sin(heading_angle)
    # Rotation matrix around Y-axis
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)

    # Unit box corners in object frame (before rotation/translation)
    x_c = np.array([ l / 2,  l / 2, -l / 2, -l / 2,
                     l / 2,  l / 2, -l / 2, -l / 2])
    y_c = np.array([ h / 2,  h / 2,  h / 2,  h / 2,
                    -h / 2, -h / 2, -h / 2, -h / 2])
    z_c = np.array([ w / 2, -w / 2, -w / 2,  w / 2,
                     w / 2, -w / 2, -w / 2,  w / 2])

    corners = np.vstack([x_c, y_c, z_c]).T       # [8, 3]
    corners = (R @ corners.T).T                   # rotate
    corners += np.array(center, dtype=np.float64) # translate
    return corners


# --------------------------------------------------------------------------- #
# 3-D IoU                                                                    #
# --------------------------------------------------------------------------- #

def _polygon_clip(subjectPolygon, clipPolygon):
    """Sutherland-Hodgman polygon clipping (2D)."""
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) >= (b[1] - a[1]) * (p[0] - a[0])

    def intersection(a, b, c, d):
        A1 = b[1] - a[1]; B1 = a[0] - b[0]; C1 = A1 * a[0] + B1 * a[1]
        A2 = d[1] - c[1]; B2 = c[0] - d[0]; C2 = A2 * c[0] + B2 * c[1]
        det = A1 * B2 - A2 * B1
        x = (C1 * B2 - C2 * B1) / det if det != 0 else 0.0
        y = (A1 * C2 - A2 * C1) / det if det != 0 else 0.0
        return x, y

    output = list(subjectPolygon)
    if not output:
        return output
    for i in range(len(clipPolygon)):
        if not output:
            break
        inp = output
        output = []
        e_start = clipPolygon[i - 1]
        e_end = clipPolygon[i]
        for j in range(len(inp)):
            curr = inp[j]
            prev = inp[j - 1]
            if inside(curr, e_start, e_end):
                if not inside(prev, e_start, e_end):
                    output.append(intersection(prev, curr, e_start, e_end))
                output.append(curr)
            elif inside(prev, e_start, e_end):
                output.append(intersection(prev, curr, e_start, e_end))
    return output


def _polygon_area(p):
    """Shoelace formula for signed polygon area (2D)."""
    n = len(p)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += p[i][0] * p[j][1]
        area -= p[j][0] * p[i][1]
    return abs(area) / 2.0


def box3d_iou(corners1: np.ndarray, corners2: np.ndarray):
    """Compute volumetric IoU between two 3D boxes given their 8 corners.

    The algorithm:
        1. Projects both boxes onto the ground plane (XZ) and computes the
           2D intersection polygon area.
        2. Multiplies by the height-axis intersection to get 3D overlap
           volume.

    Args:
        corners1: ``[8, 3]`` corners of box 1 (output of :func:`get_3d_box`).
        corners2: ``[8, 3]`` corners of box 2.

    Returns:
        Tuple ``(iou3d, iou2d)`` — volumetric and bird's-eye-view IoU.
    """
    # Intersection in the height dimension (Y-axis)
    ymax = min(corners1[:, 1].max(), corners2[:, 1].max())
    ymin = max(corners1[:, 1].min(), corners2[:, 1].min())
    h_inter = max(0, ymax - ymin)

    # 2-D (XZ) footprints — use CCW ordering (required by Sutherland-Hodgman).
    # Corner layout from get_3d_box: [0, 3, 2, 1] gives CCW in the XZ plane.
    poly1 = [(corners1[i, 0], corners1[i, 2]) for i in [0, 3, 2, 1]]
    poly2 = [(corners2[i, 0], corners2[i, 2]) for i in [0, 3, 2, 1]]

    inter_poly = _polygon_clip(poly1, poly2)
    area_inter = _polygon_area(inter_poly) if len(inter_poly) >= 3 else 0.0

    vol_inter = area_inter * h_inter

    # Individual volumes
    def _box_vol(c):
        return (
            _polygon_area([(c[i, 0], c[i, 2]) for i in [0, 3, 2, 1]])
            * (c[:, 1].max() - c[:, 1].min())
        )

    vol1 = _box_vol(corners1)
    vol2 = _box_vol(corners2)

    iou3d = vol_inter / (vol1 + vol2 - vol_inter + 1e-8)

    # BEV IoU
    area1 = _polygon_area([(corners1[i, 0], corners1[i, 2]) for i in [0, 3, 2, 1]])
    area2 = _polygon_area([(corners2[i, 0], corners2[i, 2]) for i in [0, 3, 2, 1]])
    iou2d = area_inter / (area1 + area2 - area_inter + 1e-8)

    return float(iou3d), float(iou2d)


def compute_box3d_iou(
    center_pred: np.ndarray,
    heading_scores: np.ndarray,
    heading_residuals: np.ndarray,
    size_scores: np.ndarray,
    size_residuals: np.ndarray,
    center_gt: np.ndarray,
    heading_gt: np.ndarray,
    size_gt: np.ndarray,
    num_heading_bin: int = NUM_HEADING_BIN,
    num_size_cluster: int = NUM_SIZE_CLUSTER,
):
    """Compute per-sample 3D and BEV IoU for a batch of predictions.

    Args:
        center_pred: ``[B, 3]``  predicted centres.
        heading_scores: ``[B, NH]``  heading-bin logits.
        heading_residuals: ``[B, NH]``  heading residuals.
        size_scores: ``[B, NS]``  size-cluster logits.
        size_residuals: ``[B, NS, 3]``  size residuals.
        center_gt: ``[B, 3]``  ground-truth centres.
        heading_gt: ``[B]``  ground-truth heading angles (radians).
        size_gt: ``[B, 3]``  ground-truth sizes ``(l, w, h)``.
        num_heading_bin: Number of heading bins.
        num_size_cluster: Number of size clusters.

    Returns:
        Tuple ``(iou3d_list, iou2d_list)`` — lists of per-sample IoU values.
    """
    batch_size = center_pred.shape[0]
    iou3d_list, iou2d_list = [], []

    mean = MEAN_SIZES.numpy()[:num_size_cluster]

    for b in range(batch_size):
        # Predicted heading
        h_bin = int(np.argmax(heading_scores[b]))
        h_res = heading_residuals[b, h_bin]
        heading_p = class2angle(h_bin, h_res, num_heading_bin)

        # Predicted size
        s_bin = int(np.argmax(size_scores[b]))
        size_p = mean[s_bin] + size_residuals[b, s_bin]

        corners_p = get_3d_box(size_p, heading_p, center_pred[b])

        # Ground-truth box
        heading_g = float(heading_gt[b])
        corners_g = get_3d_box(size_gt[b], heading_g, center_gt[b])

        iou3d, iou2d = box3d_iou(corners_p, corners_g)
        iou3d_list.append(iou3d)
        iou2d_list.append(iou2d)

    return iou3d_list, iou2d_list
