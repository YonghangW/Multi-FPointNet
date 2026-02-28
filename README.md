# Muti-FPointNet

多模态三维目标检测算法，基于 F-PointNet (Frustum PointNets) 框架，将 **ResNet-50 + Transformer Encoder** 提取的图像特征与点云全局特征向量在回归网络中融合，实现更准确的 3D 目标检测。

## Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                      Muti-FPointNet Pipeline                      │
├──────────────────────────────────────────────────────────────────┤
│  Image Crop [B,3,H,W]  ──►  ResNet-50  ──►  Transformer Encoder  │
│                                                │                  │
│                                                ▼                  │
│                                         img_feat [B, D]           │
│                                                │                  │
│  Frustum PC [B,N,C]  ──►  InstanceSeg  ──►  T-Net (center est.)  │
│                                  │                    │           │
│                           seg_logits            centered pts       │
│                                                        │           │
│                                         BoxEstimation(pts + img_feat)│
│                                                        │           │
│                                              3D Box Parameters     │
└──────────────────────────────────────────────────────────────────┘
```

## Architecture

### Image Feature Extractor (`models/image_feature_extractor.py`)
- **ResNet-50** backbone (pretrained on ImageNet) — extracts spatial feature maps from 2D bounding-box crops
- **1×1 Conv** — projects 2048-channel maps to Transformer d_model dimension
- **Transformer Encoder** — multi-head self-attention over spatial tokens to capture global image context
- **Global Average Pooling + Linear** — produces a compact image feature vector

### F-PointNet Components (`models/fpointnet.py`)
- **PointNetInstanceSeg** — PointNet that segments frustum points into object/background
- **STNxyz (T-Net)** — lightweight PointNet estimating the 3D translation to the object centre
- **PointNetBoxEstimation** — PointNet regressing 3D bounding-box parameters, supports optional `extra_feat` for image-feature fusion

### Multi-modal Fusion (`models/muti_fpointnet.py`)
Image features are concatenated with the **global point-cloud feature vector** (after max-pooling) inside `PointNetBoxEstimation`, before the box regression head.  This is the key multi-modal fusion step.

## Project Structure

```
Muti-FPointNet/
├── requirements.txt
├── train.py                        # Training script
├── test.py                         # Evaluation script
├── models/
│   ├── image_feature_extractor.py  # ResNet-50 + Transformer Encoder
│   ├── fpointnet.py                # PointNet components
│   └── muti_fpointnet.py          # Full multi-modal model
├── dataset/
│   └── kitti_dataset.py           # KITTI frustum dataset loader
└── utils/
    └── box_util.py                # 3D box utilities, IoU metrics
```

## Installation

```bash
pip install -r requirements.txt
```

## Data Preparation

Pre-process KITTI data into frustum format (using the standard F-PointNet data
preparation pipeline).  Each pickle file should contain a list of sample dicts
with keys: `point_cloud`, `box2d`, `box3d_center`, `angle_class`,
`angle_residual`, `size_class`, `size_residual`, `seg_label`, `image_path`.

## Training

```bash
python train.py \
    --train_pkl data/train_frustum.pkl \
    --val_pkl   data/val_frustum.pkl   \
    --img_root  data/kitti/images      \
    --output_dir checkpoints/          \
    --epochs 100 --batch_size 32
```

Key arguments:

| Argument | Default | Description |
|---|---|---|
| `--num_points` | 1024 | Points per frustum |
| `--num_object_points` | 512 | Points used for box estimation |
| `--img_feat_dim` | 256 | Image feature vector dimension |
| `--img_size` | 224 224 | Crop resize (H W) |
| `--lr` | 1e-3 | Initial learning rate |
| `--lr_decay_step` | 20 | LR decay period (epochs) |
| `--no_pretrain` | — | Skip pretrained ResNet weights |

## Evaluation

```bash
python test.py \
    --val_pkl       data/val_frustum.pkl   \
    --img_root      data/kitti/images      \
    --checkpoint    checkpoints/best_model.pth
```

Reports segmentation accuracy, mean IoU 3D/BEV, and AP at IoU thresholds 0.25 and 0.50.

## Loss Functions

The total training loss is a weighted sum of six terms:

| Term | Type | Description |
|---|---|---|
| `L_seg` | Cross-entropy | Per-point instance segmentation |
| `L_center` | Smooth-L1 | 3D centre prediction |
| `L_heading_cls` | Cross-entropy | Heading bin classification |
| `L_heading_res` | Smooth-L1 | Heading residual regression |
| `L_size_cls` | Cross-entropy | Size cluster classification |
| `L_size_res` | Smooth-L1 | Size residual regression |

## References

- [Frustum PointNets for 3D Object Detection from RGB-D Data](https://arxiv.org/abs/1711.08488) (Qi et al., CVPR 2018)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) (He et al., CVPR 2016)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., NeurIPS 2017)
