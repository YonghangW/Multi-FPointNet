"""
Image Feature Extractor using ResNet-50 backbone and Transformer Encoder.

Extracts image features from 2D bounding box cropped images for multi-modal
3D object detection. The ResNet-50 feature maps are reshaped into a sequence
of spatial tokens and processed by a Transformer Encoder before global pooling
to produce a compact image feature vector.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ImageFeatureExtractor(nn.Module):
    """
    Image feature extractor combining ResNet-50 and Transformer Encoder.

    Architecture:
        1. ResNet-50 backbone (pretrained) — extracts spatial feature maps.
        2. 1×1 convolution — projects channel dimension to d_model for the
           Transformer without increasing parameter count.
        3. Transformer Encoder — captures long-range spatial dependencies
           across the feature-map tokens.
        4. Global average pooling — aggregates spatial tokens into a single
           feature vector.
        5. Linear projection — maps to the desired ``output_dim``.

    Args:
        output_dim (int): Dimensionality of the output feature vector.
        d_model (int): Internal dimension used by the Transformer Encoder.
        nhead (int): Number of attention heads in each Transformer layer.
        num_encoder_layers (int): Number of stacked Transformer Encoder layers.
        dropout (float): Dropout probability applied in the Transformer.
        pretrained (bool): Whether to initialise the ResNet-50 with
            ImageNet-pretrained weights.
    """

    # ResNet-50 outputs 2048 channels at its final convolutional stage.
    _RESNET50_OUT_CHANNELS = 2048

    def __init__(
        self,
        output_dim: int = 256,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 2,
        dropout: float = 0.1,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------ #
        # ResNet-50 backbone — remove the average-pool and fully-connected    #
        # layers so that we get spatial feature maps [B, 2048, H', W'].       #
        # ------------------------------------------------------------------ #
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        # Keep everything up to (and including) layer4; drop avgpool and fc.
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # ------------------------------------------------------------------ #
        # Channel projection: 2048 → d_model (cheaper than embedding the     #
        # full 2048-dim vector in the Transformer).                           #
        # ------------------------------------------------------------------ #
        self.channel_proj = nn.Sequential(
            nn.Conv2d(self._RESNET50_OUT_CHANNELS, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
        )

        # ------------------------------------------------------------------ #
        # Transformer Encoder — processes the sequence of spatial tokens.    #
        # ``batch_first=True`` so shapes are [B, seq_len, d_model].          #
        # ------------------------------------------------------------------ #
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ------------------------------------------------------------------ #
        # Output projection: d_model → output_dim.                           #
        # ------------------------------------------------------------------ #
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Cropped image tensor of shape ``[B, 3, H, W]``.

        Returns:
            Image feature vector of shape ``[B, output_dim]``.
        """
        # 1. ResNet-50 feature extraction → [B, 2048, H', W']
        feat = self.backbone(x)

        # 2. Channel projection → [B, d_model, H', W']
        feat = self.channel_proj(feat)

        # 3. Reshape to sequence: [B, H'*W', d_model]
        B, C, H, W = feat.shape
        tokens = feat.flatten(2).permute(0, 2, 1)  # [B, H'*W', d_model]

        # 4. Transformer Encoder → [B, H'*W', d_model]
        tokens = self.transformer_encoder(tokens)

        # 5. Global average pooling → [B, d_model]
        feat_vec = tokens.mean(dim=1)

        # 6. Linear projection → [B, output_dim]
        out = self.output_proj(feat_vec)
        return out
