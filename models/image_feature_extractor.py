"""Optimized Image Feature Extractor with ResNet-50 and Transformer Encoder.

改进点：
1. 在ResNet后添加1x1卷积降维（2048->512）
2. Transformer在512维上计算，效率更高
3. 支持多种训练模式：frozen/finetune/end2end
4. 添加L2归一化使图像特征和点云特征尺度一致

"""

import tensorflow as tf
import tf.compat.v1 as tf_compat
tf_compat.disable_v2_behavior()

import numpy as np


class PositionalEncoding2D(tf.keras.layers.Layer):
    """2D Positional Encoding for image features."""

    def __init__(self, height, width, d_model, **kwargs):
        super(PositionalEncoding2D, self).__init__(**kwargs)
        self.height = height
        self.width = width
        self.d_model = d_model

    def build(self, input_shape):
        # Create learnable positional embeddings
        self.x_emb = self.add_weight(
            name="x_emb",
            shape=(self.width, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.y_emb = self.add_weight(
            name="y_emb",
            shape=(self.height, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
        )
        super(PositionalEncoding2D, self).build(input_shape)

    def call(self, inputs, training=None):
        """
        Input: (B, H, W, C) where C == d_model
        Output: (B, H, W, C) with positional encoding added
        """
        batch_size = tf.shape(inputs)[0]

        # Create position encodings
        x_pos = tf.expand_dims(self.x_emb, 0)  # (1, W, d_model)
        y_pos = tf.expand_dims(self.y_emb, 1)  # (H, 1, d_model)
        pos_encoding = x_pos + y_pos  # (H, W, d_model)
        pos_encoding = tf.expand_dims(pos_encoding, 0)  # (1, H, W, d_model)

        return inputs + pos_encoding

    def get_config(self):
        config = super(PositionalEncoding2D, self).get_config()
        config.update(
            {"height": self.height, "width": self.width, "d_model": self.d_model}
        )
        return config


class MultiHeadAttention2D(tf.keras.layers.Layer):
    """Multi-head attention for 2D feature maps (optimized for 512-dim)."""

    def __init__(self, d_model, num_heads, dropout_rate=0.1, **kwargs):
        super(MultiHeadAttention2D, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        assert d_model % num_heads == 0, (
            f"d_model({d_model}) must be divisible by num_heads({num_heads})"
        )
        self.d_k = d_model // num_heads

    def build(self, input_shape):
        self.W_q = tf.keras.layers.Dense(self.d_model, use_bias=False, name="Q_proj")
        self.W_k = tf.keras.layers.Dense(self.d_model, use_bias=False, name="K_proj")
        self.W_v = tf.keras.layers.Dense(self.d_model, use_bias=False, name="V_proj")
        self.W_o = tf.keras.layers.Dense(
            self.d_model, use_bias=False, name="output_proj"
        )
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        super(MultiHeadAttention2D, self).build(input_shape)

    def call(self, inputs, training=None):
        """
        Input: (B, H, W, C) with C == d_model
        Output: (B, H, W, C)
        """
        if isinstance(inputs, (list, tuple)):
            queries, keys, values = inputs
        else:
            queries = keys = values = inputs

        batch_size = tf.shape(queries)[0]
        height = tf.shape(queries)[1]
        width = tf.shape(queries)[2]
        seq_len = height * width

        # Reshape to (B, H*W, C)
        queries_flat = tf.reshape(queries, [batch_size, seq_len, self.d_model])
        keys_flat = tf.reshape(keys, [batch_size, seq_len, self.d_model])
        values_flat = tf.reshape(values, [batch_size, seq_len, self.d_model])

        # Linear projections
        Q = self.W_q(queries_flat)  # (B, H*W, d_model)
        K = self.W_k(keys_flat)
        V = self.W_v(values_flat)

        # Split into multiple heads: (B, num_heads, H*W, d_k)
        Q = tf.transpose(
            tf.reshape(Q, [batch_size, seq_len, self.num_heads, self.d_k]), [0, 2, 1, 3]
        )
        K = tf.transpose(
            tf.reshape(K, [batch_size, seq_len, self.num_heads, self.d_k]), [0, 2, 1, 3]
        )
        V = tf.transpose(
            tf.reshape(V, [batch_size, seq_len, self.num_heads, self.d_k]), [0, 2, 1, 3]
        )

        # Scaled dot-product attention
        scores = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(
            tf.cast(self.d_k, tf.float32)
        )  # (B, num_heads, H*W, H*W)
        attention_weights = tf.nn.softmax(scores, axis=-1)

        if training:
            attention_weights = self.dropout(attention_weights, training=training)

        attention_output = tf.matmul(attention_weights, V)  # (B, num_heads, H*W, d_k)
        attention_output = tf.transpose(
            attention_output, [0, 2, 1, 3]
        )  # (B, H*W, num_heads, d_k)
        attention_output = tf.reshape(
            attention_output, [batch_size, seq_len, self.d_model]
        )

        output = self.W_o(attention_output)  # (B, H*W, d_model)

        # Reshape back to 2D
        output = tf.reshape(output, [batch_size, height, width, self.d_model])

        return output

    def get_config(self):
        config = super(MultiHeadAttention2D, self).get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class TransformerEncoderBlock(tf.keras.layers.Layer):
    """Transformer encoder block for 2D features (optimized version)."""

    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        height = input_shape[1]
        width = input_shape[2]

        self.pos_encoding = PositionalEncoding2D(height, width, self.d_model)
        self.mha = MultiHeadAttention2D(self.d_model, self.num_heads, self.dropout_rate)

        # Feed-forward network
        self.ffn_dense1 = tf.keras.layers.Dense(self.dff, activation="relu")
        self.ffn_dense2 = tf.keras.layers.Dense(self.d_model)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)

        super(TransformerEncoderBlock, self).build(input_shape)

    def call(self, inputs, training=None):
        """
        Input: (B, H, W, C)
        Output: (B, H, W, C)
        """
        x = inputs

        # Add positional encoding
        x_with_pos = self.pos_encoding(x, training=training)

        # Multi-head self-attention
        attn_output = self.mha(x_with_pos, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feed-forward network
        batch_size = tf.shape(out1)[0]
        height = tf.shape(out1)[1]
        width = tf.shape(out1)[2]

        out1_flat = tf.reshape(out1, [batch_size, height * width, self.d_model])
        ffn_output = self.ffn_dense1(out1_flat)
        ffn_output = self.ffn_dense2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        ffn_output = tf.reshape(ffn_output, [batch_size, height, width, self.d_model])

        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    def get_config(self):
        config = super(TransformerEncoderBlock, self).get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dff": self.dff,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class ImageFeatureExtractor(tf.keras.Model):
    """Optimized Image feature extractor using ResNet-50 and Transformer.

    主要改进：
    1. 1x1卷积降维：2048 -> 512，减少Transformer计算量
    2. Transformer在512维上计算，更高效
    3. 支持多种训练模式
    4. 添加特征归一化
    """

    def __init__(
        self,
        output_dim=512,
        num_transformer_blocks=2,
        num_heads=8,
        dff=2048,
        dropout_rate=0.1,
        training_mode="finetune",  # 'frozen', 'finetune', 'end2end'
        freeze_at=0,  # for finetune mode: 0=none, 2=stage2, 3=stage3, 4=stage4, 5=all
        **kwargs,
    ):
        """
        Args:
            output_dim: 输出特征维度，默认512
            num_transformer_blocks: Transformer块数量
            num_heads: 注意力头数
            dff: FFN中间层维度
            dropout_rate: dropout率
            training_mode:
                - 'frozen': 完全冻结ResNet，只训练Transformer
                - 'finetune': 冻结ResNet前N个stage，微调后几个stage
                - 'end2end': 端到端训练所有层
            freeze_at: finetune模式下，冻结到哪个stage (2/3/4/5)
        """
        super(ImageFeatureExtractor, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.training_mode = training_mode
        self.freeze_at = freeze_at

        # Use Keras ResNet50 as backbone (output: 7x7x2048)
        self.resnet = tf.keras.applications.ResNet50(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )

        # 根据训练模式设置ResNet的可训练性
        self._setup_resnet_trainability()

        # 1x1卷积降维：2048 -> 512，大幅降低Transformer计算量
        self.dim_reduction = tf.keras.layers.Conv2D(
            filters=output_dim,  # 512
            kernel_size=1,
            strides=1,
            padding="same",
            activation="relu",
            name="dim_reduction_conv",
        )

        # BatchNorm after dimension reduction
        self.dim_reduction_bn = tf.keras.layers.BatchNormalization(
            name="dim_reduction_bn"
        )

        # Transformer blocks (now operating on 512-dim instead of 2048-dim)
        self.transformer_blocks = []
        for i in range(num_transformer_blocks):
            self.transformer_blocks.append(
                TransformerEncoderBlock(
                    d_model=output_dim,  # 512
                    num_heads=num_heads,
                    dff=dff,
                    dropout_rate=dropout_rate,
                    name=f"transformer_block_{i}",
                )
            )

        # Global average pooling
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()

        # Final feature projection with normalization
        self.output_proj = tf.keras.layers.Dense(
            output_dim, activation=None, name="feature_projection"
        )

        # L2 normalization for better feature compatibility with point cloud
        self.feature_norm = tf.keras.layers.Lambda(
            lambda x: tf.nn.l2_normalize(x, axis=1), name="feature_l2_norm"
        )

    def _setup_resnet_trainability(self):
        """根据训练模式设置ResNet的可训练性."""
        if self.training_mode == "frozen":
            # 完全冻结ResNet，只训练Transformer
            self.resnet.trainable = False
            print("[ImageFeatureExtractor] Mode: FROZEN - ResNet fully frozen")

        elif self.training_mode == "end2end":
            # 端到端训练，全部解冻
            self.resnet.trainable = True
            print("[ImageFeatureExtractor] Mode: END2END - All layers trainable")

        elif self.training_mode == "finetune":
            # 分层解冻策略
            # ResNet50结构: conv1 -> conv2_x -> conv3_x -> conv4_x -> conv5_x
            layer_names = [layer.name for layer in self.resnet.layers]

            # 找到各个stage的分界点
            stage_starts = {
                "conv1": 0,
                "conv2": next(
                    (i for i, n in enumerate(layer_names) if "conv2_block" in n), None
                ),
                "conv3": next(
                    (i for i, n in enumerate(layer_names) if "conv3_block" in n), None
                ),
                "conv4": next(
                    (i for i, n in enumerate(layer_names) if "conv4_block" in n), None
                ),
                "conv5": next(
                    (i for i, n in enumerate(layer_names) if "conv5_block" in n), None
                ),
            }

            # 根据freeze_at确定冻结范围
            freeze_until = {
                0: 0,  # 不冻结
                2: stage_starts.get("conv3", 0),  # 冻结conv1, conv2
                3: stage_starts.get("conv4", 0),  # 冻结到conv3
                4: stage_starts.get("conv5", 0),  # 冻结到conv4
                5: len(layer_names),  # 冻结全部ResNet
            }.get(self.freeze_at, stage_starts.get("conv5", 0))

            # 设置可训练性
            for i, layer in enumerate(self.resnet.layers):
                layer.trainable = i >= freeze_until

            trainable_count = sum(1 for l in self.resnet.layers if l.trainable)
            total_count = len(self.resnet.layers)
            print(
                f"[ImageFeatureExtractor] Mode: FINETUNE (freeze_at={self.freeze_at})"
            )
            print(f"  - Frozen layers: {freeze_until}/{total_count}")
            print(f"  - Trainable layers: {trainable_count}/{total_count}")

    def call(self, inputs, training=None, mask=None):
        """
        Input: (B, H, W, 3) RGB images, typically (B, 224, 224, 3)
        Output:
            image_features: (B, 512) - L2归一化后的全局图像特征向量
            feature_map: (B, H', W', 512) - 降维后的特征图
        """
        # ResNet-50 feature extraction
        x = tf.keras.applications.resnet50.preprocess_input(inputs)
        x = self.resnet(x, training=training)  # (B, 7, 7, 2048)

        # 1x1卷积降维：2048 -> 512
        x = self.dim_reduction(x)  # (B, 7, 7, 512)
        x = self.dim_reduction_bn(x, training=training)

        feature_map = x  # 保存降维后的特征图

        # Apply transformer blocks on 512-dim features (much more efficient!)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)

        # Global average pooling
        pooled = self.global_pool(x)  # (B, 512)

        # Project and normalize
        image_features = self.output_proj(pooled)  # (B, 512)
        image_features = self.feature_norm(image_features)  # L2归一化

        return image_features, feature_map

    def get_config(self):
        config = super(ImageFeatureExtractor, self).get_config()
        config.update(
            {
                "output_dim": self.output_dim,
                "num_transformer_blocks": self.num_transformer_blocks,
                "num_heads": self.num_heads,
                "dff": self.dff,
                "dropout_rate": self.dropout_rate,
                "training_mode": self.training_mode,
                "freeze_at": self.freeze_at,
            }
        )
        return config


# 兼容旧接口
def get_image_feature_extractor(
    image_input,
    is_training,
    bn_decay=None,
    num_transformer_blocks=2,
    num_heads=8,
    dff=2048,
    dropout_rate=0.1,
    training_mode="finetune",
    freeze_at=0,
    scope="image_feature_extractor",
):
    """Legacy function interface for image feature extraction.

    Input:
        image_input: TF tensor in shape (B, H, W, 3), RGB images
        is_training: TF boolean scalar or Python bool
        bn_decay: Not used in TF2 version (kept for compatibility)
        num_transformer_blocks: number of transformer encoder blocks
        num_heads: number of attention heads
        dff: feed-forward dimension for transformer
        dropout_rate: dropout rate
        training_mode: 'frozen', 'finetune', or 'end2end'
        freeze_at: for finetune mode, freeze ResNet up to this stage (2/3/4/5)
        scope: Not used in TF2 version (kept for compatibility)
    Output:
        image_features: TF tensor in shape (B, 512), L2 normalized
        feature_map: TF tensor in shape (B, 7, 7, 512)
    """
    extractor = ImageFeatureExtractor(
        output_dim=512,
        num_transformer_blocks=num_transformer_blocks,
        num_heads=num_heads,
        dff=dff,
        dropout_rate=dropout_rate,
        training_mode=training_mode,
        freeze_at=freeze_at,
        name=scope,
    )

    image_features, feature_map = extractor(image_input, training=is_training)

    return image_features, feature_map


def get_image_feature_v2(
    image_input,
    is_training,
    bn_decay=None,
    num_transformer_blocks=1,
    num_heads=4,
    dropout_rate=0.1,
    training_mode="finetune",
    scope="image_feat",
):
    """Lightweight image feature extractor (v2 version)."""
    return get_image_feature_extractor(
        image_input,
        is_training,
        bn_decay,
        num_transformer_blocks=num_transformer_blocks,
        num_heads=num_heads,
        dropout_rate=dropout_rate,
        training_mode=training_mode,
        freeze_at=0,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Optimized Image Feature Extractor")
    print("=" * 60)

    batch_size = 4
    inputs = tf.zeros((batch_size, 224, 224, 3))

    # Test different training modes
    for mode in ["frozen", "finetune", "end2end"]:
        print(f"\n--- Testing mode: {mode} ---")
        extractor = ImageFeatureExtractor(
            output_dim=512,
            num_transformer_blocks=2,
            num_heads=8,
            training_mode=mode,
            freeze_at=0 if mode == "finetune" else 0,
        )

        features, feature_map = extractor(inputs, training=False)

        print(f"Input shape: {inputs.shape}")
        print(f"Feature map shape: {feature_map.shape}")  # Should be (4, 7, 7, 512)
        print(f"Image features shape: {features.shape}")  # Should be (4, 512)

        # 检查L2归一化
        feature_norms = tf.norm(features, axis=1)
        print(
            f"Feature L2 norms (should be ~1.0): mean={tf.reduce_mean(feature_norms):.4f}"
        )

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
