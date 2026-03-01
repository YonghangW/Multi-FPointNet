"""Frustum PointNets v1 with Lite Image Fusion (只在回归网络融合).

设计思路：
- 3D Instance Segmentation：只用点云（不融合图像）
- 3D Box Estimation：用图像特征替代/增强one-hot向量
- 简化设计，减少计算量，聚焦关键融合点

Author: Charles R. Qi (modified for TF2 compatibility)
Date: 2024
"""

from __future__ import print_function

import sys
import os
import tensorflow as tf
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, "utils"))
import tf_util
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT
from model_util import point_cloud_masking, get_center_regression_net
from model_util import placeholder_inputs, parse_output_to_tensors, get_loss

# 使用优化后的图像特征提取器
from image_feature_extractor import get_image_feature_extractor


def get_instance_seg_v1_net(
    point_cloud, one_hot_vec, is_training, bn_decay, end_points
):
    """3D实例分割网络（纯点云，不融合图像特征）.
    
    分割任务主要依赖几何形状，图像特征帮助有限。
    """
    batch_size = (
        point_cloud.get_shape()[0]
        if hasattr(point_cloud.get_shape()[0], "value")
        else tf.shape(point_cloud)[0]
    )
    num_point = (
        point_cloud.get_shape()[1]
        if hasattr(point_cloud.get_shape()[1], "value")
        else tf.shape(point_cloud)[1]
    )

    net = tf.expand_dims(point_cloud, 2)
    net = tf_util.conv2d(
        net, 64, [1, 1], padding="VALID", stride=[1, 1],
        bn=True, is_training=is_training, scope="conv1", bn_decay=bn_decay,
    )
    net = tf_util.conv2d(
        net, 64, [1, 1], padding="VALID", stride=[1, 1],
        bn=True, is_training=is_training, scope="conv2", bn_decay=bn_decay,
    )
    point_feat = tf_util.conv2d(
        net, 64, [1, 1], padding="VALID", stride=[1, 1],
        bn=True, is_training=is_training, scope="conv3", bn_decay=bn_decay,
    )
    net = tf_util.conv2d(
        point_feat, 128, [1, 1], padding="VALID", stride=[1, 1],
        bn=True, is_training=is_training, scope="conv4", bn_decay=bn_decay,
    )
    net = tf_util.conv2d(
        net, 1024, [1, 1], padding="VALID", stride=[1, 1],
        bn=True, is_training=is_training, scope="conv5", bn_decay=bn_decay,
    )
    global_feat = tf_util.max_pool2d(net, [num_point, 1], padding="VALID", scope="maxpool")

    # 只拼接one-hot（不融合图像特征）
    global_feat = tf.concat(
        [global_feat, tf.expand_dims(tf.expand_dims(one_hot_vec, 1), 1)], axis=3
    )
    global_feat_expand = tf.tile(global_feat, [1, num_point, 1, 1])
    concat_feat = tf.concat(axis=3, values=[point_feat, global_feat_expand])

    net = tf_util.conv2d(
        concat_feat, 512, [1, 1], padding="VALID", stride=[1, 1],
        bn=True, is_training=is_training, scope="conv6", bn_decay=bn_decay,
    )
    net = tf_util.conv2d(
        net, 256, [1, 1], padding="VALID", stride=[1, 1],
        bn=True, is_training=is_training, scope="conv7", bn_decay=bn_decay,
    )
    net = tf_util.conv2d(
        net, 128, [1, 1], padding="VALID", stride=[1, 1],
        bn=True, is_training=is_training, scope="conv8", bn_decay=bn_decay,
    )
    net = tf_util.conv2d(
        net, 128, [1, 1], padding="VALID", stride=[1, 1],
        bn=True, is_training=is_training, scope="conv9", bn_decay=bn_decay,
    )
    net = tf_util.dropout(net, is_training, "dp1", keep_prob=0.5)

    logits = tf_util.conv2d(
        net, 2, [1, 1], padding="VALID", stride=[1, 1],
        activation_fn=None, scope="conv10",
    )
    logits = tf.squeeze(logits, [2])
    return logits, end_points


def get_3d_box_estimation_v1_net_with_image(
    object_point_cloud, image_features, one_hot_vec, is_training, bn_decay, end_points
):
    """3D框估计网络（融合图像特征替代one-hot）.
    
    关键改进：
    1. 图像特征和one-hot拼接，提供更丰富的语义信息
    2. 点云特征L2归一化，与图像特征对齐
    3. 融合后通过FC层降维
    
    Input:
        object_point_cloud: (B,M,C) - 分割后的点云
        image_features: (B, 512) - L2归一化的图像特征
        one_hot_vec: (B,3) - 类别编码（可选，可与图像特征拼接）
    """
    num_point = (
        object_point_cloud.get_shape()[1]
        if hasattr(object_point_cloud.get_shape()[1], "value")
        else tf.shape(object_point_cloud)[1]
    )
    
    net = tf.expand_dims(object_point_cloud, 2)
    net = tf_util.conv2d(
        net, 128, [1, 1], padding="VALID", stride=[1, 1],
        bn=True, is_training=is_training, scope="conv-reg1", bn_decay=bn_decay,
    )
    net = tf_util.conv2d(
        net, 128, [1, 1], padding="VALID", stride=[1, 1],
        bn=True, is_training=is_training, scope="conv-reg2", bn_decay=bn_decay,
    )
    net = tf_util.conv2d(
        net, 256, [1, 1], padding="VALID", stride=[1, 1],
        bn=True, is_training=is_training, scope="conv-reg3", bn_decay=bn_decay,
    )
    net = tf_util.conv2d(
        net, 512, [1, 1], padding="VALID", stride=[1, 1],
        bn=True, is_training=is_training, scope="conv-reg4", bn_decay=bn_decay,
    )
    
    # 全局池化得到点云特征 (B, 512)
    net = tf_util.max_pool2d(net, [num_point, 1], padding="VALID", scope="maxpool2")
    point_cloud_feat = tf.squeeze(net, axis=[1, 2])
    
    # L2归一化
    point_cloud_feat = tf.nn.l2_normalize(point_cloud_feat, axis=1)
    
    # === 关键融合：图像特征 + one-hot ===
    # 策略1：图像特征完全替代one-hot
    # semantic_feat = image_features  # (B, 512)
    
    # 策略2：图像特征与one-hot拼接（更稳定）
    semantic_feat = tf.concat([image_features, one_hot_vec], axis=1)  # (B, 515)
    
    # 将语义特征投影到与点云特征相同维度，便于融合
    semantic_feat = tf_util.fully_connected(
        semantic_feat, 512, scope="semantic_proj", 
        bn=True, is_training=is_training, bn_decay=bn_decay
    )
    
    # 融合：点云特征 + 语义特征
    fused_feat = tf.concat([point_cloud_feat, semantic_feat], axis=1)  # (B, 1024)
    
    # FC层回归
    net = tf_util.fully_connected(
        fused_feat, 512, scope="fc1", bn=True, is_training=is_training, bn_decay=bn_decay
    )
    net = tf_util.dropout(net, is_training, "fc1_dp", keep_prob=0.5)
    
    net = tf_util.fully_connected(
        net, 256, scope="fc2", bn=True, is_training=is_training, bn_decay=bn_decay
    )
    net = tf_util.dropout(net, is_training, "fc2_dp", keep_prob=0.5)

    # 输出：3(center) + heading + size
    output = tf_util.fully_connected(
        net,
        3 + NUM_HEADING_BIN * 2 + NUM_SIZE_CLUSTER * 4,
        activation_fn=None,
        scope="fc3",
    )
    return output, end_points


def get_model_with_image(
    point_cloud, image_input, one_hot_vec, is_training, bn_decay=None,
    image_training_mode="finetune",
    image_freeze_at=4,
):
    """Lite融合版本的Frustum PointNets.
    
    只在3D框估计阶段融合图像特征，简化设计。
    
    Args:
        point_cloud: (B,N,4) - frustum点云
        image_input: (B, H, W, 3) - RGB图像块
        one_hot_vec: (B,3) - 类别one-hot
        is_training: bool
        bn_decay: batch norm decay
        image_training_mode: 图像网络训练模式 ('frozen', 'finetune', 'end2end')
        image_freeze_at: finetune模式下冻结层数
        
    Returns:
        end_points: dict with all predictions
    """
    end_points = {}

    # 提取图像特征
    image_features, image_feature_map = get_image_feature_extractor(
        image_input, is_training, bn_decay, 
        num_transformer_blocks=2, num_heads=8,
        training_mode=image_training_mode,
        freeze_at=image_freeze_at,
        scope="image_feat_extractor"
    )
    end_points["image_features"] = image_features
    end_points["image_feature_map"] = image_feature_map

    # 3D实例分割（只用点云）
    logits, end_points = get_instance_seg_v1_net(
        point_cloud, one_hot_vec, is_training, bn_decay, end_points
    )
    end_points["mask_logits"] = logits

    # 根据分割mask提取目标点云
    object_point_cloud_xyz, mask_xyz_mean, end_points = point_cloud_masking(
        point_cloud, logits, end_points
    )

    # T-Net粗估计中心
    center_delta, end_points = get_center_regression_net(
        object_point_cloud_xyz, one_hot_vec, is_training, bn_decay, end_points
    )
    stage1_center = center_delta + mask_xyz_mean
    end_points["stage1_center"] = stage1_center
    
    # 转换到对象坐标系
    object_point_cloud_xyz_new = object_point_cloud_xyz - tf.expand_dims(center_delta, 1)

    # 3D框估计（融合图像特征）
    output, end_points = get_3d_box_estimation_v1_net_with_image(
        object_point_cloud_xyz_new,
        image_features,
        one_hot_vec,
        is_training,
        bn_decay,
        end_points,
    )

    # 解析输出
    end_points = parse_output_to_tensors(output, end_points)
    end_points["center"] = end_points["center_boxnet"] + stage1_center

    return end_points


# 保持原始模型兼容性
def get_model(point_cloud, one_hot_vec, is_training, bn_decay=None):
    """原始Frustum PointNets模型（无图像）."""
    end_points = {}

    logits, end_points = get_instance_seg_v1_net(
        point_cloud, one_hot_vec, is_training, bn_decay, end_points
    )
    end_points["mask_logits"] = logits

    object_point_cloud_xyz, mask_xyz_mean, end_points = point_cloud_masking(
        point_cloud, logits, end_points
    )

    center_delta, end_points = get_center_regression_net(
        object_point_cloud_xyz, one_hot_vec, is_training, bn_decay, end_points
    )
    stage1_center = center_delta + mask_xyz_mean
    end_points["stage1_center"] = stage1_center
    object_point_cloud_xyz_new = object_point_cloud_xyz - tf.expand_dims(center_delta, 1)

    # 原始回归网络（无图像）
    num_point = object_point_cloud_xyz_new.get_shape()[1].value
    net = tf.expand_dims(object_point_cloud_xyz_new, 2)
    net = tf_util.conv2d(net, 128, [1, 1], padding="VALID", stride=[1, 1],
                         bn=True, is_training=is_training, scope="conv-reg1", bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1], padding="VALID", stride=[1, 1],
                         bn=True, is_training=is_training, scope="conv-reg2", bn_decay=bn_decay)
    net = tf_util.conv2d(net, 256, [1, 1], padding="VALID", stride=[1, 1],
                         bn=True, is_training=is_training, scope="conv-reg3", bn_decay=bn_decay)
    net = tf_util.conv2d(net, 512, [1, 1], padding="VALID", stride=[1, 1],
                         bn=True, is_training=is_training, scope="conv-reg4", bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point, 1], padding="VALID", scope="maxpool2")
    net = tf.squeeze(net, axis=[1, 2])
    net = tf.concat([net, one_hot_vec], axis=1)
    net = tf_util.fully_connected(net, 512, scope="fc1", bn=True, is_training=is_training, bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, scope="fc2", bn=True, is_training=is_training, bn_decay=bn_decay)
    output = tf_util.fully_connected(net, 3 + NUM_HEADING_BIN * 2 + NUM_SIZE_CLUSTER * 4,
                                     activation_fn=None, scope="fc3")
    
    end_points = parse_output_to_tensors(output, end_points)
    end_points["center"] = end_points["center_boxnet"] + stage1_center

    return end_points


if __name__ == "__main__":
    print("Testing Frustum PointNets v1 (Lite Image Fusion)...")

    # Test without image
    print("\n1. Testing original model (no image)...")
    with tf.Graph().as_default():
        inputs = tf.zeros((4, 1024, 4))
        outputs = get_model(inputs, tf.ones((4, 3)), tf.constant(True))
        print(f"  mask_logits: {outputs['mask_logits'].shape}")
        print(f"  center: {outputs['center'].shape}")

    # Test with image - finetune mode
    print("\n2. Testing with image fusion (finetune mode)...")
    with tf.Graph().as_default():
        point_cloud = tf.zeros((4, 1024, 4))
        image_input = tf.zeros((4, 224, 224, 3))
        one_hot_vec = tf.ones((4, 3))

        outputs = get_model_with_image(
            point_cloud, image_input, one_hot_vec, tf.constant(True),
            image_training_mode="finetune", image_freeze_at=4
        )
        print(f"  Image features: {outputs['image_features'].shape}")
        print(f"  Feature map: {outputs['image_feature_map'].shape}")
        print(f"  mask_logits: {outputs['mask_logits'].shape}")
        print(f"  center: {outputs['center'].shape}")

    print("\nAll tests passed!")
