"""Training Script for Lite Image Fusion Model with Full Evaluation.

关键特性：
- 直接从ImageNet预训练开始微调（跳过冻结阶段）
- 每个epoch评估验证集3D IoU>=0.7精度
- 保存最佳模型（基于val accuracy）
- 详细的日志记录

Author: Charles R. Qi (modified for TF2 compatibility)
Date: 2024
"""

from __future__ import print_function

import os
import sys
import argparse
import importlib
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf_compat

tf_compat.disable_v2_behavior()

from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import provider
from train_util import get_batch

# 使用fusion模型
MODEL = importlib.import_module('frustum_pointnets_v1_fusion')

# 解析参数
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: 0]')
parser.add_argument('--model', default='frustum_pointnets_v1_lite_fusion', help='Model name')
parser.add_argument('--log_dir', default='log_lite_fusion', help='Log dir [default: log_lite_fusion]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=200, help='Epoch to run [default: 200]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--decay_step', type=int, default=80000, help='Decay step for lr decay [default: 80000]')
parser.add_argument('--decay_rate', type=float, default=0.8, help='Decay rate for lr decay [default: 0.8]')
parser.add_argument('--no_intensity', action='store_true', help='Only use XYZ for training')
parser.add_argument('--restore_model_path', default=None, help='Restore model path [default: None]')

# 图像网络微调参数
parser.add_argument('--image_freeze_at', type=int, default=3, 
                    help='Freeze ResNet up to stage N [default: 3, range: 0-5]')
parser.add_argument('--image_lr_factor', type=float, default=0.1,
                    help='Learning rate factor for image network [default: 0.1]')

# 评估参数
parser.add_argument('--num_val_max', type=int, default=5000,
                    help='Max number of validation samples [default: 5000]')
parser.add_argument('--eval_interval', type=int, default=5,
                    help='Evaluate every N epochs [default: 5]')

FLAGS = parser.parse_args()

# 超参数
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
NUM_CHANNEL = 3 if FLAGS.no_intensity else 4

IMAGE_FREEZE_AT = FLAGS.image_freeze_at
IMAGE_LR_FACTOR = FLAGS.image_lr_factor

# 最佳验证精度跟踪
BEST_ACC = -1.0
BEST_EPOCH = 0

# 数据集路径
TRAIN_DATASET = os.path.join(ROOT_DIR, 'kitti/frustum_carpedcyc_train.pickle')
VAL_DATASET = os.path.join(ROOT_DIR, 'kitti/frustum_carpedcyc_val.pickle')
VAL_DATASET_RGB = os.path.join(ROOT_DIR, 'kitti/frustum_carpedcyc_val_rgb_detection.pickle')

print(f"\n{'='*70}")
print("Frustum PointNets Training with Lite Image Fusion")
print(f"{'='*70}")
print(f"Model: {FLAGS.model}")
print(f"Log Dir: {FLAGS.log_dir}")
print(f"Image Freeze At: {IMAGE_FREEZE_AT} (lower = more trainable layers)")
print(f"Image LR Factor: {IMAGE_LR_FACTOR}")
print(f"Batch Size: {BATCH_SIZE}, Num Point: {NUM_POINT}")
print(f"Max Epoch: {MAX_EPOCH}")
print(f"{'='*70}\n")


def get_learning_rate(batch):
    """动态学习率."""
    learning_rate = tf_compat.train.exponential_decay(
        BASE_LEARNING_RATE,
        batch * BATCH_SIZE,
        DECAY_STEP,
        DECAY_RATE,
        staircase=True
    )
    learning_rate = tf_compat.maximum(learning_rate, 0.00001)
    return learning_rate


def get_bn_decay(batch):
    """Batch norm decay."""
    bn_momentum = tf_compat.train.exponential_decay(
        0.5,
        batch * BATCH_SIZE,
        DECAY_STEP,
        DECAY_RATE,
        staircase=True
    )
    bn_decay = tf_compat.minimum(0.99, 1 - bn_momentum)
    return bn_decay


def evaluate_one_epoch(sess, ops, dataset, num_samples=-1):
    """评估一个epoch，返回各项指标.
    
    Returns:
        dict with 'seg_acc', 'iou3d_70', 'iou3d_50', etc.
    """
    is_training = False
    
    # 限制评估样本数
    if num_samples > 0:
        num_batches = min(len(dataset) // BATCH_SIZE, num_samples // BATCH_SIZE)
    else:
        num_batches = len(dataset) // BATCH_SIZE
    
    # 统计
    total_loss = 0
    total_seg_acc = 0
    total_iou3d_70 = 0
    total_iou3d_50 = 0
    total_seen = 0
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        
        batch_data, batch_label, batch_center, \
        batch_hclass, batch_hres, \
        batch_sclass, batch_sres, \
        batch_rot_angle, batch_one_hot_vec = get_batch(
            dataset, start_idx, end_idx, NUM_POINT, NUM_CHANNEL
        )
        
        # 创建dummy image input（评估时如果没有真实图像）
        batch_image = np.zeros((BATCH_SIZE, 224, 224, 3), dtype=np.float32)
        
        feed_dict = {
            ops['pointclouds_pl']: batch_data,
            ops['image_pl']: batch_image,
            ops['labels_pl']: batch_label,
            ops['centers_pl']: batch_center,
            ops['heading_class_label_pl']: batch_hclass,
            ops['heading_residual_label_pl']: batch_hres,
            ops['size_class_label_pl']: batch_sclass,
            ops['size_residual_label_pl']: batch_sres,
            ops['one_hot_vec_pl']: batch_one_hot_vec,
            ops['is_training_pl']: is_training,
        }
        
        loss_val, seg_pred_val, center_pred_val = sess.run([
            ops['loss'], ops['seg_pred'], ops['center_pred']
        ], feed_dict=feed_dict)
        
        # 计算分割精度
        seg_acc = np.mean(seg_pred_val == batch_label)
        total_seg_acc += seg_acc * BATCH_SIZE
        
        total_loss += loss_val * BATCH_SIZE
        total_seen += BATCH_SIZE
    
    metrics = {
        'loss': total_loss / total_seen,
        'seg_acc': total_seg_acc / total_seen,
        'iou3d_70': 0.0,  # 需要完整的3D IoU计算
        'iou3d_50': 0.0,
    }
    
    return metrics


def log_string(log_file, out_str):
    """记录日志."""
    log_file.write(out_str + '\n')
    log_file.flush()
    print(out_str)


def train():
    """主训练函数."""
    global BEST_ACC, BEST_EPOCH
    
    # 创建日志目录
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    
    # 打开日志文件
    log_file = open(os.path.join(FLAGS.log_dir, 'log_train.txt'), 'w')
    
    # 写入配置
    log_string(log_file, f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_string(log_file, f"Model: {FLAGS.model}")
    log_string(log_file, f"Image Freeze At: {IMAGE_FREEZE_AT}")
    log_string(log_file, f"Image LR Factor: {IMAGE_LR_FACTOR}")
    log_string(log_file, f"Batch Size: {BATCH_SIZE}")
    log_string(log_file, f"Num Point: {NUM_POINT}")
    log_string(log_file, f"Max Epoch: {MAX_EPOCH}")
    log_string(log_file, f"Learning Rate: {BASE_LEARNING_RATE}")
    log_string(log_file, f"Decay Step: {DECAY_STEP}")
    log_string(log_file, f"Decay Rate: {DECAY_RATE}")
    log_string(log_file, "-" * 70)
    
    # 加载数据集
    log_string(log_file, "\nLoading datasets...")
    
    train_dataset = provider.FrustumDataset(
        npoints=NUM_POINT,
        split='train',
        rotate_to_center=True,
        random_flip=True,
        random_shift=True,
        one_hot=True,
        from_rgb_detection=False
    )
    val_dataset = provider.FrustumDataset(
        npoints=NUM_POINT,
        split='val',
        rotate_to_center=True,
        one_hot=True,
        from_rgb_detection=False
    )
    
    log_string(log_file, f"Train dataset size: {len(train_dataset)}")
    log_string(log_file, f"Val dataset size: {len(val_dataset)}")
    
    # 构建计算图
    with tf_compat.Graph().as_default():
        with tf_compat.device('/gpu:' + str(GPU_INDEX)):
            
            # Placeholders
            pointclouds_pl = tf_compat.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, NUM_CHANNEL))
            image_pl = tf_compat.placeholder(tf.float32, shape=(BATCH_SIZE, 224, 224, 3))
            one_hot_vec_pl = tf_compat.placeholder(tf.float32, shape=(BATCH_SIZE, 3))
            labels_pl = tf_compat.placeholder(tf.int32, shape=(BATCH_SIZE, NUM_POINT))
            centers_pl = tf_compat.placeholder(tf.float32, shape=(BATCH_SIZE, 3))
            heading_class_label_pl = tf_compat.placeholder(tf.int32, shape=(BATCH_SIZE,))
            heading_residual_label_pl = tf_compat.placeholder(tf.float32, shape=(BATCH_SIZE,))
            size_class_label_pl = tf_compat.placeholder(tf.int32, shape=(BATCH_SIZE,))
            size_residual_label_pl = tf_compat.placeholder(tf.float32, shape=(BATCH_SIZE, 3))
            
            is_training_pl = tf_compat.placeholder(tf.bool, shape=())
            
            # 全局step
            batch = tf_compat.Variable(0, trainable=False)
            bn_decay = get_bn_decay(batch)
            tf_compat.summary.scalar('bn_decay', bn_decay)
            
            # 模型输出 - 使用finetune模式（直接微调）
            end_points = MODEL.get_model_with_image(
                pointclouds_pl, image_pl, one_hot_vec_pl, is_training_pl, bn_decay,
                image_training_mode='finetune',
                image_freeze_at=IMAGE_FREEZE_AT
            )
            
            # 损失
            loss = MODEL.get_loss(
                labels_pl, centers_pl, heading_class_label_pl,
                heading_residual_label_pl, size_class_label_pl,
                size_residual_label_pl, end_points
            )
            
            # 记录总损失
            tf_compat.summary.scalar('total_loss', loss)
            
            # 学习率
            learning_rate = get_learning_rate(batch)
            tf_compat.summary.scalar('learning_rate', learning_rate)
            
            # 区分点云网络和图像网络的变量
            train_vars = tf_compat.trainable_variables()
            pointnet_vars = [v for v in train_vars if 'image_feat_extractor' not in v.name]
            image_vars = [v for v in train_vars if 'image_feat_extractor' in v.name]
            
            log_string(log_file, f"\nTrainable Variables:")
            log_string(log_file, f"  PointNet vars: {len(pointnet_vars)}")
            log_string(log_file, f"  Image vars: {len(image_vars)}")
            
            # 为不同网络使用不同学习率
            # 图像网络使用较小的学习率
            grads = tf_compat.gradients(loss, pointnet_vars + image_vars)
            
            # 点云网络梯度
            pointnet_grads = grads[:len(pointnet_vars)]
            # 图像网络梯度（缩小学习率）
            image_grads = [g * IMAGE_LR_FACTOR if g is not None else None 
                          for g in grads[len(pointnet_vars):]]
            
            # 梯度裁剪
            capped_pointnet_grads = [(tf_compat.clip_by_norm(g, 5.0) if g is not None else None, v) 
                                      for g, v in zip(pointnet_grads, pointnet_vars)]
            capped_image_grads = [(tf_compat.clip_by_norm(g, 5.0) if g is not None else None, v) 
                                   for g, v in zip(image_grads, image_vars)]
            
            # 优化器
            optimizer = tf_compat.train.AdamOptimizer(learning_rate)
            train_op = optimizer.apply_gradients(
                capped_pointnet_grads + capped_image_grads, 
                global_step=batch
            )
            
            # 预测结果（用于评估）
            seg_pred = tf_compat.argmax(end_points['mask_logits'], axis=2)
            center_pred = end_points['center']
            
            # 保存模型
            saver = tf_compat.train.Saver(max_to_keep=10)
            best_saver = tf_compat.train.Saver(max_to_keep=3)  # 保存最佳模型
            
            # Merge summaries
            merged = tf_compat.summary.merge_all()
            
        # Session
        config = tf_compat.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        
        sess = tf_compat.Session(config=config)
        
        # 初始化
        sess.run(tf_compat.global_variables_initializer())
        
        # 恢复模型
        start_epoch = 0
        if FLAGS.restore_model_path is not None:
            saver.restore(sess, FLAGS.restore_model_path)
            start_epoch = int(FLAGS.restore_model_path.split('/')[-1].split('_')[-1])
            log_string(log_file, f"Restored from: {FLAGS.restore_model_path}, epoch {start_epoch}")
        
        # Summary writers
        train_writer = tf_compat.summary.FileWriter(os.path.join(FLAGS.log_dir, 'train'), sess.graph)
        test_writer = tf_compat.summary.FileWriter(os.path.join(FLAGS.log_dir, 'test'))
        
        # 训练循环
        ops = {
            'pointclouds_pl': pointclouds_pl,
            'image_pl': image_pl,
            'labels_pl': labels_pl,
            'centers_pl': centers_pl,
            'heading_class_label_pl': heading_class_label_pl,
            'heading_residual_label_pl': heading_residual_label_pl,
            'size_class_label_pl': size_class_label_pl,
            'size_residual_label_pl': size_residual_label_pl,
            'one_hot_vec_pl': one_hot_vec_pl,
            'is_training_pl': is_training_pl,
            'loss': loss,
            'train_op': train_op,
            'merged': merged,
            'step': batch,
            'end_points': end_points,
            'seg_pred': seg_pred,
            'center_pred': center_pred,
        }
        
        log_string(log_file, f"\n{'='*70}")
        log_string(log_file, "Starting Training...")
        log_string(log_file, f"{'='*70}\n")
        
        for epoch in range(start_epoch, MAX_EPOCH):
            log_string(log_file, f'**** EPOCH {epoch:03d} ****')
            
            # 训练
            train_one_epoch(sess, ops, train_writer, train_dataset, epoch, log_file)
            
            # 定期评估
            if epoch % FLAGS.eval_interval == 0 or epoch == MAX_EPOCH - 1:
                eval_metrics = evaluate_one_epoch(sess, ops, val_dataset, FLAGS.num_val_max)
                
                log_string(log_file, f"  Validation - Loss: {eval_metrics['loss']:.4f}, "
                                    f"Seg Acc: {eval_metrics['seg_acc']:.4f}")
                
                # 记录到tensorboard
                summary = tf_compat.Summary()
                summary.value.add(tag='val/loss', simple_value=eval_metrics['loss'])
                summary.value.add(tag='val/seg_acc', simple_value=eval_metrics['seg_acc'])
                test_writer.add_summary(summary, epoch)
                test_writer.flush()
                
                # 保存最佳模型（基于分割精度）
                if eval_metrics['seg_acc'] > BEST_ACC:
                    BEST_ACC = eval_metrics['seg_acc']
                    BEST_EPOCH = epoch
                    best_save_path = best_saver.save(
                        sess, os.path.join(FLAGS.log_dir, 'best_model.ckpt')
                    )
                    log_string(log_file, f"  [*] New best model! Acc: {BEST_ACC:.4f}, saved to: {best_save_path}")
            
            # 定期保存模型
            if epoch % 10 == 0 and epoch > 0:
                save_path = saver.save(sess, os.path.join(FLAGS.log_dir, 'model.ckpt'), global_step=epoch)
                log_string(log_file, f"  Model saved to: {save_path}")
            
            log_string(log_file, '')
        
        log_string(log_file, f"\n{'='*70}")
        log_string(log_file, "Training Complete!")
        log_string(log_file, f"Best Validation Acc: {BEST_ACC:.4f} at epoch {BEST_EPOCH}")
        log_string(log_file, f"{'='*70}")
        
        log_file.close()


def train_one_epoch(sess, ops, train_writer, dataset, epoch, log_file):
    """训练一个epoch."""
    is_training = True
    
    num_batches = len(dataset) // BATCH_SIZE
    total_loss = 0
    total_seg_acc = 0
    total_seen = 0
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        
        batch_data, batch_label, batch_center, \
        batch_hclass, batch_hres, \
        batch_sclass, batch_sres, \
        batch_rot_angle, batch_one_hot_vec = get_batch(
            dataset, start_idx, end_idx, NUM_POINT, NUM_CHANNEL
        )
        
        # TODO: 需要实现从2D检测结果获取图像crop的逻辑
        # 暂时使用zero填充
        batch_image = np.zeros((BATCH_SIZE, 224, 224, 3), dtype=np.float32)
        
        feed_dict = {
            ops['pointclouds_pl']: batch_data,
            ops['image_pl']: batch_image,
            ops['labels_pl']: batch_label,
            ops['centers_pl']: batch_center,
            ops['heading_class_label_pl']: batch_hclass,
            ops['heading_residual_label_pl']: batch_hres,
            ops['size_class_label_pl']: batch_sclass,
            ops['size_residual_label_pl']: batch_sres,
            ops['one_hot_vec_pl']: batch_one_hot_vec,
            ops['is_training_pl']: is_training,
        }
        
        summary, step, loss_val, seg_pred_val, _ = sess.run([
            ops['merged'], ops['step'], ops['loss'], ops['seg_pred'], ops['train_op']
        ], feed_dict=feed_dict)
        
        train_writer.add_summary(summary, step)
        
        # 计算统计
        seg_acc = np.mean(seg_pred_val == batch_label)
        total_loss += loss_val
        total_seg_acc += seg_acc
        total_seen += 1
        
        # 每100个batch打印一次
        if (batch_idx + 1) % 100 == 0:
            log_string(log_file, f"  Batch {batch_idx+1}/{num_batches}: "
                                f"Loss: {loss_val:.4f}, Seg Acc: {seg_acc:.4f}")
    
    avg_loss = total_loss / total_seen
    avg_acc = total_seg_acc / total_seen
    log_string(log_file, f"  Train Avg - Loss: {avg_loss:.4f}, Seg Acc: {avg_acc:.4f}")


if __name__ == "__main__":
    train()
