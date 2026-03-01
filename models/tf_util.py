"""Wrapper functions for TensorFlow layers (TF2 Compatible with tf.compat.v1).

Author: Charles R. Qi
Date: November 2017
Modified: 2024 for TF2 compatibility
"""

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf_compat

# Note: Call tf_compat.disable_v2_behavior() in training script before using these functions


def _variable_on_cpu(name, shape, initializer, use_fp16=False):
    """Helper to create a Variable stored on CPU memory."""
    with tf_compat.device("/cpu:0"):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf_compat.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
    """Helper to create an initialized Variable with weight decay."""
    if use_xavier:
        initializer = tf_compat.keras.initializers.GlorotUniform()
    else:
        initializer = tf_compat.truncated_normal_initializer(stddev=stddev)

    var = _variable_on_cpu(name, shape, initializer)
    if wd is not None:
        weight_decay = tf_compat.multiply(
            tf_compat.nn.l2_loss(var), wd, name="weight_loss"
        )
        tf_compat.add_to_collection("losses", weight_decay)
    return var


def conv1d(
    inputs,
    num_output_channels,
    kernel_size,
    scope,
    stride=1,
    padding="SAME",
    data_format="NHWC",
    use_xavier=True,
    stddev=1e-3,
    weight_decay=None,
    activation_fn=tf.nn.relu,
    bn=False,
    bn_decay=None,
    is_training=None,
):
    """1D convolution with non-linear operation."""
    with tf_compat.variable_scope(scope) as sc:
        assert data_format == "NHWC" or data_format == "NCHW"
        if data_format == "NHWC":
            num_in_channels = inputs.get_shape()[-1].value
        elif data_format == "NCHW":
            num_in_channels = inputs.get_shape()[1].value
        kernel_shape = [kernel_size, num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay(
            "weights",
            shape=kernel_shape,
            use_xavier=use_xavier,
            stddev=stddev,
            wd=weight_decay,
        )
        outputs = tf_compat.nn.conv1d(
            inputs, kernel, stride=stride, padding=padding, data_format=data_format
        )
        biases = _variable_on_cpu(
            "biases", [num_output_channels], tf_compat.constant_initializer(0.0)
        )
        outputs = tf_compat.nn.bias_add(outputs, biases, data_format=data_format)

        if bn:
            outputs = batch_norm_for_conv1d(
                outputs,
                is_training,
                bn_decay=bn_decay,
                scope="bn",
                data_format=data_format,
            )

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def conv2d(
    inputs,
    num_output_channels,
    kernel_size,
    scope,
    stride=[1, 1],
    padding="SAME",
    data_format="NHWC",
    use_xavier=True,
    stddev=1e-3,
    weight_decay=None,
    activation_fn=tf.nn.relu,
    bn=False,
    bn_decay=None,
    is_training=None,
):
    """2D convolution with non-linear operation."""
    with tf_compat.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        assert data_format == "NHWC" or data_format == "NCHW"
        if data_format == "NHWC":
            num_in_channels = inputs.get_shape()[-1].value
        elif data_format == "NCHW":
            num_in_channels = inputs.get_shape()[1].value
        kernel_shape = [kernel_h, kernel_w, num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay(
            "weights",
            shape=kernel_shape,
            use_xavier=use_xavier,
            stddev=stddev,
            wd=weight_decay,
        )
        stride_h, stride_w = stride
        outputs = tf_compat.nn.conv2d(
            inputs,
            kernel,
            [1, stride_h, stride_w, 1],
            padding=padding,
            data_format=data_format,
        )
        biases = _variable_on_cpu(
            "biases", [num_output_channels], tf_compat.constant_initializer(0.0)
        )
        outputs = tf_compat.nn.bias_add(outputs, biases, data_format=data_format)

        if bn:
            outputs = batch_norm_for_conv2d(
                outputs,
                is_training,
                bn_decay=bn_decay,
                scope="bn",
                data_format=data_format,
            )

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def conv2d_transpose(
    inputs,
    num_output_channels,
    kernel_size,
    scope,
    stride=[1, 1],
    padding="SAME",
    use_xavier=True,
    stddev=1e-3,
    weight_decay=None,
    activation_fn=tf.nn.relu,
    bn=False,
    bn_decay=None,
    is_training=None,
):
    """2D convolution transpose with non-linear operation."""
    with tf_compat.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w, num_output_channels, num_in_channels]
        kernel = _variable_with_weight_decay(
            "weights",
            shape=kernel_shape,
            use_xavier=use_xavier,
            stddev=stddev,
            wd=weight_decay,
        )
        stride_h, stride_w = stride

        def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
            dim_size *= stride_size
            if padding == "VALID":
                dim_size += max(kernel_size - stride_size, 0)
            return dim_size

        batch_size = inputs.get_shape()[0].value
        height = inputs.get_shape()[1].value
        width = inputs.get_shape()[2].value
        out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
        out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
        output_shape = [batch_size, out_height, out_width, num_output_channels]

        outputs = tf_compat.nn.conv2d_transpose(
            inputs, kernel, output_shape, [1, stride_h, stride_w, 1], padding=padding
        )
        biases = _variable_on_cpu(
            "biases", [num_output_channels], tf_compat.constant_initializer(0.0)
        )
        outputs = tf_compat.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv2d(
                outputs, is_training, bn_decay=bn_decay, scope="bn", data_format="NHWC"
            )

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def conv3d(
    inputs,
    num_output_channels,
    kernel_size,
    scope,
    stride=[1, 1, 1],
    padding="SAME",
    use_xavier=True,
    stddev=1e-3,
    weight_decay=None,
    activation_fn=tf.nn.relu,
    bn=False,
    bn_decay=None,
    is_training=None,
):
    """3D convolution with non-linear operation."""
    with tf_compat.variable_scope(scope) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [
            kernel_d,
            kernel_h,
            kernel_w,
            num_in_channels,
            num_output_channels,
        ]
        kernel = _variable_with_weight_decay(
            "weights",
            shape=kernel_shape,
            use_xavier=use_xavier,
            stddev=stddev,
            wd=weight_decay,
        )
        stride_d, stride_h, stride_w = stride
        outputs = tf_compat.nn.conv3d(
            inputs, kernel, [1, stride_d, stride_h, stride_w, 1], padding=padding
        )
        biases = _variable_on_cpu(
            "biases", [num_output_channels], tf_compat.constant_initializer(0.0)
        )
        outputs = tf_compat.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_conv3d(
                outputs, is_training, bn_decay=bn_decay, scope="bn"
            )

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def fully_connected(
    inputs,
    num_outputs,
    scope,
    use_xavier=True,
    stddev=1e-3,
    weight_decay=None,
    activation_fn=tf.nn.relu,
    bn=False,
    bn_decay=None,
    is_training=None,
):
    """Fully connected layer with non-linear operation."""
    with tf_compat.variable_scope(scope) as sc:
        num_input_units = inputs.get_shape()[-1].value
        weights = _variable_with_weight_decay(
            "weights",
            shape=[num_input_units, num_outputs],
            use_xavier=use_xavier,
            stddev=stddev,
            wd=weight_decay,
        )
        outputs = tf_compat.matmul(inputs, weights)
        biases = _variable_on_cpu(
            "biases", [num_outputs], tf_compat.constant_initializer(0.0)
        )
        outputs = tf_compat.nn.bias_add(outputs, biases)

        if bn:
            outputs = batch_norm_for_fc(outputs, is_training, bn_decay, "bn")

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def max_pool2d(inputs, kernel_size, scope, stride=[2, 2], padding="VALID"):
    """2D max pooling."""
    with tf_compat.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf_compat.nn.max_pool(
            inputs,
            ksize=[1, kernel_h, kernel_w, 1],
            strides=[1, stride_h, stride_w, 1],
            padding=padding,
            name=sc.name,
        )
        return outputs


def avg_pool2d(inputs, kernel_size, scope, stride=[2, 2], padding="VALID"):
    """2D avg pooling."""
    with tf_compat.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf_compat.nn.avg_pool(
            inputs,
            ksize=[1, kernel_h, kernel_w, 1],
            strides=[1, stride_h, stride_w, 1],
            padding=padding,
            name=sc.name,
        )
        return outputs


def max_pool3d(inputs, kernel_size, scope, stride=[2, 2, 2], padding="VALID"):
    """3D max pooling."""
    with tf_compat.variable_scope(scope) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        stride_d, stride_h, stride_w = stride
        outputs = tf_compat.nn.max_pool3d(
            inputs,
            ksize=[1, kernel_d, kernel_h, kernel_w, 1],
            strides=[1, stride_d, stride_h, stride_w, 1],
            padding=padding,
            name=sc.name,
        )
        return outputs


def avg_pool3d(inputs, kernel_size, scope, stride=[2, 2, 2], padding="VALID"):
    """3D avg pooling."""
    with tf_compat.variable_scope(scope) as sc:
        kernel_d, kernel_h, kernel_w = kernel_size
        stride_d, stride_h, stride_w = stride
        outputs = tf_compat.nn.avg_pool3d(
            inputs,
            ksize=[1, kernel_d, kernel_h, kernel_w, 1],
            strides=[1, stride_d, stride_h, stride_w, 1],
            padding=padding,
            name=sc.name,
        )
        return outputs


def batch_norm_template(inputs, is_training, scope, bn_decay, data_format="NHWC"):
    """Batch normalization using tf.compat.v1.layers for TF1 style compatibility."""
    bn_decay = bn_decay if bn_decay is not None else 0.9

    axis = 3 if data_format == "NHWC" else 1
    return tf_compat.layers.batch_normalization(
        inputs,
        axis=axis,
        momentum=bn_decay,
        epsilon=1e-5,
        scope=scope,
        is_training=is_training,
    )


def batch_norm_for_fc(inputs, is_training, bn_decay, scope):
    """Batch normalization on FC data."""
    return batch_norm_template(inputs, is_training, scope, bn_decay)


def batch_norm_for_conv1d(inputs, is_training, bn_decay, scope, data_format):
    """Batch normalization on 1D convolutional maps."""
    return batch_norm_template(inputs, is_training, scope, bn_decay, data_format)


def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope, data_format):
    """Batch normalization on 2D convolutional maps."""
    return batch_norm_template(inputs, is_training, scope, bn_decay, data_format)


def batch_norm_for_conv3d(inputs, is_training, bn_decay, scope):
    """Batch normalization on 3D convolutional maps."""
    return batch_norm_template(inputs, is_training, scope, bn_decay)


def dropout(inputs, is_training, scope, keep_prob=0.5, noise_shape=None):
    """Dropout layer."""
    with tf_compat.variable_scope(scope) as sc:
        outputs = tf_compat.cond(
            is_training,
            lambda: tf_compat.nn.dropout(inputs, keep_prob, noise_shape),
            lambda: inputs,
        )
        return outputs
