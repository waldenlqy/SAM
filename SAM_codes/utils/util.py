import uuid
import numpy as np
import tensorflow as tf


def ltr_relu(inputs):

    # Define the op in python
    def _relu(x):
        return np.maximum(x, 0.)

    # Define the op's gradient in python
    def _relu_grad(x, grad):
        alpha = 0.01
        limit = -0.01
        g = np.float32(x > 0)
        g[grad < 0] = 1
        g[np.bitwise_and(grad > 0, np.bitwise_and(x < 0, x > limit))] = alpha
        return g

    # An adapter that defines a gradient op compatible with Tensorflow
    def _relu_grad_op(op, grad):
        x = op.inputs[0]
        x_grad = grad * tf.py_func(_relu_grad, [x, grad], tf.float32)
        return x_grad

    # Register the gradient with a unique id
    grad_name = "LtrReluGrad_" + str(uuid.uuid4())
    tf.RegisterGradient(grad_name)(_relu_grad_op)
    # Override the gradient of the custom op
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": grad_name}):
        output = tf.py_func(_relu, [inputs], tf.float32)
    return output


def prelu(_x, scope=None):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1),
                                 collections=[tf.GraphKeys.MODEL_VARIABLES, tf.GraphKeys.TRAINABLE_VARIABLES, tf.GraphKeys.GLOBAL_VARIABLES])
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def get_act_fn(name):
    activations = {
        'relu': tf.nn.relu,
        'lrelu': lambda x: tf.nn.leaky_relu(x, alpha=0.01),
        'tanh': tf.tanh,
        'sigmoid': tf.sigmoid,
        'softmax': tf.nn.softmax,
        'elu': tf.nn.elu,
        'softplus': tf.nn.softplus,
        'ltr_relu': ltr_relu,
        'prelu' : prelu,
        'crelu':tf.nn.crelu
    }
    return activations[name]
