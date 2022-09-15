from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.ops import variable_scope
from model_ops.attention import multihead_mask_attention
from utils.util import get_act_fn

def _create_ubr_network(self, long_seq_input_layer, long_seq_ord_layer, long_seq_ts_layer, item_eb, key_masks, units=16):
        # seq prep
        seq_layer = long_seq_input_layer + long_seq_ord_layer + long_seq_ts_layer  # (512, 1000, 32)
        # seq_layer = long_seq_input_layer  # backup sequence

        # base operation: reduce mean
        # s = long_seq_input_layer.shape.as_list()
        # k = tf.cast(key_masks, tf.float32)
        # num_true = tf.reduce_sum(k, axis=2)
        # num_true_no_nan = tf.where(tf.equal(num_true, 0), tf.ones_like(num_true), num_true)
        # k = tf.tile(k, [1, s[2], 1])
        # k = tf.transpose(k, [0, 2, 1])
        # total = tf.reduce_sum(long_seq_input_layer*k, axis=1)
        # long_seq_input_layer = total / num_true_no_nan
        # end base

        # key_masks = tf.to_float(tf.squeeze(key_masks, axis=1))  # (512, 1000)
        key_masks = tf.squeeze(key_masks, axis=1)

        # sample probability
        with variable_scope.variable_scope('sample'):
            outputs, _ = attention(
                queries=seq_layer,
                queries_length=None,
                keys=seq_layer,
                keys_length=None,
                scope="attention",
                reuse=None,
                query_masks=key_masks,
                key_masks=key_masks)
            probs = layers.fully_connected(outputs, 20, activation_fn=tf.nn.relu)
            probs = layers.fully_connected(probs, 1, activation_fn=tf.nn.sigmoid)
            probs = tf.squeeze(probs, axis=2)

            uniform = tf.random_uniform(tf.shape(probs), 0, 1)
            condition = probs - uniform
            index = tf.where(condition >= 0, tf.ones_like(probs), tf.zeros_like(probs))
            log_probs = tf.log(tf.clip_by_value(probs, 1e-10, 1))

        key_masks = tf.logical_and(key_masks, tf.cast(index, tf.bool))
        with variable_scope.variable_scope('lsq'):
            outputs, _ = attention(
                queries=item_eb,
                queries_length=None,
                keys=seq_layer,
                keys_length=None,
                scope="attention",
                reuse=None,
                query_masks=None,
                key_masks=key_masks)
            outputs = tf.reduce_sum(outputs, axis=1)
