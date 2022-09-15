from collections import OrderedDict
from tensorflow.contrib import layers
import tensorflow as tf
from utils.util import get_act_fn
import json

def feedforward(inputs,
                num_units=[2048, 512],
                activation_fn=None,
                scope="feedforward",
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                is_training=True):
    with tf.variable_scope(scope, reuse=reuse):
        outputs = layers.fully_connected(inputs,
                                         num_units[0],
                                         activation_fn=get_act_fn(activation_fn),
                                         variables_collections=variables_collections,
                                         outputs_collections=outputs_collections,
                                         )
        outputs = layers.fully_connected(outputs,
                                         num_units[1],
                                         activation_fn=None,
                                         variables_collections=variables_collections,
                                         outputs_collections=outputs_collections)

    return outputs


def memory(queries,
           keys,
           memory,
           num_units=None,
           num_output_units=None,
           num_heads=8,
           scope="dynamic_memory",
           reuse=None,
           query_masks=None,
           key_masks=None,
           atten_mode='base',
           linear_projection=True,
           is_target_attention=False,
           variables_collections=None,
           outputs_collections=None,
           activation_fn=None,
           first_n_att_weight_report=9,
           atten_weights_collections=None,
           forward_layer_units=[80, 40],
           din_activation_fn='sigmoid',
           atten_activation_fn='softmax',
           is_self_mask=False):
    """Applies multihead attention.
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]

        query_len = queries.get_shape().as_list()[1]  # T_q
        key_len = keys.get_shape().as_list()[1]  # T_k

        # Linear projections, C = # dim or column, T_x = # vectors or actions
        if linear_projection:
            queries_2d = tf.reshape(queries, [-1, queries.get_shape().as_list()[-1]])
            keys_2d = tf.reshape(keys, [-1, keys.get_shape().as_list()[-1]])
            Q = layers.fully_connected(queries_2d,
                                       num_units,
                                       activation_fn=get_act_fn(activation_fn),
                                       variables_collections=variables_collections,
                                       outputs_collections=outputs_collections, scope="Q")  # (N, T_q, C)
            Q = tf.reshape(Q, [-1, queries.get_shape().as_list()[1], Q.get_shape().as_list()[-1]])
            K = layers.fully_connected(keys_2d,
                                       num_units,
                                       activation_fn=get_act_fn(activation_fn),
                                       variables_collections=variables_collections,
                                       outputs_collections=outputs_collections, scope="K")  # (N, T_k, C)
            K = tf.reshape(K, [-1, keys.get_shape().as_list()[1], K.get_shape().as_list()[-1]])
            V = layers.fully_connected(keys_2d,
                                       num_output_units,
                                       activation_fn=get_act_fn(activation_fn),
                                       variables_collections=variables_collections,
                                       outputs_collections=outputs_collections, scope="V")  # (N, T_k, C')
            V = tf.reshape(V, [-1, keys.get_shape().as_list()[1], V.get_shape().as_list()[-1]])
        else:
            Q = queries
            K = keys
            V = keys
            M = memory

        # Split and concat
        if num_heads > 1:
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C'/h)
            M_ = tf.concat(tf.split(M, num_heads, axis=2), axis=0)
        else:
            Q_ = Q
            K_ = K
            V_ = V
            M_ = M

        # Multiplication & Scale
        if atten_mode == 'simcos':
            # Multiplication

            outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k) batch matmul time is not huge
            # Scale
            outputs = outputs * 20
        elif atten_mode == 'cos':
            # Multiplication
            Q_cos = tf.nn.l2_normalize(Q_, dim=-1)
            K_cos = tf.nn.l2_normalize(K_, dim=-1)
            outputs = tf.matmul(Q_cos, K_cos, transpose_b=True)  # (h*N, T_q, T_k) batch matmul time is not huge
            # Scale
            outputs = outputs * 20
        elif atten_mode == 'ln':
            # Layer Norm
            Q_ = layers.layer_norm(Q_, begin_norm_axis=-1, begin_params_axis=-1)
            K_ = layers.layer_norm(K_, begin_norm_axis=-1, begin_params_axis=-1)
            # Multiplication
            outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
            # Scale
            outputs = outputs * (K_.get_shape().as_list()[-1] ** (-0.5))
        elif atten_mode == 'din':
            Q_ = tf.tile(tf.expand_dims(Q_, axis=2), [1, 1, key_len, 1])
            K_ = tf.tile(tf.expand_dims(K_, axis=1), [1, query_len, 1, 1])
            M_ = tf.tile(tf.expand_dims(M_, axis=2), [1, 1, key_len, 1])
            
            din_layer = tf.concat([Q_ - K_, Q_ * K_, M_ - K_, M_ * K_], axis=-1)

            for i, din_units in enumerate(forward_layer_units):
                din_layer = layers.fully_connected(din_layer, din_units, activation_fn=get_act_fn(din_activation_fn),
                                                   scope='din_{}_layer'.format(i))
            din_layer = layers.fully_connected(din_layer, 1, activation_fn=None, scope='din_logits')
            outputs = tf.reshape(din_layer, [-1, query_len, key_len])
        else:
            # Multiplication
            outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
            outputs = outputs * (K_.get_shape().as_list()[-1] ** (-0.5))

        # key Masking
        key_masks = tf.tile(tf.reshape(key_masks, [-1, 1, key_len]), [num_heads, query_len, 1])  # (h*N, T_q, T_k)
        min_paddings = tf.fill(tf.shape(outputs), tf.constant(-2 ** 32 + 1, dtype=tf.float32))

        self_mask = tf.tile(tf.expand_dims(tf.eye(query_len, key_len), axis=0), [tf.shape(key_masks)[0], 1, 1])
        zero_paddings = tf.fill(tf.shape(outputs), tf.constant(0, dtype=tf.float32))
        self_mask = tf.where(key_masks, self_mask, zero_paddings)

        if atten_activation_fn is None:
            outputs = tf.where(key_masks, outputs, zero_paddings)
        else:
            if is_self_mask:
                key_masks = tf.logical_and(key_masks, tf.cast((1 - self_mask), dtype=tf.bool))
            outputs = tf.where(key_masks, outputs, min_paddings)
            outputs = get_act_fn(atten_activation_fn)(outputs)
            if is_self_mask:
                outputs += self_mask

        if not is_target_attention:
            # Query Masking
            query_masks = tf.tile(tf.reshape(query_masks, [-1, query_len]), [num_heads, 1])  # (h*N, T_q)
            outputs = tf.reshape(outputs, [-1, key_len])  # (h*N*T_q, T_k)
            paddings = tf.zeros_like(outputs, dtype=tf.float32)  # (h*N*T_q, T_k)
            outputs = tf.where(tf.reshape(query_masks, [-1]), outputs,
                               paddings)  # tf.where((h*N*T_q), (h*N*T_q, T_k), (h*N*T_q, T_k)) => (h*N*T_q, T_k)
            outputs = tf.reshape(outputs, [-1, query_len, key_len])  # (h*N, T_q, T_k)

        # Attention vector
        att_vec = outputs
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C'/h)

        # Restore shape
        if num_heads > 1:
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C')

    return outputs, att_vec

def multihead_mask_attention(queries,
                             keys,
                             num_units=None,
                             num_output_units=None,
                             num_heads=8,
                             scope="multihead_soft_attention",
                             reuse=None,
                             query_masks=None,
                             key_masks=None,
                             atten_mode='base',
                             linear_projection=True,
                             is_target_attention=False,
                             variables_collections=None,
                             outputs_collections=None,
                             activation_fn=None,
                             first_n_att_weight_report=9,
                             atten_weights_collections=None,
                             din_layer_units=[80, 40],
                             din_activation_fn='sigmoid',
                             atten_activation_fn='softmax',
                             is_self_mask=False):
    """Applies multihead attention.

    Returns
        A 3d tensor with shape of (N, T_q, C)
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list()[-1]

        query_len = queries.get_shape().as_list()[1]  # T_q
        key_len = keys.get_shape().as_list()[1]  # T_k

        # Linear projections, C = # dim or column, T_x = # vectors or actions
        if linear_projection:
            queries_2d = tf.reshape(queries, [-1, queries.get_shape().as_list()[-1]])
            keys_2d = tf.reshape(keys, [-1, keys.get_shape().as_list()[-1]])
            Q = layers.fully_connected(queries_2d,
                                       num_units,
                                       activation_fn=get_act_fn(activation_fn),
                                       variables_collections=variables_collections,
                                       outputs_collections=outputs_collections, scope="Q")  # (N, T_q, C)
            Q = tf.reshape(Q, [-1, queries.get_shape().as_list()[1], Q.get_shape().as_list()[-1]])
            K = layers.fully_connected(keys_2d,
                                       num_units,
                                       activation_fn=get_act_fn(activation_fn),
                                       variables_collections=variables_collections,
                                       outputs_collections=outputs_collections, scope="K")  # (N, T_k, C)
            K = tf.reshape(K, [-1, keys.get_shape().as_list()[1], K.get_shape().as_list()[-1]])
            V = layers.fully_connected(keys_2d,
                                       num_output_units,
                                       activation_fn=get_act_fn(activation_fn),
                                       variables_collections=variables_collections,
                                       outputs_collections=outputs_collections, scope="V")  # (N, T_k, C')
            V = tf.reshape(V, [-1, keys.get_shape().as_list()[1], V.get_shape().as_list()[-1]])
        else:
            Q = queries
            K = keys
            V = keys

        # Split and concat
        if num_heads > 1:
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C'/h)
        else:
            Q_ = Q
            K_ = K
            V_ = V

        # Multiplication & Scale
        if atten_mode == 'cos':
            # Multiplication
            Q_cos = tf.nn.l2_normalize(Q_, dim=-1)
            K_cos = tf.nn.l2_normalize(K_, dim=-1)
            outputs = tf.matmul(Q_cos, K_cos, transpose_b=True)  # (h*N, T_q, T_k)
            # Scale
            outputs = outputs * 20
        elif atten_mode == 'ln':
            # Layer Norm
            Q_ = layers.layer_norm(Q_, begin_norm_axis=-1, begin_params_axis=-1)
            K_ = layers.layer_norm(K_, begin_norm_axis=-1, begin_params_axis=-1)
            # Multiplication
            outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
            # Scale
            outputs = outputs * (K_.get_shape().as_list()[-1] ** (-0.5))
        elif atten_mode == 'din':
            Q_ = tf.tile(tf.expand_dims(Q_, axis=2), [1, 1, key_len, 1])
            K_ = tf.tile(tf.expand_dims(K_, axis=1), [1, query_len, 1, 1])
            din_layer = tf.concat([Q_ - K_, Q_ * K_], axis=-1)

            for i, din_units in enumerate(din_layer_units):
                din_layer = layers.fully_connected(din_layer, din_units, activation_fn=get_act_fn(din_activation_fn),
                                                   scope='din_{}_layer'.format(i))
            din_layer = layers.fully_connected(din_layer, 1, activation_fn=None, scope='din_logits')
            outputs = tf.reshape(din_layer, [-1, query_len, key_len])
        else:
            # Multiplication
            outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
            # Scale
            outputs = outputs * (K_.get_shape().as_list()[-1] ** (-0.5))

        # key Masking
        key_masks = tf.tile(tf.reshape(key_masks, [-1, 1, key_len]), [num_heads, query_len, 1])  # (h*N, T_q, T_k)
        min_paddings = tf.fill(tf.shape(outputs), tf.constant(-2 ** 32 + 1, dtype=tf.float32))

        self_mask = tf.tile(tf.expand_dims(tf.eye(query_len, key_len), axis=0), [tf.shape(key_masks)[0], 1, 1])
        zero_paddings = tf.fill(tf.shape(outputs), tf.constant(0, dtype=tf.float32))
        self_mask = tf.where(key_masks, self_mask, zero_paddings)

        if atten_activation_fn is None:
            outputs = tf.where(key_masks, outputs, zero_paddings)
        else:
            if is_self_mask:
                key_masks = tf.logical_and(key_masks, tf.cast((1 - self_mask), dtype=tf.bool))
            outputs = tf.where(key_masks, outputs, min_paddings)
            outputs = get_act_fn(atten_activation_fn)(outputs)
            if is_self_mask:
                outputs += self_mask

        if not is_target_attention:
            # Query Masking
            query_masks = tf.tile(tf.reshape(query_masks, [-1, query_len]), [num_heads, 1])  # (h*N, T_q)
            outputs = tf.reshape(outputs, [-1, key_len])  # (h*N*T_q, T_k)
            paddings = tf.zeros_like(outputs, dtype=tf.float32)  # (h*N*T_q, T_k)
            outputs = tf.where(tf.reshape(query_masks, [-1]), outputs,
                               paddings)  # tf.where((h*N*T_q), (h*N*T_q, T_k), (h*N*T_q, T_k)) => (h*N*T_q, T_k)
            outputs = tf.reshape(outputs, [-1, query_len, key_len])  # (h*N, T_q, T_k)

        # Attention vector
        att_vec = outputs

        # Weighted sum (h*N, T_q, T_k) * (h*N, T_k, C'/h)
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C'/h)

        # Restore shape
        if num_heads > 1:
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C')
        
    return outputs, att_vec