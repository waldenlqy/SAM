from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.ops import variable_scope


def _create_pooling_network(self,
                                attention_input_layer_dict,
                                sequence_input_layer_dict,
                                sequence_length_layer_dict,
                                units=16,
                                dropout_rate=0.0):
        """Input the input sequence with features, and use average pooling to output the user interest vector.

          Args:
            attention_input_layer_dict : Key-value pair with key the feature name for the target item and the 
            value the Tensor for the target item. 

            sequence_input_layer_dict: Key-value pair with key the feature name for the sequence item and the 
            value the Tensor for the sequence item. 

            sequence_length_layer_dict: Key-value pair with key the feature name for the sequence item and the 
            value the Tensor for sequence length.


          Returns:
            User interest vector.
        """

        seq_layer_list, query_layer_list, seq_length_list, key_masks = [], [], [], None
        for name, query_layers in attention_input_layer_dict.items():
            seq_layer = sequence_input_layer_dict[name]
            seq_length = sequence_length_layer_dict.get(name, None)
            if seq_length is None:
                raise Exception('keys {} length is None'.format(name))
            query_layer = tf.concat([] + query_layers, axis=1)
            if key_masks is None:
                key_len = seq_layer.get_shape().as_list()[1]
                key_masks = tf.sequence_mask(seq_length, key_len) 

            seq_layer_list.append(seq_layer)
            query_layer_list.append(query_layer)

        seq_layer = tf.concat(seq_layer_list, axis=-1)

        s = seq_layer.shape.as_list()

        k = tf.cast(key_masks, tf.float32)
        num_true = tf.reduce_sum(k, axis=2)
        num_true_no_nan = tf.where(tf.equal(num_true, 0), tf.ones_like(num_true), num_true)
        k = tf.tile(k, [1, s[2], 1])
        k = tf.transpose(k, [0, 2, 1])
        total = tf.reduce_sum(seq_layer * k, axis=1)
        seq_layer = total / num_true_no_nan

        cross_att_vec = seq_layer

        return cross_att_vec