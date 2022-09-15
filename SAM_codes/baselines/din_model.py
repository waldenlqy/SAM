from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.ops import variable_scope
from model_ops.attention import multihead_mask_attention
from utils.util import get_act_fn


def _create_attention_network(self,
                                  attention_input_layer_dict,
                                  sequence_input_layer_dict,
                                  sequence_length_layer_dict,
                                  units=16,
                                  dropout_rate=0.0):
        """Input the input sequence with features, and outputs the user interest vector.

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

        seq_layer_list, query_layer_list, seq_length_list, long_seq_layer_list, long_seq_length_list, long_query_layer_list, long_key_masks = [], [], [], [], [], [], None

        for name, query_layers in attention_input_layer_dict.items():
            seq_layer = sequence_input_layer_dict[name]
            seq_length = sequence_length_layer_dict.get(name, None)


            if seq_length is None:
                raise Exception('keys {} length is None'.format(name))

            query_layer = tf.concat([] + query_layers, axis=1)
            long_key_len = seq_layer.get_shape().as_list()[1]
            long_key_masks = tf.sequence_mask(seq_length, long_key_len)

            seq_layer_list.append(tf.expand_dims(seq_layer, axis=1))
            seq_length_list.append(tf.expand_dims(seq_length, axis=1))
            query_layer_list.append(tf.expand_dims(query_layer, axis=1))

        seq_layer = tf.concat(seq_layer_list, axis=1)
        seq_length = tf.concat(seq_length_list, axis=1)
        query_layer = tf.concat(query_layer_list, axis=1)
        long_seq_layer = seq_layer
        long_seq_layer = tf.squeeze(long_seq_layer, [1])
        query_layer = tf.squeeze(query_layer, [1])

        long_seq_layer = self._create_target_attn_network(long_seq_layer, query_layer, long_key_masks, units)

        return None, long_seq_layer

def _create_target_attn_network(self, long_seq_input_layer, query_layer, key_masks, units=16):
        seq_layer = long_seq_input_layer
        query_dim = query_layer.shape.as_list()
    
        with variable_scope.variable_scope('target_attn'):
            mem_seq_vec, mem_att_vec = multihead_mask_attention(queries=query_layer,
                                                        keys=seq_layer,
                                                        num_heads=1,
                                                        query_masks=None,
                                                        key_masks=key_masks,
                                                        linear_projection=False,
                                                        num_units=query_layer.shape.as_list()[-1],
                                                        num_output_units=seq_layer.shape.as_list()[-1],
                                                        atten_mode='din',
                                                        is_target_attention=True,
                                                        din_layer_units=[8],
                                                        din_activation_fn='sigmoid',
                                                        atten_activation_fn='sigmoid',
                                                        is_self_mask=False
                                                        )
            mem_att_vec = tf.concat(tf.split(mem_att_vec, 1, axis=0), axis=2)
        
            small_mem_att_vec = layers.fully_connected(
                        mem_att_vec,
                        units,
                        activation_fn=get_act_fn(self.dnn_hidden_units_act_op[0])
                    ) 
            cross_att_vec = tf.concat([mem_seq_vec, small_mem_att_vec], axis=-1)
            shape = cross_att_vec.shape.as_list()
            cross_att_vec = tf.reshape(cross_att_vec, [-1, shape[1] * shape[2]])
        return cross_att_vec