from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.ops import variable_scope
from model_ops.attention import feedforward
from model_ops.attention import multihead_mask_attention

def _create_self_network(self, long_seq_input_layer, query_layer, key_masks, SEQ_LEN, units=16):
        """Input the sequence, the target item, and outputs the Transformer-encoded sequence.

          Args:
            long_seq_input_layer : sequence embedding with size (batch_size, sequence_length, model dimension)
            query_layer : target item embedding embedding with size (batch_size, 1, model dimension)
            key_masks : boolean tensor derived from tf.sequence_mask(seq_length, max_seq_length) if the current sequence length
                        does not exceed max_seq_length

          Returns:
            Transformer-encoded user interest vector.
        """
        long_seq_input_layer = long_seq_input_layer[:, :SEQ_LEN, :]
        key_masks = key_masks[:, :, :SEQ_LEN]

        seq_layer = long_seq_input_layer

        seq_shape = seq_layer.shape.as_list()
        # self attention
        with variable_scope.variable_scope('lseq-self'):
            # layer norm
            seq_layer_norm = layers.layer_norm(seq_layer)
            # self-attention layer
            seq_vec, att_vec = multihead_mask_attention(queries=seq_layer_norm,
                                                        keys=seq_layer_norm,
                                                        num_heads=1,
                                                        query_masks=key_masks,
                                                        key_masks=key_masks,
                                                        linear_projection=True,  # as in SASRec
                                                        num_units=seq_layer.shape.as_list()[-1],
                                                        num_output_units=seq_layer.shape.as_list()[-1],
                                                        atten_mode='base',
                                                        is_target_attention=False,
                                                        first_n_att_weight_report=0,
                                                        din_layer_units=[8],
                                                        activation_fn='relu',
                                                        din_activation_fn='sigmoid',
                                                        atten_activation_fn='softmax',
                                                        is_self_mask=False
                                                        )
            seq_vec += seq_layer
            seq_vec_norm = layers.layer_norm(seq_vec)
            seq_vec_ffn = feedforward(seq_vec_norm, num_units=[seq_layer.shape.as_list()[-1],
                                                               seq_layer.shape.as_list()[-1]],
                                      activation_fn='relu',
                                      scope="feed_forward",
                                      reuse=tf.AUTO_REUSE,
                                      variables_collections=None,
                                      outputs_collections=None,
                                      is_training=self.is_training)
            seq_vec_ffn += seq_vec
            # flatten
            self_att_vec = tf.reshape(self_vec_ffn, [-1, seq_shape[1] * seq_shape[2]])

        return self_att_vec


def _create_attention_network(self,
                                  attention_input_layer_dict,
                                  sequence_input_layer_dict,
                                  sequence_length_layer_dict,
                                  modul_name,
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
            print("query_layer-%s" % query_layer)

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

        long_seq_layer = self._create_self_network(long_seq_layer, query_layer, long_key_masks, 100, units)

        return None, long_seq_layer


