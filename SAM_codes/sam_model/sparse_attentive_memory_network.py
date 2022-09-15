from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.ops import variable_scope

from model_ops.attention import memory


def _create_sparse_attentive_memory_network(self,
                                  attention_input_layer_dict,
                                  sequence_input_layer_dict,
                                  sequence_length_layer_dict,
                                  units=16,
                                  dropout_rate=0.0):
        """Input the input sequence with features, and outputs the memory vector.

          Args:
            attention_input_layer_dict : Key-value pair with key the feature name for the target item and the 
            value the Tensor for the target item. 

            sequence_input_layer_dict: Key-value pair with key the feature name for the sequence item and the 
            value the Tensor for the sequence item. 

            sequence_length_layer_dict: Key-value pair with key the feature name for the sequence item and the 
            value the Tensor for sequence length.


          Returns:
            Memory vector.
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
        if self.FLAGS.gru_out:
            long_seq_layer = self._create_memory_network_gru(long_seq_layer, query_layer, long_key_masks, units, 'stk')
        else:
            long_seq_layer = self._create_memory_network(long_seq_layer, query_layer, long_key_masks, units)
        return None, long_seq_layer

def _create_memory_network(self, long_seq_input_layer, query_layer, key_masks, units=16, mode='stk'):
        seq_layer = long_seq_input_layer
        query_dim = query_layer.shape.as_list()

        # initialize to query layer
        mem_cell = tf.nn.rnn_cell.GRUCell(num_units=query_dim[2])
        h0 = mem_cell.zero_state(self.FLAGS.batch_size, tf.float32)
        query_layer_reshape = tf.reshape(query_layer, [-1, query_dim[2]])  # (?, 32)
        mem_out = query_layer_reshape

        mem_out_expand = tf.reshape(mem_out, [-1, 1, query_dim[2]])

        attention_output_layer = None

        with variable_scope.variable_scope('memory'):
            # (?, T_q, C), (?, T_q, T_k
            for i in range(self.FLAGS.num_mem_pass):
                mem_seq_vec, mem_att_vec = memory(queries=query_layer,
                                                  keys=seq_layer,
                                                  memory=mem_out_expand,
                                                  num_heads=1,
                                                  scope='memory_layer_%d' % i,
                                                  query_masks=None,
                                                  key_masks=key_masks,
                                                  linear_projection=False,
                                                  num_units=query_layer.shape.as_list()[-1],
                                                  num_output_units=seq_layer.shape.as_list()[-1],
                                                  atten_mode='din',
                                                  is_target_attention=True,
                                                  forward_layer_units=[8],
                                                  din_activation_fn='sigmoid',
                                                  atten_activation_fn='sigmoid',
                                                  is_self_mask=False
                                                  )
                mem_att_vec = tf.concat(tf.split(mem_att_vec, 1, axis=0), axis=2)

                mem_shape = mem_att_vec.shape.as_list()
                mem_att_vec = tf.reshape(mem_att_vec, [-1, mem_shape[1] * mem_shape[2]])

                def avg_entropy_per_batch(att_vec):
                    att_vec_sum = tf.add(tf.reduce_sum(att_vec, axis=1), tf.constant(1e-12))
                    att_vec_norm = att_vec / tf.reshape(att_vec_sum, (-1, 1))
                    bool_mask = tf.not_equal(att_vec_norm, 0)
                    att_vec_entropy = -tf.reduce_sum(att_vec_norm * (tf.log(att_vec_norm + 1e-12) / tf.log(2.0)),
                                                     axis=1)
                    avg_entropy_per_batch = tf.reduce_mean(att_vec_entropy)
                    return avg_entropy_per_batch

                s = mem_att_vec.shape.as_list()
            

                mem_seq_vec_reshape = tf.reshape(mem_seq_vec, [-1, query_dim[2]])
                mem_out, h_after = mem_cell(mem_seq_vec_reshape, mem_out)  # mem out is initial state
                if mode == 'iter':
                  mem_out_expand = tf.reshape(mem_out, [-1, 1, query_dim[2]])
                else:
                  memory_out_expand = tf.reshape(mem_out, [-1, 1, query_dim[2]])
                small_mem_att_vec = layers.fully_connected(
                    mem_att_vec,
                    units,
                    activation_fn=get_act_fn(self.dnn_hidden_units_act_op[0])
                )

                if attention_output_layer is None:
                    attention_output_layer = tf.concat([small_mem_att_vec, mem_out], axis=-1)
                else:
                    attention_output_layer = tf.concat([attention_output_layer, small_mem_att_vec, mem_out], axis=-1)
            attention_output_layer = layers.batch_norm(attention_output_layer, scale=True, is_training=self.is_training)

        print('attention_output_layer : %s' % attention_output_layer)
        return attention_output_layer

 def _create_memory_network_gru(self, long_seq_input_layer, query_layer, key_masks, units=16, mode='stk'):
        seq_layer = long_seq_input_layer
        query_dim = query_layer.shape.as_list()

        # initialize to query layer
        mem_cell = tf.nn.rnn_cell.GRUCell(num_units=query_dim[2])
        h0 = mem_cell.zero_state(self.FLAGS.batch_size, tf.float32)
        query_layer_reshape = tf.reshape(query_layer, [-1, query_dim[2]])  # (?, 32)
        mem_out = query_layer_reshape

        mem_out_expand = tf.reshape(mem_out, [-1, 1, query_dim[2]])
        attention_output_layer = None

        with variable_scope.variable_scope('memory'):
            # (?, T_q, C), (?, T_q, T_k
            for i in range(self.FLAGS.num_mem_pass):
                mem_seq_vec, mem_att_vec = memory(queries=query_layer,
                                                  keys=seq_layer,
                                                  memory=mem_out_expand,
                                                  num_heads=1,
                                                  scope='memory_layer_%d' % i,
                                                  query_masks=None,
                                                  key_masks=key_masks,
                                                  linear_projection=False,
                                                  num_units=query_layer.shape.as_list()[-1],
                                                  num_output_units=seq_layer.shape.as_list()[-1],
                                                  atten_mode='din',
                                                  is_target_attention=True,
                                                  forward_layer_units=[8],
                                                  din_activation_fn='sigmoid',
                                                  atten_activation_fn='sigmoid',
                                                  is_self_mask=False
                                                  )
                mem_att_vec = tf.concat(tf.split(mem_att_vec, 1, axis=0), axis=2)
            
                mem_shape = mem_att_vec.shape.as_list()
                mem_att_vec = tf.reshape(mem_att_vec, [-1, mem_shape[1] * mem_shape[2]])

                def avg_entropy_per_batch(att_vec):
                    att_vec_sum = tf.add(tf.reduce_sum(att_vec, axis=1), tf.constant(1e-12))
                    att_vec_norm = att_vec / tf.reshape(att_vec_sum, (-1, 1))
                    bool_mask = tf.not_equal(att_vec_norm, 0)
                    att_vec_entropy = -tf.reduce_sum(att_vec_norm * (tf.log(att_vec_norm + 1e-12) / tf.log(2.0)),
                                                     axis=1)
                    avg_entropy_per_batch = tf.reduce_mean(att_vec_entropy)
                    return avg_entropy_per_batch

                s = mem_att_vec.shape.as_list()

                mem_seq_vec_reshape = tf.reshape(mem_seq_vec, [-1, query_dim[2]])
                mem_out, h_after = mem_cell(mem_seq_vec_reshape, mem_out)  # mem out is initial state
                if mode == 'iter':
                  mem_out_expand = tf.reshape(mem_out, [-1, 1, query_dim[2]])
                else:
                  memory_out_expand = tf.reshape(mem_out, [-1, 1, query_dim[2]])
                
                small_mem_att_vec = layers.fully_connected(
                    mem_att_vec,
                    units,
                    activation_fn=get_act_fn(self.dnn_hidden_units_act_op[0])
                )

            with variable_scope.variable_scope('output'):
                output_cell = tf.nn.rnn_cell.GRUCell(num_units=query_dim[2])
                state = mem_out
                for i in range(3):
                    trans_state = layers.fully_connected(state, query_dim[2], activation_fn=None)
                    concat_layer = tf.concat([trans_state, query_layer_reshape], axis=-1)
                    output, state = output_cell(concat_layer, state)
                attention_output_layer = tf.concat([small_mem_att_vec, output], axis=-1)

        return attention_output_layer
