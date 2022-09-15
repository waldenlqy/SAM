import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.rnn_cell import GRUCell
from model_ops.mimn import din_attention
from model_ops.rnn import dynamic_rnn, static_rnn, static_rnn_t, VecAttGRUCell
from model_ops.mimn import din_attention


def _create_dien_network(self, long_seq_input_layer, item_eb, key_masks, units=16):
        # seq prep
        seq_layer = long_seq_input_layer
        key_masks = tf.to_float(tf.squeeze(key_masks, axis=1))  # (512, 1000)
        item_eb = tf.squeeze(item_eb, axis=1)

        # hyper parameter
        BATCH_SIZE = self.FLAGS.batch_size
        EMBEDDING_DIM = seq_layer.shape.as_list()[-1]
        HIDDEN_SIZE = EMBEDDING_DIM
        SEQ_LEN = seq_layer.shape.as_list()[-2]
        seq_layer = seq_layer[:, :SEQ_LEN, :]
        key_masks = key_masks[:, :SEQ_LEN]

        k_seq_layer = tf.transpose(seq_layer, perm=[1, 0, 2])
        t_seq_layer = []
        for i in range(SEQ_LEN):
            t_seq_layer.append(k_seq_layer[i])

        with tf.name_scope('rnn_1'):
            self.sequence_length = tf.Variable([SEQ_LEN] * BATCH_SIZE)
            rnn_outputs, _ = static_rnn(GRUCell(EMBEDDING_DIM), inputs=t_seq_layer,
                                        sequence_length=self.sequence_length, dtype=tf.float32,
                                        scope="gru1")
            tf.summary.histogram('GRU_outputs', rnn_outputs)

        # Attention layer
        with tf.name_scope('Attention_layer_1'):
            att_outputs, alphas = din_attention(item_eb, tf.convert_to_tensor(rnn_outputs), HIDDEN_SIZE, mask=key_masks, mode="LIST", return_alphas=True, time_major=True)
            tf.summary.histogram('alpha_outputs', alphas)

        with tf.name_scope('rnn_2'):
            rnn_outputs2, final_state2 = static_rnn_t(VecAttGRUCell(HIDDEN_SIZE), inputs=rnn_outputs,
                                                      att_scores=tf.expand_dims(alphas, -1),
                                                      sequence_length=self.sequence_length, dtype=tf.float32,
                                                      scope="gru2")
            tf.summary.histogram('GRU2_Final_State', final_state2)

        inp = tf.concat([item_eb, final_state2, tf.reduce_sum(seq_layer, axis=1), item_eb * tf.reduce_sum(seq_layer, axis=1)], 1)
        return inp