from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.ops import variable_scope
from model_ops.mimn import MIMNCell, din_attention

def _create_mimn_network(self, long_seq_input_layer, item_eb, key_masks, units=16):
        # seq prep
        seq_layer = long_seq_input_layer
        key_masks = tf.to_float(tf.squeeze(key_masks, axis=1))  # (512, 1000)
        item_eb = tf.squeeze(item_eb, axis=1)

        # hyper parameter
        BATCH_SIZE = self.FLAGS.batch_size
        MEMORY_SIZE = 4
        EMBEDDING_DIM = seq_layer.shape.as_list()[-1]
        HIDDEN_SIZE = EMBEDDING_DIM
        Mem_Induction = 0
        Util_Reg = 0
        SEQ_LEN = seq_layer.shape.as_list()[-2]
        mask_flag = True

        def clear_mask_state(state, begin_state, begin_channel_rnn_state, mask, cell, t):
            state["controller_state"] = ((1 - tf.reshape(mask[:, t], (-1, 1))) * begin_state["controller_state"]
                                         + tf.reshape(mask[:, t], (-1, 1)) * state["controller_state"])
            state["M"] = (1 - tf.reshape(mask[:, t], (-1, 1, 1))) * begin_state["M"] + tf.reshape(mask[:, t], (-1, 1, 1)) * state["M"]
            state["key_M"] = ((1 - tf.reshape(mask[:, t], (-1, 1, 1))) * begin_state["key_M"]
                              + tf.reshape(mask[:, t], (-1, 1, 1)) * state["key_M"])
            state["sum_aggre"] = ((1 - tf.reshape(mask[:, t], (-1, 1, 1))) * begin_state["sum_aggre"]
                                  + tf.reshape(mask[:, t], (-1, 1, 1)) * state["sum_aggre"])
            if Mem_Induction > 0:
                temp_channel_rnn_state = []
                for i in range(MEMORY_SIZE):
                    temp_channel_rnn_state.append(cell.channel_rnn_state[i] * tf.expand_dims(mask[:, t], axis=1)
                                                  + begin_channel_rnn_state[i] * (1 - tf.expand_dims(mask[:, t], axis=1)))
                cell.channel_rnn_state = temp_channel_rnn_state
                temp_channel_rnn_output = []
                for i in range(MEMORY_SIZE):
                    temp_output = (cell.channel_rnn_output[i] * tf.expand_dims(mask[:, t], axis=1)
                                   + begin_channel_rnn_output[i] * (1 - tf.expand_dims(key_masks[:, t], axis=1)))
                    temp_channel_rnn_output.append(temp_output)
                cell.channel_rnn_output = temp_channel_rnn_output
            return state

        cell = MIMNCell(controller_units=HIDDEN_SIZE, memory_size=MEMORY_SIZE, memory_vector_dim=EMBEDDING_DIM, read_head_num=1, write_head_num=1,
                        reuse=False, output_dim=HIDDEN_SIZE, clip_value=20, batch_size=BATCH_SIZE, mem_induction=Mem_Induction, util_reg=Util_Reg)

        state = cell.zero_state(BATCH_SIZE, tf.float32)
        if Mem_Induction > 0:
            begin_channel_rnn_output = cell.channel_rnn_output
        else:
            begin_channel_rnn_output = 0.0

        begin_state = state
        self.state_list = [state]
        self.mimn_o = []
        for t in range(SEQ_LEN):
            output, state, temp_output_list = cell(seq_layer[:, t, :], state)
            if mask_flag:
                state = clear_mask_state(state, begin_state, begin_channel_rnn_output, key_masks, cell, t)
            self.mimn_o.append(output)
            self.state_list.append(state)

        self.mimn_o = tf.stack(self.mimn_o, axis=1)
        self.state_list.append(state)
        mean_memory = tf.reduce_mean(state['sum_aggre'], axis=-2)

        read_out, _, _ = cell(item_eb, state)

        inp = tf.concat([item_eb, tf.reduce_sum(seq_layer, axis=1), read_out, mean_memory * item_eb], 1)
        return inp