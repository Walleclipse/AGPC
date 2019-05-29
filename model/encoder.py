from tensorflow.contrib.rnn import LSTMStateTuple

import tensorflow as tf
from .cell import create_rnn_cell


def build_encoder(embeddings, encoder_inputs, encoder_length,
                  num_layers, num_units, cell_type,
                  bidir=False, dtype=tf.float32,name='encoder'):
    """
    encoder: build rnn encoder for Seq2seq
        source_ids: [batch_size, max_time]
        bidir: bidirectional or unidirectional

    Returns:
        encoder_outputs: [batch_size, max_time, num_units]
        encoder_states: (StateTuple(shape=(batch_size, num_units)), ...)
    """
    # embedding lookup, embed_inputs: [max_time, batch_size, num_units]
    embed_inputs = tf.nn.embedding_lookup(embeddings, encoder_inputs)

    # bidirectional
    if bidir:
        encoder_states = []
        layer_inputs = embed_inputs

        # build rnn layer-by-layer
        for i in range(num_layers):
            with tf.variable_scope(name+"_layer_%d" % (i + 1)):
                fw_cell = create_rnn_cell(
                    1, num_units, cell_type)
                bw_cell = create_rnn_cell(
                    1, num_units, cell_type)

                dyn_rnn = tf.nn.bidirectional_dynamic_rnn(
                    fw_cell, bw_cell, layer_inputs,
                    sequence_length=encoder_length,
                    dtype=dtype,
                    swap_memory=True)

                bi_outputs, (state_fw, state_bw) = dyn_rnn
                # handle cell memory state
                if cell_type == "LSTM":
                    state_c = state_fw.c + state_bw.c
                    state_h = state_fw.h + state_bw.h
                    encoder_states.append(LSTMStateTuple(state_c, state_h))
                else:
                    encoder_states.append(state_fw + state_bw)
                # concat and map as inputs of next layer
                layer_inputs = tf.layers.dense(
                    tf.concat(bi_outputs, -1), num_units)

        encoder_outputs = layer_inputs
        encoder_states = tuple(encoder_states)

    # unidirectional
    else:
        rnn_cell = create_rnn_cell(num_layers, num_units, cell_type)

        encoder_outputs, encoder_states = tf.nn.dynamic_rnn(
            rnn_cell, embed_inputs,
            sequence_length=encoder_length,
            dtype=dtype,
            swap_memory=True)

    return encoder_outputs, encoder_states
