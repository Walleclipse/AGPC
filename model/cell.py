# -*- coding: utf-8 -*-
import tensorflow as tf


def single_cell(num_units, cell_type, keep_prob=1.0):
    """
    Cell: build a recurrent cell
        num_units: number of hidden cell units
        cell_type: LSTM, GRU, LN_LSTM (layer_normalize)
    """
    if cell_type == "LSTM":
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

    elif cell_type == "GRU":
        cell = tf.nn.rnn_cell.GRUCell(num_units)

    elif cell_type == "LN_LSTM":
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(
            num_units,
            layer_norm=True)
    else:
        raise ValueError("Unknown cell type %s" % cell_type)
    if keep_prob<1:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

    return cell


def create_rnn_cell(num_layers, num_units, cell_type):
    """
    RNN_cell: build a multi-layer rnn cell
        num_layers: number of hidden layers
    """
    if num_layers > 1:
        rnn_cell = tf.nn.rnn_cell.MultiRNNCell(
            [single_cell(num_units, cell_type) for _ in range(num_layers)]
        )
    else:
        rnn_cell = single_cell(num_units, cell_type)

    return rnn_cell
