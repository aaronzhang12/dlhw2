from __future__ import absolute_import
from preprocess import unpickle, get_next_batch, get_data

import os
import tensorflow as tf
import numpy as np
import random
import math


class ManualConv2d(tf.keras.layers.Layer):
    def __init__(self, filter_shape: list[int], strides: list[int]=[1,1,1,1], padding = "VALID", use_bias = True, trainable=True, *args, **kwargs):
        """
        :param filter_shape: list of [filter_height, filter_width, in_channels, out_channels]
        :param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
        :param padding: either "SAME" or "VALID", capitalization matters
        """
        super().__init__()

        self.strides = strides
        self.padding = padding

        def get_var(name, shape, trainable):
            return tf.Variable(tf.random.truncated_normal(shape, dtype=tf.float32, stddev=1e-1), name=name, trainable = trainable)

        self.filters = get_var("conv_filters", filter_shape, trainable)
        self.use_bias = use_bias
        if use_bias: self.bias = get_var("conv_bias", [filter_shape[-1]], trainable)
        else: self.bias = None

    def get_weights(self):
        if self.bias is not None: return self.filters, self.bias
        return self.filters

    def set_weights(self, filters, bias=None): 
        self.filters = filters
        if bias is not None: self.bias = bias

    def call(self, inputs):
        """
        :param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
        """

        #define some useful variables
        num_examples, in_height, in_width, input_in_channels = inputs.shape
        filter_height, filter_width, filter_in_channels, filter_out_channels = self.filters.shape
        assert input_in_channels == filter_in_channels

        if self.padding == "VALID":
            pad_h_1 = 0
            pad_w_1 = 0
            pad_h_2 = 0
            pad_w_2 = 0
        elif self.padding == "SAME":
            # pad sizes depend on parity
            pad_h_1 = (filter_height-1)//2
            pad_w_1 = (filter_width-1)//2
            pad_h_2 = filter_height - pad_h_1-1
            pad_w_2= filter_width-pad_w_1-1

        # calculate output image dimensions
        out_height = in_height + pad_h_1 + pad_h_2 -filter_height + 1
        out_width = in_width + pad_w_1 + pad_w_2 -filter_width + 1

        # pad
        padded = tf.pad(
            inputs,
            ((0,0), (pad_h_1, pad_h_2), (pad_w_1, pad_w_2), (0, 0)),
            mode = "constant",
            constant_values=0
        )

        rows = []
        # iterate through the output image dimensions
        for i in range(out_height):
            columns = []
            for j in range(out_width):
                window = padded[:, i:i+filter_height, j:j+filter_width, :]
                # broadcast, find (num_examples, filter_out_channel) tensor at each pixel
                values = tf.reduce_sum(
                        self.filters[None, ...] * window[..., None],
                        axis=(1, 2, 3)
                    )
                if self.use_bias:
                    values = values + self.bias
                columns.append(values)
            # stack the values at each pixel together
            row = tf.stack(columns, axis = 1)
            rows.append(row)
        conv_images = tf.stack(rows, axis = 1)
        return tf.convert_to_tensor(conv_images, dtype = tf.float32)