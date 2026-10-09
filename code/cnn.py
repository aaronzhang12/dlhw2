from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data, get_next_batch
from manual_convolution import ManualConv2d
from base_model import CifarModel

import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class CNN(CifarModel):
    def __init__(self, classes):
        """
        This model class will contain the architecture for your CNN that
        classifies images. Do not modify the constructor, as doing so
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(CNN, self).__init__()

        # Initialize all hyperparameters
        self.loss_list = []
        self.batch_size = 64
        # make sure to change these
        self.input_width = 32
        self.input_height = 32
        self.image_channels = 3
        self.num_classes = len(classes)

        self.hidden_layer_size = 320
        self.dropout_rate = 0.1
        self.pool_size = 2

        self.epsilon = 1e-3  # this is used for batch normalization only!
        self.conv_1 = tf.keras.layers.Conv2D(
            filters = 32,
            kernel_size = (3,3),
            strides=(1, 1),
            padding="same",
            activation=None,
        )
        self.conv_2 = tf.keras.layers.Conv2D(
            filters = 64, 
            kernel_size = (3,3), 
            strides=(1, 1), 
            padding="same", 
            activation=None, 
        )
        self.conv_3 = tf.keras.layers.Conv2D(
            filters = 64, 
            kernel_size = (3,3), 
            strides=(1, 1), 
            padding="same", 
            activation=None, 
        )
        # manual layer for testing
        self.manual_conv_3 = ManualConv2d(
            filter_shape = [3,3,64,64],
            padding = "SAME",
            trainable = False
        )
        self.batch_norm_1 = tf.keras.layers.BatchNormalization(epsilon=self.epsilon)
        self.batch_norm_2 = tf.keras.layers.BatchNormalization(epsilon=self.epsilon)
        self.batch_norm_3 = tf.keras.layers.BatchNormalization(epsilon=self.epsilon)
        self.dense_1 = tf.keras.layers.Dense(units = self.hidden_layer_size, activation = "relu")
        self.dense_2 = tf.keras.layers.Dense(units = self.hidden_layer_size, activation = "relu")
        self.dense_3 = tf.keras.layers.Dense(units = self.num_classes)


    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, num_classes)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)

        if not is_testing:
            inputs = tf.image.random_flip_left_right(inputs)
        x = self.conv_1(inputs)
        x = self.batch_norm_1(x, training=not is_testing)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=self.pool_size, strides=self.pool_size, padding="SAME")
        x = self.conv_2(x)
        x = self.batch_norm_2(x, training=not is_testing)
        x = tf.nn.relu(x)
        x = tf.nn.max_pool(x, ksize=self.pool_size, strides=self.pool_size, padding="SAME")
        # keep 3rd base convolution layer if training
        if not is_testing:
            x = self.conv_3(x)
        # otherwise use manual convolution layer
        else:
            self.manual_conv_3.set_weights(
                self.conv_3.kernel, self.conv_3.bias
            )
            x = self.manual_conv_3(x)
        x = self.batch_norm_3(x, training=not is_testing)
        x = tf.nn.relu(x)
        x = tf.reshape(x, [-1, x.shape[1]*x.shape[2]*x.shape[3]])
        x = self.dense_1(x)
        if not is_testing:
            x = tf.nn.dropout(x, rate=self.dropout_rate)
        x = self.dense_2(x)
        if not is_testing:
            x = tf.nn.dropout(x, rate=self.dropout_rate)
        return self.dense_3(x)

        
