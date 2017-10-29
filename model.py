#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf

N_CL = 14


class FullyConvNN():
    def __init__(self, input):
        """
        Build the model.
        """
        self.x = input

    def build(self):
        """
        Build the FCN.
        """
        # conv1 layer
        self.x = tf.pad(self.x, [[0, 0], [100, 100], [100, 100], [0, 0]],
                        'CONSTANT')  # custom padding for convolution1_1
        self.conv1_1 = tf.layers.conv2d(
            self.x, 64, [3, 3], activation=tf.nn.relu)
        self.conv1_2 = tf.layers.conv2d(
            self.conv1_1, 64, [3, 3], padding='SAME', activation=tf.nn.relu)
        self.pool1 = tf.layers.max_pooling2d(self.conv1_2, [2, 2], strides=2)
        # conv2 layer
        self.conv2_1 = tf.layers.conv2d(
            self.pool1, 128, [3, 3], padding='SAME', activation=tf.nn.relu)
        self.conv2_2 = tf.layers.conv2d(
            self.conv2_1, 128, [3, 3], padding='SAME', activation=tf.nn.relu)
        self.conv2_2 = tf.pad(self.conv2_2, [[0, 0], [0, 1], [0, 1], [0, 0]],
                              'CONSTANT')  # add padding
        self.pool2 = tf.layers.max_pooling2d(self.conv2_2, [2, 2], strides=2)
        # conv3 layer
        self.conv3_1 = tf.layers.conv2d(
            self.pool2, 256, [3, 3], padding='SAME', activation=tf.nn.relu)
        self.conv3_2 = tf.layers.conv2d(
            self.conv3_1, 256, [3, 3], padding='SAME', activation=tf.nn.relu)
        self.conv3_3 = tf.layers.conv2d(
            self.conv3_2, 256, [3, 3], padding='SAME', activation=tf.nn.relu)
        self.conv3_3 = tf.pad(self.conv3_3, [[0, 0], [0, 1], [0, 1], [0, 0]],
                              'CONSTANT')  # add padding
        self.pool3 = tf.layers.max_pooling2d(self.conv3_3, [2, 2], strides=2)
        # conv4 layer
        self.conv4_1 = tf.layers.conv2d(
            self.pool3, 512, [3, 3], padding='SAME', activation=tf.nn.relu)
        self.conv4_2 = tf.layers.conv2d(
            self.conv4_1, 512, [3, 3], padding='SAME', activation=tf.nn.relu)
        self.conv4_3 = tf.layers.conv2d(
            self.conv4_2, 512, [3, 3], padding='SAME', activation=tf.nn.relu)
        self.conv4_3 = tf.pad(self.conv4_3, [[0, 0], [0, 1], [0, 1], [0, 0]],
                              'CONSTANT')  # add padding
        self.pool4 = tf.layers.max_pooling2d(self.conv4_3, [2, 2], strides=2)
        # conv5 layer
        self.conv5_1 = tf.layers.conv2d(
            self.pool4, 512, [3, 3], padding='SAME', activation=tf.nn.relu)
        self.conv5_2 = tf.layers.conv2d(
            self.conv5_1, 512, [3, 3], padding='SAME', activation=tf.nn.relu)
        self.conv5_3 = tf.layers.conv2d(
            self.conv5_2, 512, [3, 3], padding='SAME', activation=tf.nn.relu)
        self.pool5 = tf.layers.max_pooling2d(self.conv5_3, [2, 2], strides=2)
        # fc_conv6 layer
        self.fc_conv6 = tf.layers.conv2d(
            self.pool5, 4096, [7, 7], activation=tf.nn.relu)
        # fc_conv7 layer
        self.fc_conv7 = tf.layers.conv2d(
            self.fc_conv6, 4096, [1, 1], activation=tf.nn.relu)
        # upconvolution layer1
        self.upconv1 = tf.layers.conv2d_transpose(
            self.fc_conv7, N_CL, [4, 4], strides=2, activation=tf.nn.relu)
        # upconvolution layer2
        self.upconv2 = tf.layers.conv2d_transpose(
            self.upconv1, N_CL, [4, 4], strides=2, activation=tf.nn.relu)
        # upconvolution layer3
        self.upconv3 = tf.layers.conv2d_transpose(
            self.upconv2, N_CL, [4, 4], strides=2, activation=tf.nn.relu)
        # upconvolution layer4
        self.upconv4 = tf.layers.conv2d_transpose(
            self.upconv3, N_CL, [4, 4], strides=2, activation=tf.nn.relu)
        # upconvolution layer5
        self.upconv5 = tf.layers.conv2d_transpose(
            self.upconv4, N_CL, [4, 4], strides=2, activation=tf.nn.relu)
        # output
        self.output = tf.image.resize_images(self.upconv5, [300, 300])
        # final output
        return self.output
