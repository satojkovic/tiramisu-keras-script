#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf


class FullyConvNN():
    def __init__(self):
        """
        Build the model.
        """
        self.build()

    def build(self, x):
        """
        Build the FCN.
        """
        print('Building the FCN..')

        # conv1 layer
        self.conv1_1 = tf.layers.conv2d(x, 64, [3, 3], activation=tf.nn.relu)
        self.conv1_2 = tf.layers.conv2d(
            self.conv1_1, 64, [3, 3], activation=tf.nn.relu)
        self.pool1 = tf.layers.max_pooling2d(self.conv1_2, [2, 2], strides=2)
        # conv2 layer
        self.conv2_1 = tf.layers.conv2d(
            self.pool1, 128, [3, 3], activation=tf.nn.relu)
        self.conv2_2 = tf.layers.conv2d(
            self.conv2_1, 128, [3, 3], activation=tf.nn.relu)
        self.pool2 = tf.layers.max_pooling2d(self.conv2_2, [2, 2], strides=2)
        # conv3 layer
        self.conv3_1 = tf.layers.conv2d(
            self.pool2, 256, [3, 3], activation=tf.nn.relu)
        self.conv3_2 = tf.layers.conv2d(
            self.conv3_1, 256, [3, 3], activation=tf.nn.relu)
        self.pool3 = tf.layers.max_pooling2d(self.conv3_2, [2, 2], strides=2)
        # conv4 layer
        self.conv4_1 = tf.layers.conv2d(
            self.pool3, 512, [3, 3], activation=tf.nn.relu)
        self.conv4_2 = tf.layers.conv2d(
            self.conv4_1, 512, [3, 3], activation=tf.nn.relu)
        self.pool4 = tf.layers.max_pooling2d(self.conv4_2, [2, 2], strides=2)
        # conv5 layer
        self.conv5_1 = tf.layers.conv2d(
            self.pool4, 512, [3, 3], activation=tf.nn.relu)
        self.conv5_2 = tf.layers.conv2d(
            self.conv5_1, 512, [3, 3], activation=tf.nn.relu)
        self.pool5 = tf.layers.max_pooling2d(self.conv5_2, [2, 2], strides=2)
        # fc1 layer
        self.fc1 = tf.layers.dense(self.pool5, 4096)
        # fc2 layer
        self.fc2 = tf.layers.dense(self.fc1, 4096)
