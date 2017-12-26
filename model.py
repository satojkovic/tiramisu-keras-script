# The MIT License (MIT)
# Copyright (c) 2016 satojkovic

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#

import tensorflow as tf

KERNEL_SIZE3 = 3
KERNEL_SIZE7 = 7
KERNEL_SIZE1 = 1


def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    return tf.constant(0.1, shape=shape)


def params():
    # Weights and biases
    params = {}

    # Layer1
    params['conv1_1'] = weight_variable([KERNEL_SIZE3, KERNEL_SIZE3, 3, 64])
    params['bconv1_1'] = bias_variable([64])
    params['conv1_2'] = weight_variable([KERNEL_SIZE3, KERNEL_SIZE3, 64, 64])
    params['bconv1_2'] = bias_variable([64])

    # Layer2
    params['conv2_1'] = weight_variable([KERNEL_SIZE3, KERNEL_SIZE3, 64, 128])
    params['bconv2_1'] = bias_variable([128])
    params['conv2_2'] = weight_variable([KERNEL_SIZE3, KERNEL_SIZE3, 128, 128])
    params['bconv2_2'] = bias_variable([128])

    # Layer3
    params['conv3_1'] = weight_variable([KERNEL_SIZE3, KERNEL_SIZE3, 128, 256])
    params['bconv3_1'] = bias_variable([256])
    params['conv3_2'] = weight_variable([KERNEL_SIZE3, KERNEL_SIZE3, 256, 256])
    params['bconv3_2'] = bias_variable([256])
    params['conv3_3'] = weight_variable([KERNEL_SIZE3, KERNEL_SIZE3, 256, 256])
    params['bconv3_3'] = bias_variable([256])

    # Layer4
    params['conv4_1'] = weight_variable([KERNEL_SIZE3, KERNEL_SIZE3, 256, 512])
    params['bconv4_1'] = bias_variable([512])
    params['conv4_2'] = weight_variable([KERNEL_SIZE3, KERNEL_SIZE3, 512, 512])
    params['bconv4_2'] = bias_variable([512])
    params['conv4_3'] = weight_variable([KERNEL_SIZE3, KERNEL_SIZE3, 512, 512])
    params['bconv4_3'] = bias_variable([512])

    # Layer5
    params['conv5_1'] = weight_variable([KERNEL_SIZE3, KERNEL_SIZE3, 512, 512])
    params['bconv5_1'] = bias_variable([512])
    params['conv5_2'] = weight_variable([KERNEL_SIZE3, KERNEL_SIZE3, 512, 512])
    params['bconv5_2'] = bias_variable([512])
    params['conv5_3'] = weight_variable([KERNEL_SIZE3, KERNEL_SIZE3, 512, 512])
    params['bconv5_3'] = bias_variable([512])

    # Layer6
    params['conv6'] = weight_variable([KERNEL_SIZE7, KERNEL_SIZE7, 512, 4096])
    params['bconv6'] = bias_variable([4096])

    # Layer7
    params['conv7'] = weight_variable([KERNEL_SIZE1, KERNEL_SIZE1, 4096, 4096])
    params['bconv7'] = bias_variable([4096])

    # score_fr
    params['score_fr'] = weight_variable(
        [KERNEL_SIZE1, KERNEL_SIZE1, 4096, 21])
    params['bscore_fr'] = bias_variable([21])

    # score2
    params['bscore2'] = bias_variable([21])

    # score4
    params['bscore4'] = bias_variable([21])

    # upsample
    params['bupsample'] = bias_variable([21])

    return params


def fcn(x, model_params, keep_prob):
    # Padding
    # rank of 'data' is 4
    PAD100 = tf.constant([[0, 0], [100, 100], [100, 100], [0, 0]])
    # Layer1
    x_pad = tf.pad(x, PAD100, 'CONSTANT')
    conv1_1 = tf.nn.relu(
        tf.nn.conv2d(
            x_pad, model_params['conv1_1'], [1, 1, 1, 1], padding='VALID') +
        model_params['bconv1_1'])
    conv1_2 = tf.nn.relu(
        tf.nn.conv2d(
            conv1_1, model_params['conv1_2'], [1, 1, 1, 1], padding='SAME') +
        model_params['bconv1_2'])
    pool1 = tf.nn.max_pool(
        conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Layer2
    conv2_1 = tf.nn.relu(
        tf.nn.conv2d(
            pool1, model_params['conv2_1'], [1, 1, 1, 1], padding='SAME') +
        model_params['bconv2_1'])
    conv2_2 = tf.nn.relu(
        tf.nn.conv2d(
            conv2_1, model_params['conv2_2'], [1, 1, 1, 1], padding='SAME') +
        model_params['bconv2_2'])
    pool2 = tf.nn.max_pool(
        conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Layer3
    conv3_1 = tf.nn.relu(
        tf.nn.conv2d(
            pool2, model_params['conv3_1'], [1, 1, 1, 1], padding='SAME') +
        model_params['bconv3_1'])
    conv3_2 = tf.nn.relu(
        tf.nn.conv2d(
            conv3_1, model_params['conv3_2'], [1, 1, 1, 1], padding='SAME') +
        model_params['bconv3_2'])
    conv3_3 = tf.nn.relu(
        tf.nn.conv2d(
            conv3_2, model_params['conv3_3'], [1, 1, 1, 1], padding='SAME') +
        model_params['bconv3_3'])
    pool3 = tf.nn.max_pool(
        conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Layer4
    conv4_1 = tf.nn.relu(
        tf.nn.conv2d(
            pool3, model_params['conv4_1'], [1, 1, 1, 1], padding='SAME') +
        model_params['bconv4_1'])
    conv4_2 = tf.nn.relu(
        tf.nn.conv2d(
            conv4_1, model_params['conv4_2'], [1, 1, 1, 1], padding='SAME') +
        model_params['bconv4_2'])
    conv4_3 = tf.nn.relu(
        tf.nn.conv2d(
            conv4_2, model_params['conv4_3'], [1, 1, 1, 1], padding='SAME') +
        model_params['bconv4_3'])
    pool4 = tf.nn.max_pool(
        conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Layer5
    conv5_1 = tf.nn.relu(
        tf.nn.conv2d(
            pool4, model_params['conv5_1'], [1, 1, 1, 1], padding='SAME') +
        model_params['bconv5_1'])
    conv5_2 = tf.nn.relu(
        tf.nn.conv2d(
            conv5_1, model_params['conv5_2'], [1, 1, 1, 1], padding='SAME') +
        model_params['bconv5_2'])
    conv5_3 = tf.nn.relu(
        tf.nn.conv2d(
            conv5_2, model_params['conv5_3'], [1, 1, 1, 1], padding='SAME') +
        model_params['bconv5_3'])
    pool5 = tf.nn.max_pool(
        conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Layer6
    fc6 = tf.nn.relu(
        tf.nn.conv2d(
            pool5, model_params['fc6'], [1, 1, 1, 1], padding='VALID') +
        model_params['bfc6'])
    fc6_dropout = tf.nn.dropout(fc6, keep_prob=keep_prob)

    # Layer7
    fc7 = tf.nn.relu(
        tf.nn.conv2d(
            fc6_dropout, model_params['fc7'], [1, 1, 1, 1], padding='SAME') +
        model_params['bfc7'])
    fc7_dropout = tf.nn.dropout(fc7, keep_prob=keep_prob)

    # score-fr
    score_fr = tf.nn.relu(
        tf.nn.conv2d(
            fc7_dropout,
            model_params['score_fr'], [1, 1, 1, 1],
            padding='SAME') + model_params['bscore_fr'])

    # score2
    score2 = tf.nn.relu(
        tf.layers.conv2d_transpose(
            score_fr, 21, kernel_size=(4, 4), strides=(2, 2), padding='VALID')
        + model_params('bscore2'))

    # Not to use: score_pool4, crop_pool4, fuse_1

    # score4
    score4 = tf.nn.relu(
        tf.layers.conv2d_transpose(
            score2, 21, kernel_size=(4, 4), strides=(2, 2), padding='VALID') +
        model_params['bscore4'])

    # upsample (= 500x500x21)
    upsample = tf.nn.relu(
        tf.layers.conv2d_transpose(
            score4, 21, kernel_size=(17, 17), strides=(7, 7), padding='VALID')
        + model_params['bupsample'])

    return upsample
