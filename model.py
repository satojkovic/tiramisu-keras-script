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


def relu(x):
    return tf.nn.relu(x)


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob=keep_prob) if keep_prob else x


def batch_norm(x):
    return tf.layers.batch_normalization(x, axis=-1)


def relu_batch_norm(x):
    return relu(batch_norm(x))


def concat(xs):
    return tf.concat(xs, axis=-1)


def conv(x, nb_filter, ksize, scale, keep_prob, stride=1):
    x = tf.layers.conv2d(
        x,
        nb_filter,
        ksize,
        strides=(stride, stride),
        padding='SAME',
        kernel_initializer=tf.contrib.layers.xavier_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale))
    return dropout(x, keep_prob)


def conv_relu_batch_norm(x, nb_filter, ksize=3, scale=0, keep_prob=0,
                         stride=1):
    return conv(relu_batch_norm(x), nb_filter, ksize, scale, keep_prob, stride)


def dense_block(n_layers, x, growth_rate, keep_prob, scale):
    added = []
    for i in range(n_layers):
        b = conv_relu_batch_norm(x, growth_rate, keep_prob, scale)
        x = concat([x, b])
        added.append(x)
    return x, added


def create_tiramisu(nb_classes,
                    img_input,
                    nb_dense_block=6,
                    growth_rate=16,
                    nb_filter=48,
                    nb_layers_per_block=5,
                    keep_prob=None,
                    scale=0):
    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)
    else:
        nb_layers = [nb_layers_per_block] * nb_dense_block

    x = conv(img_input, nb_filter, 3, scale, 0)
    return x
