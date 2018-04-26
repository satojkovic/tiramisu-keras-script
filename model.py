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
from keras.layers import BatchNormalization, Conv2DTranspose, Conv2D
from keras.layers import Activation, Dropout
from keras.layers import concatenate, Reshape
from keras.regularizers import l2


def relu(x):
    return Activation('relu')(x)


def dropout(x, keep_prob):
    return Dropout(keep_prob)(x) if keep_prob else x


def batch_norm(x):
    return BatchNormalization(axis=-1)(x)


def relu_batch_norm(x):
    return relu(batch_norm(x))


def concat(xs):
    return concatenate(xs, axis=-1)


def conv(x, nb_filter, ksize, scale, keep_prob, stride=1):
    x = Conv2D(
        nb_filter,
        ksize,
        kernel_initializer='he_uniform',
        padding='same',
        strides=(stride, stride),
        kernel_regularizer=l2(scale))(x)
    return dropout(x, keep_prob)


def conv_relu_batch_norm(x, nb_filter, ksize=3, scale=0, keep_prob=0,
                         stride=1):
    return conv(relu_batch_norm(x), nb_filter, ksize, scale, keep_prob, stride)


def dense_block(nb_layers, x, growth_rate, keep_prob, scale):
    added = []
    for i in range(nb_layers):
        b = conv_relu_batch_norm(
            x, growth_rate, keep_prob=keep_prob, scale=scale)
        x = concat([x, b])
        added.append(b)
    return x, added


def transition_down(x, keep_prob, scale):
    return conv_relu_batch_norm(
        x,
        x.get_shape().as_list()[-1],
        ksize=1,
        scale=scale,
        keep_prob=keep_prob,
        stride=2)


def down_path(x, nb_layers, growth_rate, keep_prob, scale):
    skips = []
    for i, nb_layer in enumerate(nb_layers):
        x, added = dense_block(nb_layer, x, growth_rate, keep_prob, scale)
        skips.append(x)
        x = transition_down(x, keep_prob, scale)
    return skips, added


def up_path(added, skips, nb_layers, growth_rate, keep_prob, scale):
    for i, nb_layer in enumerate(nb_layers):
        x = transition_up(added, scale)
        x = concat([x, skips[i]])
        x, added = dense_block(nb_layer, x, growth_rate, keep_prob, scale)
    return x


def transition_up(added, scale):
    x = concat(added)
    _, row, col, ch = x.get_shape().as_list()
    return Conv2DTranspose(
        ch,
        3,
        kernel_initializer='he_uniform',
        padding='same',
        strides=(2, 2),
        kernel_regularizer=l2(scale))(x)


def reverse(a):
    return list(reversed(a))


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
    skips, added = down_path(x, nb_layers, growth_rate, keep_prob, scale)
    x = up_path(added,
                reverse(skips[:-1]),
                reverse(nb_layers[:-1]), growth_rate, keep_prob, scale)
    x = conv(x, nb_classes, 1, scale, 0)
    x = Reshape((-1, nb_classes))(x)
    return Activation('softmax')(x)
