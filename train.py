#!/usr/bin/env python
# -*- coding: utf-8 -*-

import model
import bcolz
import os
import numpy as np
import threading
import random
import time
import matplotlib.pyplot as plt

import keras
from keras.layers import Input
from keras.optimizers import RMSprop
from keras.models import Model

NUM_CLASSES = 33
DATASET_ROOT = 'camvid'


class BatchIndices(object):
    def __init__(self, n, bs, shuffle=False):
        self.n, self.bs, self.shuffle = n, bs, shuffle
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.idxs = (np.random.permutation(self.n)
                     if self.shuffle else np.arange(0, self.n))
        self.curr = 0

    def __next__(self):
        with self.lock:
            if self.curr >= self.n:
                self.reset()
            ni = min(self.bs, self.n - self.curr)
            res = self.idxs[self.curr:self.curr + ni]
            self.curr += ni
            return res


class segm_generator(object):
    def __init__(self, x, y, bs=64, out_sz=(224, 224), train=True):
        self.x, self.y, self.bs, self.train = x, y, bs, train
        self.n, self.rowi, self.coli, _ = x.shape
        self.idx_gen = BatchIndices(self.n, self.bs, train)
        self.rowo, self.colo = out_sz
        self.ych = self.y.shape[-1] if len(y.shape) == 4 else 1

    def get_slice(self, i, o):
        start = random.randint(0, i - o) if self.train else (i - o)
        return slice(start, start + o)

    def get_item(self, idx):
        slice_r = self.get_slice(self.rowi, self.rowo)
        slice_c = self.get_slice(self.coli, self.colo)
        x = self.x[idx, slice_r, slice_c]
        y = self.y[idx, slice_r, slice_c]
        if self.train and (random.random() > 0.5):
            # horizontal flipping
            y = y[:, ::-1]
            x = x[:, ::-1]
        return x, y

    def __next__(self):
        idxs = next(self.idx_gen)
        items = (self.get_item(idx) for idx in idxs)
        xs, ys = zip(*items)
        return np.stack(xs), np.stack(ys).reshape(len(ys), -1, self.ych)


def load_array(fname):
    return bcolz.open(fname)


def standardize(imgs):
    avg = np.mean(imgs)
    sd = np.std(imgs)
    imgs -= avg
    imgs /= sd
    return imgs


def main():
    # load dataset
    imgs = load_array(os.path.join(DATASET_ROOT, 'results/imgs.bc'))
    labels_int = load_array(
        os.path.join(DATASET_ROOT, 'results/labels_int.bc'))

    # standardize values of image data
    imgs = standardize(imgs)

    # Split dataset
    n_data = len(imgs)
    n_trn_imgs = int(0.7 * n_data)
    n_test_imgs = n_data - n_trn_imgs
    trn_imgs = imgs[:n_trn_imgs]
    trn_labels = labels_int[:n_trn_imgs]
    test_imgs = imgs[n_trn_imgs:]
    test_labels = labels_int[n_trn_imgs:]
    print('Training data:', trn_imgs.shape, trn_labels.shape)
    print('Test data:', test_imgs.shape, test_labels.shape)

    # Train the model
    img_input = Input(shape=(224, 224, 3))
    x = model.create_tiramisu(
        33,
        img_input,
        growth_rate=12,
        nb_layers_per_block=4,
        keep_prob=0.2,
        scale=1e-4)
    md = Model(img_input, x)
    md.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=RMSprop(1e-3, decay=1 - 0.99995),
        metrics=["accuracy"])
    filepath = 'tiramisu.{epoch:02d}-{val_acc:.2f}.hdf5'
    mc_cb = keras.callbacks.ModelCheckpoint(
        filepath,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='max')

    batch_size = 3
    epochs = 50
    t = time.time()

    gen = segm_generator(trn_imgs, trn_labels, 3, train=True)
    gen_test = segm_generator(test_imgs, test_labels, 3, train=False)
    history = md.fit_generator(
        gen,
        n_trn_imgs // batch_size,
        epochs,
        verbose=2,
        validation_data=gen_test,
        validation_steps=n_test_imgs // batch_size,
        callbacks=[mc_cb])
    t2 = time.time()
    print(round(t2 - t, 5), 'seconds to predict')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.show()

    md.summary()


if __name__ == '__main__':
    main()
