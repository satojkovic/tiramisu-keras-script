#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import threading
import random
from PIL import Image
import bcolz
import os

IMAGE_WIDTH = 480
IMAGE_HEIGHT = 360
CNN_IMAGE_WIDTH = 224
CNN_IMAGE_HEIGHT = 224
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
    def __init__(self,
                 x,
                 y,
                 bs=64,
                 out_sz=(CNN_IMAGE_HEIGHT, CNN_IMAGE_WIDTH),
                 train=True):
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


def open_image(fn):
    return np.asarray(
        Image.open(fn).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.NEAREST))


def standardize(imgs):
    avg = np.mean(imgs)
    sd = np.std(imgs)
    imgs -= avg
    imgs /= sd
    return imgs


def normalize(imgs):
    return imgs / 255.


def load_array(fname):
    return bcolz.open(fname)


def save_array(save_fname, arr):
    base_dir = os.path.dirname(save_fname)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    c = bcolz.carray(arr, rootdir=save_fname, mode='w')
    c.flush()
