#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
from PIL import Image
import bcolz

DATASET_ROOT = 'camvid'
IMAGE_WIDTH = 480
IMAGE_HEIGHT = 360


def open_image(fn):
    return np.asarray(
        Image.open(fn).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.NEAREST))


def save_array(save_fname, arr):
    base_dir = os.path.dirname(save_fname)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    c = bcolz.carray(arr, rootdir=save_fname, mode='w')
    c.flush()


def main():
    frames_path = os.path.join(DATASET_ROOT, '701_StillsRaw_full/')
    labels_path = os.path.join(DATASET_ROOT, 'LabeledApproved_full/')

    fnames = glob.glob(os.path.join(frames_path, '*.png'))
    lnames = [
        os.path.join(labels_path, os.path.basename(fn)[:-4] + '_L.png')
        for fn in fnames
    ]

    imgs = np.stack([open_image(fn) for fn in fnames])
    labels = np.stack([open_image(fn) for fn in lnames])

    # Normalize
    imgs = imgs / 255.

    save_array(os.path.join(DATASET_ROOT, 'results/imgs.bc'), imgs)
    save_array(os.path.join(DATASET_ROOT, 'results/labels.bc'), labels)


if __name__ == '__main__':
    main()
