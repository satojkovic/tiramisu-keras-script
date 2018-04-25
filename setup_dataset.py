#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
from PIL import Image
import bcolz
from concurrent.futures import ProcessPoolExecutor
import common


def parse_line(line):
    elems = line.strip().split('\t')
    code, name = [e for e in elems if len(e) != 0]
    return tuple(int(c) for c in code.split(' ')), name


def parse_label_colors(fn):
    label_codes, label_names = zip(* [parse_line(l) for l in open(fn)])
    return list(label_codes), list(label_names)


def conv_one_label(params, i):
    (labels, code2id, failed_code, row, col) = params
    res = np.zeros((row, col), 'uint8')
    for j in range(row):
        for k in range(col):
            try:
                res[j, k] = code2id[tuple(labels[i, j, k])]
            except:
                res[j, k] = failed_code
    return res


def conv_all_labels(labels, code2id, failed_code, n, row, col):
    res = np.zeros((n, row, col), 'uint8')
    for i in range(n):
        for j in range(row):
            for k in range(col):
                try:
                    res[i, j, k] = code2id[tuple(labels[i, j, k])]
                except:
                    res[i, j, k] = failed_code
    return res


def conv_all_labels_multi(labels, code2id, failed_code, n, row, col):
    with ProcessPoolExecutor(max_workers=8) as ex:
        params = map(lambda _: (labels, code2id, failed_code, row, col),
                     range(n))
        results = np.stack(ex.map(conv_one_label, params, range(n)))
    return results


def main():
    frames_path = os.path.join(common.DATASET_ROOT, '701_StillsRaw_full/')
    labels_path = os.path.join(common.DATASET_ROOT, 'LabeledApproved_full/')

    fnames = glob.glob(os.path.join(frames_path, '*.png'))
    lnames = [
        os.path.join(labels_path, os.path.basename(fn)[:-4] + '_L.png')
        for fn in fnames
    ]

    imgs = np.stack([common.open_image(fn) for fn in fnames])
    labels = np.stack([common.open_image(fn) for fn in lnames])

    # Normalize and standardize
    imgs = common.normalize(imgs)
    imgs = common.standardize(imgs)

    common.save_array(
        os.path.join(common.DATASET_ROOT, 'results/imgs.bc'), imgs)
    common.save_array(
        os.path.join(common.DATASET_ROOT, 'results/labels.bc'), labels)

    # Convert labels
    label_codes, label_names = parse_label_colors(
        os.path.join(common.DATASET_ROOT, 'label_colors.txt'))
    code2id = {v: k for k, v in enumerate(label_codes)}
    failed_code = len(label_codes) + 1
    label_codes.append((0, 0, 0))
    label_names.append('unk')
    labels_int = conv_all_labels(labels, code2id, failed_code, imgs.shape[0],
                                 imgs.shape[1], imgs.shape[2])
    labels_int[labels_int == failed_code] = 0
    common.save_array(
        os.path.join(common.DATASET_ROOT, 'results/labels_int.bc'), labels_int)


if __name__ == '__main__':
    main()
