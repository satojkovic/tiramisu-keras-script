#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import common
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

colors = [(64, 128, 64), (192, 0, 128), (0, 128, 192), (0, 128, 64), (
    128, 0, 0), (64, 0, 128), (64, 0, 192), (192, 128, 64), (192, 192, 128),
          (64, 64, 128), (128, 0, 192), (192, 0, 64), (128, 128, 64), (
              192, 0, 192), (128, 64, 64), (64, 192, 128), (64, 64, 0),
          (128, 64, 128), (128, 128, 192), (0, 0, 192), (192, 128, 128), (
              128, 128, 128), (64, 128, 192), (0, 0, 64), (0, 64, 64), (
                  192, 64, 128), (128, 128, 0), (192, 128, 192), (64, 0, 64), (
                      192, 192, 0), (0, 0, 0), (64, 192, 0)]


def color_label(p):
    res = np.zeros((common.CNN_IMAGE_HEIGHT, common.CNN_IMAGE_WIDTH, 3),
                   'uint8')
    for j in range(common.CNN_IMAGE_HEIGHT):
        for k in range(common.CNN_IMAGE_WIDTH):
            res[j, k, :] = colors[p[j, k]]
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'trained_model_filename', type=str, help='trained model filename')
    args = parser.parse_args()
    trained_model_filename = os.path.abspath(args.trained_model_filename)

    # Check inputs
    if not os.path.exists(trained_model_filename):
        print('Not found:', trained_model_filename)
        sys.exit(-1)

    # Load model
    trained_model = load_model(trained_model_filename)

    # Select target image at random
    testset_imgs = common.load_array(
        os.path.join(common.DATASET_ROOT, 'results', 'testset_imgs.bc'))
    testset_labels = common.load_array(
        os.path.join(common.DATASET_ROOT, 'results', 'testset_labels.bc'))
    gen = common.segm_generator(testset_imgs, testset_labels, 1, train=False)
    target_img, target_label = next(gen)

    # Prediction
    pred = trained_model.predict(target_img)
    pred_img = np.argmax(pred[0], -1).reshape(common.CNN_IMAGE_HEIGHT,
                                              common.CNN_IMAGE_WIDTH)

    # View the segmentation result of targe image
    tl = target_label.reshape(common.CNN_IMAGE_HEIGHT, common.CNN_IMAGE_WIDTH,
                              3)
    pred_img_color = color_label(pred_img)
    target_shape = target_img.shape
    target_img = target_img.reshape(target_shape[1:])
    imm = [(target_img * 0.3 + 0.4), tl, pred_img_color]
    title = ['originial', 'ground-truth', 'prediction']
    plt.figure(figsize=(20, 15))

    for i in range(len(imm)):
        plt.subplot(1, 3, i + 1)
        plt.imshow(imm[i])
        plt.title(title[i], fontsize=30)
    plt.show()


if __name__ == '__main__':
    main()
