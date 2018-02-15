# The MIT License (MIT)
# Copyright (c) 2018 satojkovic

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

import os
import scipy.io as sio
import model
import cv2
import joblib
import numpy as np

PASCAL_PARTS_ANNOT_DIR = os.path.join('PASCAL_parts', 'trainval',
                                      'Annotations_Part')

PASCAL_IMAGE_DIR = os.path.join('VOCdevkit', 'VOC2010', 'JPEGImages')
PASCAL_IMAGE_EXT = '.jpg'

TRAIN_RATIO = 0.7


def get_n_obj(mat_data):
    return mat_data['anno'][0][0][1].shape[1]


def get_n_parts(person_mat_data):
    return person_mat_data[0][3].shape[1]


def get_obj_class(mat_data, i):
    return mat_data['anno'][0][0][1][:, i][0][0]


def get_person_mat_data(mats):
    for mat in mats:
        mat_data = sio.loadmat(os.path.join(PASCAL_PARTS_ANNOT_DIR, mat))
        n_obj = get_n_obj(mat_data)
        for i in range(n_obj):
            if get_obj_class(mat_data, i) == 'person':
                yield mat_data['anno'][0][0][0][0], mat_data['anno'][0][0][
                    1][:, i]


def get_mask_image(person_mat_data, j):
    return person_mat_data[0][3][:, j][1]


def main():
    # Part annotation files
    mats = [mat for mat in os.listdir(PASCAL_PARTS_ANNOT_DIR)]

    # Generate [train|test]_input_images.pickle, [train|test]_ouput_mask_images.pickle
    # input images = (h, w, ch)
    # output mask images = (h, w, n_dims), where n_dims is one hot vector(24 parts)
    input_images = []
    output_mask_images = []

    for fn, person_mat_data in get_person_mat_data(mats):
        img = cv2.imread(os.path.join(PASCAL_IMAGE_DIR, fn + PASCAL_IMAGE_EXT))
        if img is None:
            continue
        img = cv2.resize(img, (model.IMG_WIDTH, model.IMG_HEIGHT))
        input_images.append(img)

        n_parts = get_n_parts(person_mat_data)
        for j in range(n_parts):
            mask_image = get_mask_image(person_mat_data, j)
            output_mask_images.append(mask_image)

    # split input images into train and test set
    n_images = len(input_images)
    all_idxes = np.arange(n_images)
    train_idxes = np.random.choice(
        n_images, int(n_images * TRAIN_RATIO), replace=False)
    test_idxes = np.asarray(sorted(set(all_idxes) - set(train_idxes)))
    train_images, test_images = np.asarray(input_images)[
        train_idxes], np.asarray(input_images)[test_idxes]
    train_masks, test_masks = np.asarray(output_mask_images)[
        train_idxes], np.asarray(output_mask_images)[test_idxes]

    # Output stats
    print('All images:', n_images)
    print('Train images, Train masks:', train_images.shape, train_masks.shape)
    print('Test images, Test masks:', test_images.shape, test_masks.shape)

    # save as pickle
    joblib.dump(train_images, 'train_images.pickle', compress=5)
    joblib.dump(test_images, 'test_images.pickle', compress=5)
    joblib.dump(train_masks, 'train_masks.pickle', compress=5)
    joblib.dump(test_masks, 'test_masks.pickle', compress=5)


if __name__ == '__main__':
    main()
