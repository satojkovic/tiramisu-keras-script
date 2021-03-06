#!/usr/bin/env python
# -*- coding: utf-8 -*-

import model
import os
import time
import matplotlib.pyplot as plt
import argparse
import common

import keras
from keras.layers import Input
from keras.optimizers import RMSprop
from keras.models import Model

NUM_CLASSES = 33


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batchsize',
        '-b',
        type=int,
        default=10,
        help='Number of samples in each mini batch')
    parser.add_argument(
        '--epochs', '-e', type=int, default=1, help='Number of epochs')
    args = parser.parse_args()

    # load dataset
    imgs = common.load_array(
        os.path.join(common.DATASET_ROOT, 'results/imgs.bc'))
    labels = common.load_array(
        os.path.join(common.DATASET_ROOT, 'results/labels.bc'))
    labels_int = common.load_array(
        os.path.join(common.DATASET_ROOT, 'results/labels_int.bc'))

    # Split dataset
    n_data = len(imgs)
    n_trn_imgs = int(0.7 * n_data)
    n_test_imgs = n_data - n_trn_imgs
    trn_imgs = imgs[:n_trn_imgs]
    trn_labels_int = labels_int[:n_trn_imgs]
    test_imgs = imgs[n_trn_imgs:]
    test_labels_int = labels_int[n_trn_imgs:]
    test_labels = labels[n_trn_imgs:]

    # Save test data for prediction
    common.save_array(
        os.path.join(common.DATASET_ROOT, 'results/testset_imgs.bc'),
        test_imgs)
    common.save_array(
        os.path.join(common.DATASET_ROOT, 'results/testset_labels.bc'),
        test_labels)

    print('Training data:', trn_imgs.shape, trn_labels_int.shape)
    print('Test data:', test_imgs.shape, test_labels_int.shape)

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

    batch_size = args.batchsize
    epochs = args.epochs
    t = time.time()

    gen = common.segm_generator(trn_imgs, trn_labels_int, 3, train=True)
    gen_test = common.segm_generator(
        test_imgs, test_labels_int, 3, train=False)
    history = md.fit_generator(
        gen,
        n_trn_imgs // batch_size,
        epochs,
        verbose=1,
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
