# Medhi, Pranav
# 1001-756-326
# 2020_04_20
# Assignment-04-03


import tensorflow as tf
import tensorflow.keras as keras
import pytest
import numpy as np
from cnn import CNN
import os
from tensorflow.keras import utils as np_utils
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist



def test_1():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
    X_train = X_train / 255
    X_test = X_test / 255

    tc = np_utils.to_categorical(y_test)
    num_classes = tc.shape[1]
    model=CNN()
    model.add_input_layer(shape=(28, 28, 1), name="input0")
    model.append_conv2d_layer(32, (5, 5), activation='relu')
    model.append_maxpooling2d_layer(pool_size=(2, 2), padding="same", strides=2, name='pooling')
    model.append_flatten_layer(name='flatten')
    model.append_dense_layer(num_nodes=128, activation='relu')
    model.append_dense_layer(num_nodes=10, activation='softmax')
    
    model.set_loss_function(loss="SparseCategoricalCrossentropy")
    model.set_optimizer(optimizer="SGD")
    model.set_metric('accuracy')
    batch_size=1000
    num_epoch=10
    y=model.train(X_train,y_train,batch_size=batch_size,num_epochs=num_epoch)

    # y=history.history['loss']
    assert y[-1]<y[0]
    # x=y.history['acc']
    x=model.accuracies
    assert x[-1]>x[0]


def test_2():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1)).astype('float32')
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32')
    X_train = X_train / 255
    X_test = X_test / 255
    tc = np_utils.to_categorical(y_test)
    num_classes = tc.shape[1]
    model=CNN()
    model.add_input_layer(shape=(28, 28, 1), name="input0")
    model.append_conv2d_layer(32, (5, 5), activation='relu')
    model.append_maxpooling2d_layer(pool_size=(2, 2), padding="same", strides=2, name='pooling')
    model.append_flatten_layer(name='flatten')
    model.append_dense_layer(num_nodes=128, activation='relu')
    model.append_dense_layer(num_nodes=10, activation='softmax')
    
    model.set_loss_function(loss="SparseCategoricalCrossentropy")
    model.set_optimizer(optimizer="SGD")
    model.set_metric('accuracy')
    batch_size=1000
    num_epoch=10
    history=model.train(X_train,y_train,batch_size=batch_size,num_epochs=num_epoch)
    evaluate = model.evaluate(X_test, y_test)

    assert evaluate[0] < 0.35 and evaluate[1]> 0.9
