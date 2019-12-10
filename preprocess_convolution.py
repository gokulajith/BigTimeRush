import pickle
import sys

import numpy as np
import tensorflow as tf
import os


# a normal image takes in lets say 300x300x3 where 300 is the dimensions of the picture
# and 3 is the 3 rbg channels
# instead our model will take in an play as an "image"
# this image will be of size 11,2,10 where the first 11 rows is defense and 2nd 11 rows
# is offense

def get_data(file_path, first_class, second_class):
    """
    Given a file path and two target classes, returns an array of
    normalized inputs (images) and an array of labels.
    You will want to first extract only the data that matches the
    corresponding classes we want (there are 10 classes and we only want 2).
    You should make sure to normalize all inputs and also turn the labels
    into one hot vectors using tf.one_hot().
    Note that because you are using tf.one_hot() for your labels, your
    labels will be a Tensor, while your inputs will be a NumPy array. This
    is fine because TensorFlow works with NumPy arrays.
    :param file_path: file path for inputs and labels, something
    like 'CIFAR_data_compressed/train'
    :param first_class:  an integer (0-9) representing the first target
    class in the CIFAR10 dataset, for a cat, this would be a 3
    :param first_class:  an integer (0-9) representing the second target
    class in the CIFAR10 dataset, for a dog, this would be a 5
    :return: normalized NumPy array of inputs and tensor of labels, where
    inputs are of type np.float32 and has size (num_inputs, width, height, num_channels) and labels
    has size (num_examples, num_classes)
    """
    unpickled_file = unpickle(file_path)
    inputs = unpickled_file[b'data']
    labels = unpickled_file[b'labels']

    correct_inputs = []
    correct_labels = []
    for i in range(len(inputs)):
        if labels[i] == first_class or labels[i] == second_class:
            if labels[i] == first_class:
                correct_labels.append(0)
            else:
                correct_labels.append(1)
            correct_inputs.append(inputs[i])

    correct_inputs = np.reshape(correct_inputs, (-1, 3, 32, 32))
    np.set_printoptions(threshold=sys.maxsize)
    correct_inputs = np.transpose(correct_inputs, (0, 2, 3, 1))

    correct_labels = tf.one_hot(correct_labels, depth=2)

    return np.float32(correct_inputs) / 255, correct_labels
