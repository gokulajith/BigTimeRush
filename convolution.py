from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_convolution_data
from assignment import label_converter
from sklearn.metrics import mean_absolute_error
import time
import os
import tensorflow as tf
import numpy as np
import random


class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that
        classifies images. Do not modify the constructor, as doing so
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()
        self.epsilon = 0.001


        self.batch_size = 800
        self.hidden_layer = 100
        # self.opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1 = 0.9, beta_2=0.999, epsilon = 1e-8)
        self.opt = tf.keras.optimizers.Adam(learning_rate=0.05)

        # TODO: Initialize all hyperparameters
        # TODO: Initialize all trainable parameters
        #self.CNN_layer_1 = tf.Variable(tf.random.truncated_normal([5, 5, self.input_channel_size, 16], stddev=0.1),                              name="CNN_layer_1")
        #self.CNN_layer_1_b = tf.Variable(tf.random.truncated_normal([16]), name="CNN_layer_1_b")

        #self.CNN_layer_2 = tf.Variable(tf.random.truncated_normal([5, 5, 16, 20], stddev=0.1), name="CNN_layer_2")
        #self.CNN_layer_2_b = tf.Variable(tf.random.truncated_normal([20]), name="CNN_layer_2_b")

        #self.CNN_layer_3 = tf.Variable(tf.random.truncated_normal([5, 5, 20, 20], stddev=0.1), name="CNN_layer_3")
        #self.CNN_layer_3_b = tf.Variable(tf.random.truncated_normal([20]), name="CNN_layer_3_b")

        #self.dense_1 = tf.Variable(tf.random.truncated_normal([self.hidden_layer, self.hidden_layer], stddev=0.1),name="dense_layer_1")
        #self.dense_2 = tf.Variable(tf.random.truncated_normal([self.hidden_layer, self.hidden_layer], stddev=0.1),name="dense_layer_2")
        #self.dense_3 = tf.Variable(tf.random.truncated_normal([self.hidden_layer, self.num_classes], stddev=0.1),name="dense_layer_3")

        #self.bias_1 = tf.Variable(tf.random.truncated_normal([self.hidden_layer]))
        #self.bias_2 = tf.Variable(tf.random.truncated_normal([self.hidden_layer]))
        #self.bias_3 = tf.Variable(tf.random.truncated_normal([self.num_classes]))
        # takes in 2 x 11 x 10
        # end with 31
        # inputs is (batch_size,2, 11,10)
        self.CNN_layer1 = tf.keras.layers.Conv2D(10, (2,2), padding="same")
        self.CNN_layer2 = tf.keras.layers.Conv2D(5, (2,2), padding="same")

        self.CNN_layer3 = tf.keras.layers.Conv2D(100, (2,2), padding="same")
        self.CNN_layer4 = tf.keras.layers.Conv2D(62, (2,2), padding="same")


        self.pooling_layer1 = tf.keras.layers.MaxPool2D((2,2))
        self.pooling_layer2 = tf.keras.layers.MaxPool2D((2,2))
        self.pooling_layer3 = tf.keras.layers.MaxPool2D((2,2))

        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()

        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.dropout2 = tf.keras.layers.Dropout(0.3)
        self.dropout3 = tf.keras.layers.Dropout(0.3)
        self.relu1 = tf.keras.layers.ReLU()
        self.relu2 = tf.keras.layers.ReLU()
        self.relu3 = tf.keras.layers.ReLU()
        self.relu4 = tf.keras.layers.ReLU()

        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(100)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        #print("INPUTS SHAPE",inputs.shape)
        #(20,2,11,10)
        #(20,10,2,11,10)

        # inputs = tf.transpose(inputs, perm=[0,3,2,1])
        # output = self.CNN_layer1(inputs)
        # output = self.relu1(output)
        # output = self.pooling_layer1(output)
        # #print("OUTPUT1 SHAPE", output.shape)
        # output = self.CNN_layer2(output)
        # output = self.relu2(output)
        # output = self.pooling_layer2(output)
        # #print("OUTPUT2 SHAPE", output.shape)
        # output = self.CNN_layer3(output)
        # output = self.relu3(output)
        # output = self.pooling_layer3(output)
        # #print("OUTPUT3 SHAPE", output.shape)
        # output = self.CNN_layer4(output)
        # output = self.flatten_layer(output)
        # output = self.dense1(output)
        # output = self.relu4(output)
        # #print("OUTPUT4 SHAPE", output.shape)
        # #(20, ..,.., 2)
        # #reshape((20, -1))
        # #dense(20, 1)

        #(batch_size,2,11,10)

        lifting = tf.keras.layers.Dense(100)(inputs)

        inputs = tf.transpose(lifting, perm=[0,3,2,1])
        output = self.CNN_layer1(inputs)
        output = self.batch_norm1(output)
        output = self.relu1(output)
        output = self.pooling_layer1(output)
        #print("OUTPUT1 SHAPE", output.shape)
        output = self.CNN_layer2(output)
        output = self.batch_norm2(output)
        output = self.relu2(output)
        output = self.pooling_layer2(output)

        output = self.flatten_layer(output)

        output = self.dense1(output)
        output = self.relu4(output)
        output = self.dropout1(output)



        output = tf.reshape(output, (self.batch_size,-1))
        logits = self.dense2(output)
        # lifting = tf.keras.layers.Dense(100)(inputs)
        # inputs = tf.reshape(lifting, (20, -1))
        # output = tf.keras.layers.Dense(1000, activation = 'relu')(inputs)
        # output = tf.keras.layers.Dense(100, activation = 'relu')(output)
        # logits = tf.keras.layers.Dense(1)(output)
        #print("logits",logits)
        return logits



    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        :param logits: during training, a matrix of shape (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        Softmax is applied in this function.
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """


        #loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
        loss = tf.keras.losses.MSE(labels, logits)
        #loss = tf.nn.sigmoid_cross_entropy_with_logits(tf.reshape(labels, (20, 1)), logits)
        return loss

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT

        :return: the accuracy of the model as a Tensor
        """
        correct = 0
        if (len(logits) != len(labels)):
            print("ERROR: len legits != len labels")


        #logits = logits.astype(int).flatten()

        #print('labels', np.mean(labels))
        #labels = labels.numpy().astype(int)

        #acc = tf.keras.losses.MAE(labels, logits)

        acc = mean_absolute_error(labels, logits)
        #acc = tf.keras.losses.MSE()
        #acc = np.mean(labels == logits)
        #print("curr acc", acc)
        return acc

def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs
    and labels - ensure that they are shuffled in the same order using tf.gather.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training),
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training),
    shape (num_labels, num_classes)
    :return: None
    '''
    indices = tf.random.shuffle(list(range(len(train_inputs))))
    train_inputs = tf.gather(train_inputs, indices)
    #print("inputs", train_inputs[0])
    train_labels = tf.gather(train_labels, indices)

    total_loss = 0
    num_batches = 0
    total_acc = 0
    total_num = 0

    for i in range(0, len(train_inputs), model.batch_size):
        batch_inputs = train_inputs[i:i + model.batch_size]
        batch_labels = train_labels[i:i + model.batch_size]
        batch_labels = label_converter(batch_labels)

        if (len(batch_inputs) < model.batch_size ):
            continue
        #tf.image.random_flip_left_right(batch_inputs)
        with tf.GradientTape() as tape:
            # logits = model(batch_inputs)
            logits = model.call(batch_inputs)

            #print("loss", logits)
            acc = model.accuracy(logits, np.int32(batch_labels))
            #print(acc)
            acc = int(acc)
            total_acc += int(acc)
            #total_num += len(acc)
            loss = model.loss(logits, batch_labels)
            total_loss += sum(loss)/len(loss)
        num_batches += 1
        gradients = tape.gradient(loss, model.trainable_variables)

        model.opt.apply_gradients(zip(gradients, model.trainable_variables))
        if (i % 4800 == 0):
            print("acc", total_acc/num_batches, "loss", total_loss/num_batches)
    return total_acc/num_batches, total_loss/num_batches

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly
    flip images or do any extra preprocessing.
    :param test_inputs: test data (all images to be tested),
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this can be the average accuracy across
    all batches or the sum as long as you eventually divide it by batch_size
    """
    correct = 0
    label_size = 0
    num_batches = 0
    total_loss= 0.0
    indices = tf.random.shuffle(list(range(len(test_inputs))))
    train_inputs = tf.gather(test_inputs, indices)
    train_labels = tf.gather(test_labels, indices)
    for i in range(0, len(test_inputs), model.batch_size):
        batch_inputs = test_inputs[i:i + model.batch_size]
        batch_labels = test_labels[i:i + model.batch_size]
        batch_labels = label_converter(batch_labels)

        if (len(batch_inputs) < model.batch_size):
            continue

        logits = model.call(batch_inputs)
        loss = model.loss(logits, batch_labels)
        total_loss += sum(loss)/len(loss)
        acc = model.accuracy(logits, batch_labels)
        acc = int(acc)

        num_batches += 1
        #print(R2.dtype)
        #print(correct.dtype)

        correct += acc
        label_size += 1

    return correct/num_batches, total_loss/num_batches




def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs. For CS2470 students, you must train within 10 epochs.
    You should receive a final accuracy on the testing examples for cat and dog of >=70%.
    :return: None
    '''
    # get data will return a normalized Numpy Array of inputs
    # and tensor of labels where inputs are of type np.float32 and has size
    # (num_inputs, width, height, num_channels) and labels has size (num_examples, num_classes)
    # takes in file path, first class and second class
    #data_file_path = "data"
    data_file_path = "data/train.csv"



    first_class = 3
    second_class = 5

    train_inputs, train_labels, test_inputs, test_labels = get_convolution_data(data_file_path)

    model = Model()

    epochs = 10

    for i in range(0, epochs):
        acc,loss = train(model, train_inputs, train_labels)
        print("epoch acc ", acc)
        print("epcoh loss", loss)
    # acc = test(model, test_inputs, test_labels)
    # print(acc)
    acc, loss = test(model, test_inputs, test_labels)
    print("total acc is", acc)





    return


if __name__ == '__main__':
    main()
