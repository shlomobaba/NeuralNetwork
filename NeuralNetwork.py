import numpy as np
import matplotlib.pyplot as plt
from keras.api.datasets import mnist
import pandas as pd
import cv2 as cv


class NeuralNetwork:
    layers = []

    def __init__(self, layers):
        self.layers = layers

    def calculate_network_output(self, inputs):
        for layer in self.layers:
            inputs = layer.calculate_output(inputs)
        return inputs


class Layer(object):
    nodes = 0
    input_nodes = 0
    weights = []
    biases = []
    activation_function = None

    def __init__(self, nodes, input_nodes, activation_function):
        self.nodes = nodes
        self.input_nodes = input_nodes
        self.activation_function = activation_function
        self.weights = np.random.rand(nodes, input_nodes) - 0.5
        self.biases = np.random.rand(nodes, 1) - 0.5

    def calculate_output(self, inputs):
        return self.activation_function(np.dot(self.weights, inputs) + self.biases)

    @staticmethod
    def ReLU(output):
        return np.max(output, 0)

    @staticmethod
    def softmax(output):
        return np.exp(output) / np.sum(np.exp(output))


def main():
    # loading the dataset
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    # printing the shapes of the vectors
    print('X_train: ' + str(train_x.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  ' + str(test_x.shape))
    print('Y_test:  ' + str(test_y.shape))

    network = NeuralNetwork([
        Layer(10, 784, Layer.ReLU),
        Layer(10, 10, Layer.softmax),
    ])
    # print(network.calculate_network_output())
    # network.calculate_network_output()


if __name__ == '__main__':
    main()


