#!/usr/bin/python

import random
import collections # you can use collections.Counter if you would like
import math

import numpy as np

from util import *

SEED = 4312

############################################################
# Problem 1: hinge loss
############################################################

def problem_1a():
    """
    return a dictionary that contains the following words as keys:
        so, interesting, great, plot, bored, not
    """
    # BEGIN_YOUR_ANSWER
    return {"so":0, "interesting":0, "great":1, "plot":1, "bored":-1, "not":-1}
    # END_YOUR_ANSWER

############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_ANSWER
    data: list[str] = x.split()
    return {word : data.count(word) for word in set(data)}
    # END_YOUR_ANSWER

############################################################
# Problem 2b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor, numIters, eta):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement stochastic gradient descent.

    Note:
    1. only use the trainExamples for training!
    You can call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    2. don't shuffle trainExamples and use them in the original order to update weights.
    3. don't use any mini-batch whose size is more than 1
    '''
    weights = {}  # feature => weight

    def sigmoid(n):
        return 1 / (1 + math.exp(-n))

    # BEGIN_YOUR_ANSWER
    def loss_nll(x, y, w):
        feat = featureExtractor(x)
        scale = y * (1 - sigmoid(dotProduct(w, feat)))
        return {key: val * scale for key, val in zip(feat.keys(), feat.values())}
    
    for train_data, _ in trainExamples:
        weights[train_data] = 0

    for it in range(numIters):
        count = 0
        for train_data, train_label in trainExamples:
            if dotProduct(weights, featureExtractor(train_data)) * train_label < 1:
                increment(weights, eta, loss_nll(train_data, train_label, weights))
    
    # END_YOUR_ANSWER
    return weights

############################################################
# Problem 2c: bigram features

def extractNgramFeatures(x, n):
    """
    Extract n-gram features for a string x
    
    @param string x, int n: 
    @return dict: feature vector representation of x. (key: n consecutive word (string) / value: occurrence)
    
    For example:
    >>> extractNgramFeatures("I am what I am", 2)
    {'I am': 2, 'am what': 1, 'what I': 1}

    Note:
    There should be a space between words and NO spaces at the beginning and end of the key
    -> "I am" (O) " I am" (X) "I am " (X) "Iam" (X)

    Another example
    >>> extractNgramFeatures("I am what I am what I am", 3)
    {'I am what': 2, 'am what I': 2, 'what I am': 2}
    """
    # BEGIN_YOUR_ANSWER
    words = x.split()
    gram = [' '.join(words[i:i+n]).strip() for i in range(len(words) - n + 1)]
    gram_set = set(gram)
    return {key: gram.count(key) for key in gram}
    # END_YOUR_ANSWER

############################################################
# Problem 3: Multi-layer perceptron & Backpropagation
############################################################

class MLPBinaryClassifier:
    """
    A binary classifier with a 2-layer neural network
        input --(hidden layer)--> hidden --(output layer)--> output
    Each layer consists of an affine transformation and a sigmoid activation.
        layer(x) = sigmoid(x @ W + b)
    """
    def __init__(self):
        self.input_size = 2  # input feature dimension
        self.hidden_size = 16  # hidden layer dimension
        self.output_size = 1  # output dimension

        # Initialize the weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
        self.init_weights()

    def init_weights(self):
        weights = np.load("initial_weights.npz")
        self.W1 = weights["W1"]
        self.W2 = weights["W2"]

    def forward(self, x):
        """
        Inputs
            x: input 2-dimensional feature (B, 2), B: batch size
        Outputs
            pred: predicted probability (0 to 1), (B,)
        """
        # BEGIN_YOUR_ANSWER
        self.x = x
        self.z1 = np.matmul(x, self.W1) + self.b1
        self.a1 = 1 / (1 + np.exp(-self.z1))
        self.z2 = np.matmul(self.a1, self.W2) + self.b2
        self.a2 = 1 / (1 + np.exp(-self.z2))
        return np.transpose(self.a2).flatten()
        # END_YOUR_ANSWER

    @staticmethod
    def loss(pred, target):
        """
        Inputs
            pred: predicted probability (0 to 1), (B,)
            target: true label, 0 or 1, (B,)
        Outputs
            loss: negative log likelihood loss, (B,)
        """
        # BEGIN_YOUR_ANSWER
        zero = -np.log(1 - pred)
        one = -np.log(pred)
        return (zero * np.transpose(1 - target) + one * np.transpose(target))
        # END_YOUR_ANSWER

    def backward(self, pred, target):
        """
        Inputs
            pred: predicted probability (0 to 1), (B,)
            target: true label, 0 or 1, (B,)
        Outputs
            gradient: a dictionary of gradients, {"W1": ..., "b1": ..., "W2": ..., "b2": ...}
        """
        # BEGIN_YOUR_ANSWER
        recent_loss = self.loss(pred, target)

        x = self.x if self.x.ndim == 2 else np.expand_dims(self.x, 0)

        L_sig2 = (1 - target) / (1 - np.transpose(self.a2).squeeze(0)) - (target / np.transpose(self.a2).squeeze(0))
        sig2_z2 = np.transpose(self.a2).squeeze(0) * (1 - np.transpose(self.a2).squeeze(0))
        L_z2 = np.expand_dims(np.expand_dims((L_sig2 * sig2_z2).sum(), 0), 0)

        z2_W2 = self.a1
        L_W2 = np.expand_dims(np.matmul((L_sig2 * sig2_z2), z2_W2), 1)

        z2_sig1 = np.transpose(self.W2)
        sig1_z1 = self.a1 * (1 - self.a1)
        L_b1 = np.expand_dims((np.matmul(np.expand_dims(L_sig2 * sig2_z2, 1), z2_sig1) * sig1_z1).sum(axis = 0), 0)
        
        L_W1 = np.transpose(np.matmul(np.transpose(np.matmul(np.expand_dims(L_sig2 * sig2_z2, 1), z2_sig1) * sig1_z1), x))

        return {"W1": L_W1, "b1": L_b1, "W2": L_W2, "b2": L_z2}
        # END_YOUR_ANSWER
    
    def update(self, gradients, learning_rate):
        """
        A function to update the weights and biases using the gradients
        Inputs
            gradients: a dictionary of gradients, {"W1": ..., "b1": ..., "W2": ..., "b2": ...}
            learning_rate: step size for weight update
        Outputs
            None
        """
        # BEGIN_YOUR_ANSWER
        for attr in ["W1", "b1", "W2", "b2"]:
            setattr(self, attr, getattr(self, attr) - learning_rate * gradients[attr])
        return
        # END_YOUR_ANSWER

    def train(self, X, Y, epochs=100, learning_rate=0.1):
        """
        A training function to update the weights and biases using stochastic gradient descent
        Inputs
            X: input features, (N, 2), N: number of samples
            Y: true labels, (N,)
            epochs: number of epochs to train
            learning_rate: step size for weight update
        Outputs
            loss: the negative log likelihood loss of the last step
        """
        # BEGIN_YOUR_ANSWER
        for _ in range(epochs):
            for x, y in zip(X, Y):
                prdc = self.forward(x)
                loss = self.loss(prdc, y)
                grad = self.backward(prdc, y)
                self.update(grad, learning_rate)
            
        return loss
        # END_YOUR_ANSWER

    def predict(self, x):
        return np.round(self.forward(x))