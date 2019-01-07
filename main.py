import numpy as np
import random

input_size = 30  # size of an input array


# sigmoid activation function
def sigmoid(x, derivative=False):
    sig = 1 / (1 + np.exp(-x))
    if derivative:
        return sig * (1 - sig)
    return sig


# output cost function
def cost(prediction, actual, derivative=False):
    if not derivative:
        return (prediction - actual) ** 2
    else:
        return 2 * (prediction - actual)


# generate training data
training_outputs = []
training_inputs = []
for i in range(500):
    training_inputs.append([])
    for j in range(input_size):
        training_inputs[i].append(random.randint(0, 1))

    if training_inputs[i][1] == 1:
        training_outputs.append(1)
    else:
        training_outputs.append(0)
training_inputs = np.array(training_inputs)
training_outputs = np.array(training_outputs)


# generate random weights
weights = 2 * np.random.random(input_size) - 1


# performs one training iteration
def train(training_inputs, training_outputs, weights):
    predictions = sigmoid(np.dot(training_inputs, weights))

    zw = training_inputs
    az = sigmoid(np.dot(training_inputs, weights), derivative=True)
    ca = cost(predictions, training_outputs, derivative=True)

    adjustments = -(zw.T * az * ca).T * 10000
    weights += adjustments.mean(0)
    return weights


# Trains the model
for i in range(50000):
    weights = train(training_inputs, training_outputs, weights)


# uses the nn model
def get_predictions(inputs, weights):
    predictions = sigmoid(np.dot(inputs, weights))
    return predictions


test_inputs = [random.randint(0, 1) for i in range(input_size)]
print(test_inputs)
print(get_predictions(np.array(test_inputs), weights))
