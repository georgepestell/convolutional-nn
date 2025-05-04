#!/usr/bin/python3
import mnist
import numpy as np
from conv import Conv_3x3
from maxpool import MaxPool2
from softmax import Softmax

global output_nodes

td_images = mnist.train_images()[:10000]
td_labels = mnist.train_labels()[:10000]

no_filters = 8
input_nodes = np.prod(np.subtract(td_images[0].shape, 2)) // 4
output_nodes = 10

# For this we are using 8 filters
conv = Conv_3x3(no_filters)
pool = MaxPool2()
softmax = Softmax(input_nodes*no_filters, output_nodes)

def forward(image, label):
    global output_nodes
    output = conv.forward((image / 255) - 0.5)
    output = pool.forward(output)
    output = softmax.forward(output)

    gradient = np.zeros(output_nodes)
    gradient[label] = -1 / output[label]

#    gradient = softmax.backprop(gradient)
#    gradient = pool.backprop(gradient)
#    gradient = conv.backprop(gradient)

    loss = -np.log(output[label])
    print(loss)
    accuracy = 1 if np.argmax(output) == label else 0

    return output, loss, accuracy

loss = 0
correct = 0

for i, (image, label) in enumerate(zip(td_images, td_labels)):
    _, l, accuracy = forward(image, label)
    loss += l
    correct += accuracy

    if i % 100 == 99:
        print(
          '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
          (i + 1, loss / 100, correct)
        )
        loss = 0
        correct = 0

