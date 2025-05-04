#!/usr/bin/python3

import numpy as np

class Softmax:
    # Converts Arbitrary Values from layers to probabilties
    def __init__(self, input_nodes, output_nodes):
        self.weights = np.random.randn(input_nodes, output_nodes) / input_nodes
        self.biases = np.zeros(output_nodes)

    def backprop(self, d_L_d_out):
        ''' does the backprop stage in Softmax layer. Returns loss grad for
        inputs. d_L_d_out is the loss grad '''
        # We only know 1 element will be nonzero
        for i, grad in enumerate(d_L_d_out):
            if grad == 0:
                continue

        # e^totals
        t_exp = np.exp(self.last_totals)
        # Sum of all e^totals
        S = np.sum(t_exp)
        # Grad of out[i] against totals
        d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
        d_out_d_t[i] = t_exp[i] * (S-t_exp[i]) / (S ** 2)


    def forward(self, input):
        self.last_input_shape = input.shape

        input = input.flatten()
        self.last_input = input

        input_nodes, output_nodes = self.weights.shape

        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals

        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)

