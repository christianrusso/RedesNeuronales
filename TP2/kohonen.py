from math import exp, sqrt
import numpy as np

class som():

    def __init__(self, ninputs, output_size, learning_rate, sigma):

        self.learning_rate = learning_rate
        self.sigma = sigma

        self.ninputs = ninputs
        self.output_size = output_size
        self.noutputs = output_size[0] * output_size[1]

        self.neigx = np.arange(output_size[0])
        self.neigy = np.arange(output_size[1])

        self.weights = np.random.uniform(-0.5, 0.5, (self.ninputs, self.noutputs))

    def activate(self, input):

        y = np.linalg.norm(self.weights - np.array([input]).transpose(), None, 0)
        r = np.zeros_like(y)
        r[y.argmin()] = 1

        return r.reshape(self.output_size);

    def gaussian(self, c, sigma):

        d = 2 * np.pi * sigma * sigma
        ax = np.exp(-np.power(self.neigx - c[0], 2) / d)
        ay = np.exp(-np.power(self.neigy - c[1], 2)/ d)
        return np.outer(ax, ay)

    def correction(self, input, learning_rate, sigma):

        y = self.activate(input)
        p = np.unravel_index(y.argmax(), y.shape)
        d = self.gaussian(p, sigma)
        dw = learning_rate * (np.array([input]).transpose() - self.weights) * d.flatten()
        self.weights += dw

        return dw

    def train(self, dataset, epochs, callback=None):

        for t in range(1, epochs + 1):
            eta = t ** (- self.learning_rate)
            sigma = t ** (- self.sigma)

            tdw = np.zeros((self.ninputs, self.noutputs))
            for x in dataset:
                tdw += self.correction(x, eta, sigma) ** 2
            print "epoca: "+str(t)

            