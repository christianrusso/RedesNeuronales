from math import exp, sqrt
import numpy as np

class som():

    #def __init__(self, ninputs, output_size, learning_rate, sigma):
    def __init__(self, x,y, learning_rate, sigma):
        self.learning_rate = learning_rate
        self.sigma = sigma

        #self.ninputs = ninputs
        #self.noutputs = output_size[0] * output_size[1]

        #self.neigx = np.arange(output_size[0])
        #self.neigy = np.arange(output_size[1])
        self.X=x
        self.Y=y
        self.noutputs = x *y
        self.neigx = np.arange(x)
        self.neigy = np.arange(y)  

    def activate(self, x):
        y_mono = np.linalg.norm(x.T-self.weights, None, False)
        y = (y_mono == y_mono.min())*1
        return np.reshape(y,(1,self.noutputs))

    def gaussian(self, c, sigma):

        d = 2 * np.pi * sigma * sigma
        ax = np.exp(-np.power(self.neigx - c[0], 2) / d)
        ay = np.exp(-np.power(self.neigy - c[1], 2)/ d)
        return np.outer(ax, ay)

    def correction(self, x, learning_rate, sigma):

        y = self.activate(x)
        # print y.max()
        p = np.unravel_index(y.argmax(), y.shape)
        d = self.gaussian(p, sigma)
        dw = learning_rate * d.flatten() * (x.T - self.weights)
        self.weights += dw

        return dw

    def train(self, dataset, epochs, callback=None):
        self.ninputs = len(dataset[0])
        self.weights = np.random.uniform(-0.5, 0.5, (self.ninputs, self.noutputs))
        for t in range(1, epochs + 1):
            eta = t ** (- self.learning_rate)
            sigma = t ** (- self.sigma)

            tdw = np.zeros((self.ninputs, self.noutputs))
            for x in dataset:
                tdw += self.correction(np.array(x).reshape((1,856)), eta, sigma) ** 2
            if(t%10==0):
                print "epoca: "+str(t)

            