import numpy as np

class red_hebbiana:

    def __init__(self, ninputs, noutputs, learning_rate=None, use_oja=True, optimized=True):

        self.learning_rate = learning_rate
        self.use_oja = use_oja
        self.optimized = optimized
        self.ninputs = ninputs
        self.noutputs = noutputs
        self.weights = np.random.uniform(-0.5, 0.5, (ninputs, noutputs))

    def activate(self, x):

        y = np.dot(x, self.weights)
        return y

    def train(self, dataset, epochs):
        # TODO: En vez de epochs, agregar ortogonalidad como criterio de parada
        e = 0
        while not self.isOrtogonal() and e<epochs:
            learning_rate = self.learning_rate or (0.5 / e)

            for x in dataset:
                y = np.dot(x,self.weights)
                if self.optimized:
                    if self.use_oja:
                        x_mono = np.dot(y, self.weights.T)
                        dw = learning_rate * np.dot((x - x_mono).T, y)
                    else:
                        U = np.triu(np.ones((self.noutputs,self.noutputs)))
                        x_mono = np.dot(self.weights, np.multiply(y.T, U))
                        dw = learning_rate * np.multiply((x.T - x_mono),y)
                else:
                    dw = np.zeros((self.ninputs, self.noutputs), dtype=float)

                    for j in range(0, self.noutputs):
                        for i in range(0, self.ninputs):
                            x_mono = 0
                            for k in range(0, self.use_oja and self.noutputs or (j + 1)):
                                x_mono += y[k] * self.weights[i][k]

                            dw[i][j] = learning_rate * (x[i] - x_mono)* y[j] 

                self.weights += dw
            if(e%30==0): 
                print "epoca: "+str(e)
            e += 1

    def isOrtogonal(self):
        prod = np.dot(self.weights.T, self.weights)
        print prod
        return np.allclose(prod, np.identity(self.noutputs), atol=0.01)


