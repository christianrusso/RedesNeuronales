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
        for e in range(1, epochs + 1):
            learning_rate = self.learning_rate or (0.5 / e)

            for x in dataset:
                y = np.dot(x,self.weights)
                if self.optimized:
                    if self.use_oja:
                        x_mono = np.dot(y, self.weights.T)
                        dw = learning_rate * np.dot((x - x_mono).T, y)
                    else:
                        U = np.triu(np.ones(self.noutputs))
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
            print "epoca: "+str(e)


    def test(self, dataset):
        # TODO: En vez de epochs, agregar ortogonalidad como criterio de parada
        y=[]
        for x in dataset:
            y.append(self.activate(np.array(x[1:]).reshape((1, 856))))
        return y


###########################
# EN TEORIA LO DE ABAJO ES LO QUE NOS DIERON TODAVIA
# ES PARA OPTIMIZAR, ASI QUE NO DEBERIA SER NECESARIO

# class GHANeuralNetwork():

#     def __init__(self, ninputs, noutputs, sigma0, alfa):

#         self.sigma0 = sigma0
#         self.alfa = alfa

#         self.ninputs = ninputs
#         self.noutputs = noutputs
#         self.weights = np.random.uniform(-0.5, 0.5, (noutputs, ninputs))

#     def ninputs(self):
#         return self.ninputs

#     def noutputs(self):
#         return self.noutputs

#     def weights(self):
#         return self.weights

#     def activate(self, input):

#         x = np.array([input]).T
#         y = np.dot(self.weights, x)
#         return y.T[0]

#     def train(self, dataset, epochs, callback=None):

#         for t in range(1, epochs + 1):

#             sigma = self.sigma0 * (t ** -self.alfa)
#             tdw = 0

#             for x in dataset:

#                 x = np.array([x]).T
#                 y = np.dot(self.weights, x)

#                 dw = sigma * ( np.dot(y, x.T) - np.dot(np.tril(np.dot(y, y.T)), self.weights) )
#                 self.weights += dw
#                 tdw += dw ** 2

#             if callback: callback(self, t, tdw.sum() / len(dataset))

