import numpy

class red_hebbiana:

    def __init__(self, ninputs, noutputs, learning_rate=None, use_oja=None):

        self.learning_rate = learning_rate
        self.use_oja = use_oja

        self.ninputs = ninputs
        self.noutputs = noutputs
        self.weights = np.random.uniform(-0.5, 0.5, (ninputs, noutputs))

    def activate(self, input):

        x = np.array([input])
        y = np.dot(x, self.weights)
        return y[0]

    def train(self, dataset, epochs):
        # TODO: En vez de epochs, agregar ortogonalidad como criterio de parada
        for e in range(1, epochs + 1):
            learning_rate = self.learning_rate or (0.5 / e)

            for x in dataset:
                y = self.activate(x)
                dw = np.zeros((self.ninputs, self.noutputs), dtype=float)

                for j in range(0, self.noutputs):
                    for i in range(0, self.ninputs):
                        xe = 0
                        for k in range(0, self.use_oja and self.noutputs or (j + 1)):
                            xe += y[k] * self.weights[i][k]

                        dw[i][j] = learning_rate * y[j] * (x[i] - xe)

                self.weights += dw
