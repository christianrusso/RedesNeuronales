from math import exp, sqrt
import numpy as np

class som():

    #def __init__(self, ninputs, output_size, learning_rate, sigma):
    def __init__(self, M1,M2, learning_rate, sigma):
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.M1=M1
        self.M2=M2
        self.noutputs = M1 *M2
        self.filas = np.arange(M1)
        self.columnas = np.arange(M2)  
        

    def activate(self, x):
        y_mono = np.linalg.norm(x.T-self.weights, None, axis=0)

        y = (y_mono == y_mono.min())*1
        # print y_mono

        return np.reshape(y,(1,self.noutputs))

    def gauss(self, p_j, sigma):

        d = 2 * sigma * sigma
        filas = np.exp(-np.power(self.filas-p_j[0],2)/d)
        columnas = np.exp(-np.power(self.columnas-p_j[1],2)/d)
        
        d_matriz = np.outer(filas, columnas)
        return d_matriz.reshape(1, self.noutputs) ##### CONSULTAR
        

    def pIndexAlMapa(self, index):
        return (index / self.M2, index % self.M2)
        
    def pIndexDesdeMapa(self, x, y):
        return x*self.M2 + y%self.M2
    
    def correction(self, x,learning_rate, sigma,y):
        jasterisk = np.argmax(y)
        # print y.max()

        p_j = self.pIndexAlMapa(jasterisk)
        d = self.gauss(p_j, sigma)
        dw = learning_rate * np.multiply(d, (x.T - self.weights)) ##### CONSULTAR
        self.weights += dw
        return dw

    def train(self, dataset, epochs, callback=None):
        self.ninputs = len(dataset[0])
        self.weights = np.random.uniform(-0.5, 0.5, (self.ninputs, self.noutputs))
        for t in range(1, epochs + 1):
            eta = t ** (- 1/2)
            sigma = (self.M2/2)* (t ** (-1/3))

            #tdw = np.zeros((self.ninputs, self.noutputs))
            for x in dataset:
                #tdw += self.correction(np.array(x).reshape((1,856)), eta, sigma) ** 2
                y = self.activate(np.array(x).reshape((1,856)))
                self.correction(np.array(x).reshape((1,856)), eta, sigma, y) ** 2
            # if(t%10==0):
            print "epoca: "+str(t)

    def test(self,x):
        y=self.activate(x)
        posicion=np.argmax(y)
        return self.pIndexAlMapa(posicion)