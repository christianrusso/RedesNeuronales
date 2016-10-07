from math import exp, sqrt
import numpy as np

def norma2(vec):
	return sqrt(np.dot(vec, vec.T))


debug = False
info = False
conteo = True

def normalizar(x):
	norma = norma2(x)
	if norma > 0:
		return x/norma
	else:
		return x
		
def informar(texto, nivel):
	if nivel:
		print texto

		
class SOM:
	def __init__(self, m1, m2, nEntrada, sigma=1.0, lrate=0.5, randomSeed = None):
		self.M1 = m1
		self.M2 = m2
		self.M = m1*m2
		self.N = nEntrada
		self.sigma = sigma
		self.lrate = lrate
		self.vecindadFilas = np.arange(self.M1)
		self.vecindadColumnas = np.arange(self.M2)
		
		np.random.seed(randomSeed)
		self.W = np.random.rand(nEntrada, self.M)
		self.W = np.array([normalizar(v) for v in self.W])
		
		self.XtMenosW = np.zeros((self.N, self.M))
		
		self.lrateInicial = lrate
		self.sigmaInicial = sigma
		self.epoch = 1 #Epocas completas
		self.t = 1 #Iteraciones de entrenamiento
	def pIndexAlMapa(self, index):
		return (index / self.M2, index % self.M2)
		
	def pIndexDesdeMapa(self, x, y):
		return x*self.M2 + y%self.M2
		
	def matrizSigmaCentrada(self, puntoCentro):
		# Devuelve la matriz de factores sigma (aplanada)
		factorSigma = 2.0*(self.sigma**2.0)
		filas = np.exp(-np.power(self.vecindadFilas-puntoCentro[0],2)/factorSigma)
		columnas = np.exp(-np.power(self.vecindadColumnas-puntoCentro[1],2)/factorSigma)
		
		matriz = np.outer(filas, columnas)
		return matriz.reshape(1, self.M)
		
	def activar(self, X):
		# Devuelve el pIndex ganador (jStar)
		
		informar('X', debug)
		informar(str(X), debug)
		
		informar('W', debug)
		informar(str(self.W), debug)
		
		X = X.reshape(1, self.N)
		self.XtMenosW = X.T - self.W
		
		informar('XtMenosW', debug)
		informar(str(self.XtMenosW), debug)
		
		yTilde = [norma2(self.XtMenosW[:, col]) for col in range(self.M)]
		jStar = np.argmin(yTilde)
		
		informar('yTilde', debug)
		informar(str(yTilde), debug)
		
		informar('jStar', debug)
		informar(str(jStar), debug)
		
		return jStar
	
	def corregir(self, jStar):
		# Toma un ganador y corrije los pesos.
		D = self.matrizSigmaCentrada(self.pIndexAlMapa(jStar))
		
		informar('D', debug)
		informar(str(D), debug)
		
		varW = self.lrate * np.multiply(D, self.XtMenosW)
		
		informar('varW', debug)
		informar(str(varW), debug)
		self.W += varW
		
	def actualizarParametrosAdaptativos(self):
		self.t += 1.0
		#self.lrate = 1.0 / self.epoch**2
		#self.sigma = self.M2 / (2.0 * (self.epoch ** 3))
		self.lrate = self.lrateInicial / self.epoch
		self.sigma = self.sigmaInicial / self.epoch
		
	def trainEpoch(self, X):
		jStar = self.activar(X)
		self.corregir(jStar)
		self.actualizarParametrosAdaptativos()
		return self.pIndexAlMapa(jStar)
		
	def trainRandom(self, Xs, maxEpoch):

		while self.epoch <= maxEpoch:
			# Agarro la entrada shuffleada
			indexes = np.arange(len(Xs))
			np.random.shuffle(indexes)
			
			for index in indexes:
				informar('Epoch ' + str(self.epoch) + ' Iteration ' + str(self.t) + ' lrate: ' + str(self.lrate) + ' sigma: ' + str(self.sigma), conteo | info | debug)
				resMapeo = self.trainEpoch(Xs[index])
				informar(str(((self.t - 2) % len(indexes)) + 1) + ' / ' + str(len(indexes)) + ' - ' + str(index) + ' -> ' + str(resMapeo), conteo | info | debug)
			self.epoch += 1
			
	def mapear(self, X):
		res = self.pIndexAlMapa(self.activar(X))
		return res
	