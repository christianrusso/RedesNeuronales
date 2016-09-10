from numpy import *
import math
def sigmoidea_bipolar(vector, derivative=False):
	B = 1
	if derivative:
		return B * (1 -sigmoidea_bipolar(vector)**2)
	else:
		return tanh(B*vector)

def sigmoidea_logistica(vector, derivative=False):
	B = 1
	if derivative:
		return B * (1 -sigmoidea_logistica(vector))*sigmoidea_logistica(vector)
	else:
		return 1/(1+exp(-vector))

class perceptron_Multiple:

	def load_dataset_1(self,dataset):
		print "> Cargando dataset..."	
		f = open(dataset)
		self.X = []
		self.Z = []
		for line in f:
			if line.rstrip():
				r = line.rstrip().split(", ")
				x_i = map(float, r[1:])
				z_i = map(self.cod, r[0])
				self.X.append(x_i)
				self.Z.append(z_i)
		# basura en la primera pos pq esta indizado dsese 1
		insert(self.X, 0, [0])
		insert(self.Z, 0, [0])
		self.normalizar_input()

	def load_dataset_2(self,dataset):
		print "> Cargando dataset..."	
		f = open(dataset)
		self.X = []
		self.Z = []
		for line in f:
			if line.rstrip():
				r = line.rstrip().split(", ")
				x_i = map(float, r[:-2])
				z_i = map(float, r[-2:])
				X.append(x_i)
				Z.append(z_i)
		# basura en la primera pos pq esta indizado dsese 1
		insert(self.X, 0, [0])
		insert(self.Z, 0, [0])
		self.normalizar_input()
		self.normalizar_output()
	
	def __init__(self,UnitsXCapa,e,t,nl,funcionActivacion=sigmoidea_bipolar):
		self.funcActivacion = funcionActivacion	
		self.epsilon = e
		self.tau = t
		self.eta = nl
		self.p = 1	
		self.momentum = 0.6
	 	# CANT CAPAS
	 	self.L=len(UnitsXCapa)+2
	 	self.UnitsXCapa=UnitsXCapa
	 	self.Beta=1

	def cod(self,c):
		if c == "M":
			return 1
		else:
			return -1 

	def normalizar_input(self):
		media = mean(self.X, axis= 0) 
		varianza = std(self.X, axis=0)
		for i in xrange(len(self.X)):
			self.X[i] = (self.X[i] - media )/varianza	

	def normalizar_output(self):
		media_z = mean(self.Z, axis= 0) 
		varianza_z = std(self.Z, axis=0)
		for i in xrange(len(self.Z)):
			self.Z[i] = (self.Z[i] - media_z)/varianza_z

	def train(self):
		self.P = len(self.X)-1
		self.S = [1, shape(self.X)[1]]
		#S.extend([15 for x in range(2, L)])
		self.S.extend(self.UnitsXCapa)
		self.S.append(shape(self.Z)[1])
		# TAMANOS W, dW, Y
		self.W = [random.uniform(-sqrt(self.S[j]),sqrt(self.S[j]), (self.S[j-1]+1, self.S[j])) for j in range(0, self.L+1)]
		self.dW = [zeros((self.S[j-1]+1, self.S[j])) for j in range(0, self.L+1)]
		self.Y = [zeros((1, self.S[j]+1)) for j in range(0, self.L)]
		self.Y.append([zeros((1,shape(self.Z)[1]))])
		t,error_v_hist,error_t_hist = self.holdout(self.epsilon, self.tau, self.p)
		print error_t_hist[-1],error_v_hist[-1] ,t 
		return error_v_hist,error_t_hist,t

	def holdout(self,epsilon, tau, p):
		e_t = 1
		e_v = 1
		t = 0
		v = int(p*self.P)
		error_v_hist=[]
		error_t_hist=[]
		while(t<self.tau and e_t > self.epsilon):
			
			e_t = self.incremental(self.X[:v+1],self.Z[:v+1])
			e_v = self.testing(self.X[v:],self.Z[v:])
			error_v_hist.append(e_v)
			error_t_hist.append(e_t)
			t += 1
			if(t % 10==1):
				print "epoch", t, "   e_training", e_t, "	e_validation", e_v
		return t,error_v_hist,error_t_hist

	def incremental(self,X, Z):
		e = 0
		for h in range(1, self.P+1): 
			self.activation(X[h])
			e += self.correction(Z[h])
			self.adaptation()
		return e/len(X) 	

	def activation(self,X_h):
		self.Y[1] = append(X_h, [-1])[newaxis]
		for j in range(2, self.L+1):
			if j == self.L:
				self.Y[j] = self.funcActivacion(dot(self.Y[j-1], self.W[j]))
			else:
				self.Y[j] = append(self.funcActivacion(dot(self.Y[j-1], self.W[j])), [-1])[newaxis]
		return self.Y[self.L]

	def correction(self,Z_h):
		E = Z_h - self.Y[self.L]
		e = (E**2).sum()
		for j in range(self.L, 1, -1):   
			D = E*self.funcActivacion(dot(self.Y[j-1], self.W[j]), True)
			self.dW[j] = ((self.eta*dot(transpose(self.Y[j-1]), D)) + self.momentum*self.dW[j])
			# El error no tiene sentido q tenga el -1 del final
			E = dot(D, transpose(self.W[j]))[0][:-1]
		return e

	def adaptation(self):
		for j in range(2, self.L+1):
			self.W[j] += self.dW[j]

	def testing(self,X, Z):
		e = 0
		for (X_h, Z_h) in zip(self.X, self.Z):
			E = self.activation(X_h)-Z_h
			e += square(linalg.norm(E))
		return e/len(self.X) 
