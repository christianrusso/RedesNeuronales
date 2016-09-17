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

	def load_dataset(self,dataset, ejercicio):
		# print "> Cargando dataset..."	
		f = open(dataset)
		self.X = []
		self.Z = []
		for line in f:
			if line.rstrip():
				if ejercicio == 1:
					r = line.rstrip().split(", ")
					x_i = map(float, r[1:])
					z_i = map(self.cod, r[0])
				else:
					r = line.rstrip().split(",	")
					x_i = map(float, r[:-2])
					z_i = map(float, r[-2:])
				self.X.append(x_i)
				self.Z.append(z_i)
		self.normalizar(self.X)
		if ejercicio != 1:
			self.normalizar(self.Z)
		self.X = array(self.X) # Para pretty print
		self.Z = array(self.Z)

	def __init__(self,UnitsXCapa=[15],e=0,t=0,nl=0,m=0.6,holdout=1,funcionActivacion=sigmoidea_bipolar):
		self.funcActivacion = funcionActivacion	
		self.epsilon = e
		self.tau = t
		self.eta = nl
		self.p = holdout
		self.momentum = m
	 	# CANT CAPAS
	 	self.L=len(UnitsXCapa)+2
	 	self.UnitsXCapa=UnitsXCapa
	 	self.Beta=1

	def cod(self,c):
		if c == "M":
			return 1
		else:
			return -1 

	def normalizar(self, array):
		media = mean(array, axis= 0) 
		varianza = std(array, axis=0)
		for i in xrange(len(array)):
			array[i] = (array[i] - media )/varianza	

	def train(self,modo=0, early=0):
		self.P = len(self.X)
		self.S = [shape(self.X)[1]]
		self.S.extend(self.UnitsXCapa)
		self.S.append(shape(self.Z)[1])
		self.S = array(self.S)
		# TAMANOS W, dW, Y
		# Basura en la pos 0, indizado desde 1
		self.W = array([random.uniform(-sqrt(self.S[j]),sqrt(self.S[j]), (self.S[j-1]+1, self.S[j])) for j in range(self.L)])
		# Basura en la pos 0, indizado desde 1
		self.dW = array([zeros((self.S[j-1]+1, self.S[j])) for j in range(self.L)])
		self.Y = [zeros((1, self.S[j]+1)) for j in range(self.L-1)]
		self.Y.append([zeros((1,shape(self.Z)[1]))])
		t,error_v_hist,error_t_hist, error_t_hist_sum, error_v_hist_sum = self.holdout(self.epsilon, self.tau, self.p, modo, early)
		# print error_t_hist[-1],error_v_hist[-1] ,t 
		return error_t_hist,error_v_hist, error_v_hist_sum, error_t_hist_sum, t

	def holdout(self,epsilon, tau, p, modo, early):
		e_t = 1
		e_v = 1
		t = 0
		v = int(p*self.P)
		error_v_hist=[]
		error_t_hist=[]
		error_v_hist_sum=[]
		error_t_hist_sum=[]
		early_count = 0
		while(t<self.tau and e_t > self.epsilon):
			# if t == 0 and modo:
			# 	print "Modo incremental"
			# elif t == 0 and not modo:
			# 	print "Modo batch"			
			e_t, e_t_sum = self.training(self.X[:v],self.Z[:v], modo)
			e_v, e_v_sum = self.testing(self.X[v:],self.Z[v:])
			error_v_hist.append(e_v)
			error_t_hist.append(e_t)
			error_v_hist_sum.append(e_v_sum)
			error_t_hist_sum.append(e_t_sum)
			t += 1
			#early_count = (early_count+1) if t > 2 and error_v_hist[-1]>error_v_hist[-2] else 0
			# if(t % 10==1):
			# 	print "epoch", t, "   e_training", e_t, "	e_validation", e_v
			# if early and early_count>=30:
			# 	print "Early Stopping - 30 epochs de crecimiento de error de validacion"
			# 	break
		return t,error_v_hist,error_t_hist, error_t_hist_sum, error_v_hist_sum

	def training(self,X, Z, modo):
		e_count = 0
		e_sum = 0
		for h in range(len(X)): 
			self.activation(X[h])
			e = self.correction(Z[h])
			if e > self.epsilon:
				e_count += 1
			e_sum += e
			if modo:
				self.adaptation()
		if not modo:
			self.adaptation()	
		return e_sum/(len(X) if len(X) != 0 else 1), e_count

	def activation(self,X_h):
		self.Y[0] = append(X_h, [-1])[newaxis]
		for j in range(1, self.L):
			if j == self.L-1:
				self.Y[j] = self.funcActivacion(dot(self.Y[j-1], self.W[j]))
			else:
				self.Y[j] = append(self.funcActivacion(dot(self.Y[j-1], self.W[j])), [-1])[newaxis]
		return self.Y[-1]

	def correction(self,Z_h):
		E = Z_h - self.Y[-1]
		e = (E**2).sum()
		for j in range(self.L-1, 0, -1):   
			D = E*self.funcActivacion(dot(self.Y[j-1], self.W[j]), True)
			self.dW[j] += (self.eta*dot(transpose(self.Y[j-1]), D)) 
			# El error no tiene sentido q tenga el -1 del final
			E = dot(D, transpose(self.W[j]))[0][:-1]
		return e

	def adaptation(self):
		for j in range(1, self.L):
			self.W[j] += self.dW[j] 
			self.dW[j] *= self.momentum

	def testing(self,X, Z):
		e_count = 0
		e_sum = 0
		for (X_h, Z_h) in zip(X, Z):
			E = self.activation(X_h)-Z_h
			e = (E**2).sum()
			if e > self.epsilon:
				e_count += 1
			e_sum += e
		return e_sum/(len(X) if len(X) != 0 else 1), e_count

	def evaluate(self):	
		e_sum = 0
		list_error=[]
		numero_error=0
		for (X_h, Z_h) in zip(self.X, self.Z):
			E=self.activation(X_h)-Z_h
			if((E**2).sum()>=1):
				numero_error+=1
			e_sum+=(E**2).sum()
			list_error.append((E**2).sum())
		return list_error,e_sum/(len(self.X) if len(self.X) != 0 else 1),numero_error	




