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
		self.normalizar_input()
		self.X = array(self.X) # Para pretty print
		self.Z = array(self.Z)
		
	def load_dataset_2(self,dataset):
		print "> Cargando dataset..."	
		f = open(dataset)
		self.X = []
		self.Z = []
		for line in f:
			if line.rstrip():
				r = line.rstrip().split(",	")
				x_i = map(float, r[:-2])
				z_i = map(float, r[-2:])
				self.X.append(x_i)
				self.Z.append(z_i)
		self.normalizar_input()
		self.normalizar_output()
		self.X = array(self.X) # Para pretty print
		self.Z = array(self.Z)

	def __init__(self,UnitsXCapa=[],e=0,t=0,nl=0,m=0.6,holdout=1,funcionActivacion=sigmoidea_bipolar):
		self.funcActivacion = funcionActivacion	
		self.epsilon = e
		self.tau = t
		self.eta = nl
		self.p = holdout
		self.momentum = m
	 	# CANT CAPAS
	 	self.L=len(UnitsXCapa)+2
	 	self.UnitsXCapa=UnitsXCapa
	 	self.UnitsXCapa.append(5)
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

	def train(self,modo=0):
		#print self.
		self.P = len(self.X)
		self.S = [shape(self.X)[1]]
		#S.extend([15 for x in range(2, L)])
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
		if modo==1:
			print "corriendo en modo incremental"
			t,error_v_hist,error_t_hist = self.holdout(self.epsilon, self.tau, self.p)
		else:
			print "corriendo en modo batch"
			t,error_v_hist,error_t_hist = self.holdout_batch(self.epsilon, self.tau, self.p)
		print error_t_hist[-1],error_v_hist[-1] ,t 
		return error_t_hist,error_v_hist,t

	def holdout(self,epsilon, tau, p):
		e_t = 1
		e_v = 1
		t = 0
		v = int(p*self.P)
		error_v_hist=[]
		error_t_hist=[]
		while(t<self.tau and e_t > self.epsilon):
			
			e_t = self.incremental(self.X[:v],self.Z[:v])
			e_v = self.testing(self.X[v:],self.Z[v:])
			error_v_hist.append(e_v)
			error_t_hist.append(e_t)
			t += 1
			if(t % 10==1):
				print "epoch", t, "   e_training", e_t, "	e_validation", e_v
		return t,error_v_hist,error_t_hist

	def holdout_batch(self,epsilon, tau, p):
		e_t = 1
		e_v = 1
		t = 0
		v = int(p*self.P)
		error_v_hist=[]
		error_t_hist=[]
		while(t<self.tau and e_t > self.epsilon):
			
			e_t = self.batch(self.X[:v],self.Z[:v])
			e_v = self.testing(self.X[v:],self.Z[v:])
			error_v_hist.append(e_v)
			error_t_hist.append(e_t)
			t += 1
			if(t % 10==1):
				print "epoch", t, "   e_training", e_t, "	e_validation", e_v
		return t,error_v_hist,error_t_hist


	def incremental(self,X, Z):
		e = 0
		for h in range(len(X)): 
			self.activation(X[h])
			e += self.correction(Z[h])
			self.adaptation()
		return e/(len(X) if len(X) != 0 else 1) 	

	def batch(self,X, Z):
		e=0
		for h in range(len(X)): 
			self.activation(X[h])
			e += self.correction(Z[h])
		self.adaptation()
		return e/(len(X) if len(X) != 0 else 1) 	

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
			self.dW[j] = ((self.eta*dot(transpose(self.Y[j-1]), D)) + self.momentum*self.dW[j])
			# El error no tiene sentido q tenga el -1 del final
			E = dot(D, transpose(self.W[j]))[0][:-1]
		return e

	def adaptation(self):
		for j in range(1, self.L):
			self.W[j] += self.dW[j]

	def testing(self,X, Z):
		e = 0
		for (X_h, Z_h) in zip(X, Z):
			E = self.activation(X_h)-Z_h
			e += (E**2).sum()
		return e/(len(X) if len(X) != 0 else 1) 

	def evaluate(self,input,ejercicio):	
		e=[10]
		res=[]
		# error_v_hist=[]
		# for i range()
		# 	e_v = self.testing(self.X[v:],self.Z[v:])
		# 	error_v_hist.append(e_v)
			
		#for Y_h in input:
		#	res.append(self.activation(Y_h))
		return e,e,res	

	def save(self,file):
		f = open(file, 'w')
		f.write('Net Save ...\n')
		f.write(str(self.funcActivacion)+"\n")
		f.write(str(self.epsilon)+"\n")
		f.write(str(self.tau)+"\n")
		f.write(str(self.eta)+"\n")
		f.write(str(self.p)+"\n")
		f.write(str(self.momentum)+"\n")
		f.write(str(self.L)+"\n")
		f.write(str(self.UnitsXCapa)+"\n")
		f.write(str(self.Beta)+"\n")
		f.write(str(self.W)+"\n")
		f.write(str(self.dW))
		f.close()

	def load(self,file):
		f=open(file)
		self.funcActivacion = f.read()	
		self.epsilon = f.read()
		self.tau = f.read()
		self.eta = f.read()
		self.p = f.read()
		self.momentum = f.read()
	 	self.L=f.read()
	 	self.UnitsXCapa=f.read()
	 	self.Beta=f.read()
	 	self.W=f.read()
	 	self.dw=f.read()
		f.close()


