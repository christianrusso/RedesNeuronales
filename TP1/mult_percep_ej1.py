#!/usr/bin/python

from numpy import *

def cod(c):
	if c == "M":
		return 1
	else:
		return 0

def main():
	global X, Z, L, W, dW, eta, P, Y, S
	epsilon = 0.1
	tau = 1000
	eta = 0.01
	p = 1
	
	f = open('./datasets/tp1_ej1_training.csv')
	X = []
	Z = []
	for line in f:
		if line.rstrip():
			# print line #debug
			r = line.rstrip().split(", ")
			# print r #debug
			x_i = map(float, r[1:])
			z_i = map(cod, r[0])
			X.append(x_i)
			Z.append(z_i)
	X = array(X)
	# print X[0] #debug
	# print Z[0] #debug

	# XOR DE DOS VARIABLES 
	# X = array([[-1,-1], [0,1], [1,0], [1,1], [0, 0]])
	# Z = [[-1], [1], [1], [1], [0]]
	
	# CANT PATRONES
	P = len(X)-1
	# CANT CAPAS
	L = 5
	# UNIDADES POR CAPA 
	#PREG: NO ENTIENDO ESTO, que son unidades por capa?
	# Unidades = Neuronas
	S = [0, shape(X)[1]]
	S.extend([L*2 for x in range(2, L)])
	S.append(shape(Z)[1])
	
	# TAMANOS W, dW, Y
	W = [random.uniform(-1/sqrt(2),1/sqrt(2), (S[j-1], S[j])) for j in range(0, L+1)]
	dW = copy(W)
	Y = [random.uniform(-1/sqrt(2),1/sqrt(2), (1, S[j]+1))[newaxis] for j in range(0, L)]
	Y.append([random.uniform(-1/sqrt(2),1/sqrt(2),(1,shape(Z)[1]))[newaxis]])
	
	# print L, S #debug
	# print P  #debug
	# print shape(X), shape(Z) #debug
	# print shape(W), shape(Y)  #debug
	
	e_t, e_v, t = holdout(epsilon, tau, p)
	print e_t, e_v, t 

def holdout(epsilon, tau, p):
	e_t = 1
	e_v = 1
	t = 0
	v = int(p*P)
	while(e_t>epsilon and t<tau):
		print "epoch", t #debug 
		print "error", e_t  #debug
		# print X[:v+1] 
		# print X[v+1:]
		e_t = incremental(X[:v+1],Z[:v+1])
		e_v = testing(X[v+1:],Z[v+1:])
		t += 1
	return e_t, e_v, t 


def incremental(X, Z):
	global W 
	e = 0
	for h in range(1, len(X)): 
		#PREG: porque usas range? si en la clase puso "permutaciones"?
		# Pq no estaba seguro si era una sola permutacion o que
		# si es una sola, es lo mismo q hacer shuffle 
		# si no, no tengo idea
		# print h
		#print shape(X) #debug 
		activation(X[h])
		e += correction(Z[h])
		adaptation()
	return e 	

def activation(X_h):
	global Y
	Y[1] = X_h[newaxis]
	for j in range(2, L+1):
		#print "W[j]: ", W[j] #debug 
		Y[j] = activationFunction(dot(Y[j-1], W[j]))
		#print "j: ", j #debug 
		#print " Y[j]: ", Y[j] #debug 
	return Y[L]

def correction(Z_h):
	global dW
	E = Z_h - Y[L]
	e = square(linalg.norm(E))
	for j in range(L, 1, -1):   
		#PREG: Porque rang de L a 1, yo tengo anotado <L..2>
		# -1 = 1 step negativo = ir para atras 
		# print "j: ", j #debug 
		# print "S[j]: ", S[j] #debug 
		# print "S[j-1]", S[j-1] #debug 
		D = E*activationFunction(dot(Y[j-1], W[j]), True)
		# print "W_j: ", W[j] #debug 
		# print "Y_j-1: ", Y[j-1] #debug 
		# print "dW[j]: ", dW[j] #debug 
		# print "D: ", D #debug 

		dW[j] += eta*dot(transpose(Y[j-1]), D)
		# print dW[j] #debug 
		E = dot(D, transpose(W[j]))
	return e 

def adaptation():
	global W
	global dW
	for j in range(2, L+1):
		W[j] += dW[j]
		#print "j: ", j #debug 
		#print "W[j]: ", W[j] #debug 
		dW[j] = 0 
		#print "W[j]: ", W[j] #debug 

def testing(X, Z):
	e = 0
	for (X_h, Z_h) in zip(X, Z):
		E = activation(X_h)-Z_h
		e += square(linalg.norm(E))
	return e 

def logistica(vector, derivative=False):
	B = 1
	if derivative:
		return B * (1 - square(logistica(vector)))
	else:
		return tanh(B*vector)

def activationFunction(vector, derivative=False, f=logistica):
	return f(vector, derivative)


# No anda pq no se puede aplicar math.exp a un vector
# def sigmoid(vector, derivative=False):
# 	B = 1
# 	if derivative:
# 		return B * sigmoid(vector) * (1 - sigmoid(vector))
# 	else:
# 		print -B*vector
# 		return 1 / (1 + math.exp(-B*vector))



	
if __name__ == "__main__":
	main()