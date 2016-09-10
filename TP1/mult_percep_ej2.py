#!/usr/bin/python

from numpy import *

def main():
	global X, Z, L, W, dW, eta, P, Y, S, momentum
	epsilon = 0.1
	tau = 1000
	eta = 0.05
	p = 1
	momentum = 0.5
	
	f = open('./datasets/tp1_ej2_training.csv')
	X = []
	Z = []
	for line in f:
		if line.rstrip():
			r = line.rstrip().split(",	")
			x_i = map(float, r[:-2])
			z_i = map(float, r[-2:])
			X.append(x_i)
			Z.append(z_i)
	# basura en la primera pos pq esta indizado dsese 1
	insert(X, 0, [0])
	insert(Z, 0, [0])
	# print Z
	media_x = mean(X, axis= 0) 
	varianza_x = std(X, axis=0)
	media_z = mean(Z, axis= 0) 
	varianza_z = std(Z, axis=0)
	for i in xrange(len(X)):
		X[i] = (X[i] - media_x)/varianza_x
		Z[i] = (Z[i] - media_z)/varianza_z
	# Para pretty print
	X = array(X)
	Z = array(Z)

	# CANT PATRONES
	# -1 pq x esta indizado desde 1
	P = len(X)-1
	# CANT CAPAS
	L = 3
	# UNIDADES POR CAPA 
	# basura en la primera pos pq esta indizado dsese 1
	S = [1, shape(X)[1]]
	S.extend([5 for x in range(2, L)])
	S.append(shape(Z)[1])
	# TAMANOS W, dW, Y
	W = array([random.uniform(-sqrt(S[j]),sqrt(S[j]), (S[j-1]+1, S[j])) for j in range(0, L+1)])
	dW = array([zeros((S[j-1]+1, S[j])) for j in range(0, L+1)])
	Y = [zeros((1, S[j]+1)) for j in range(0, L)]
	Y.append([zeros((1,shape(Z)[1]))])
	# Y = array(Y) no funciona pq Y no tiene columnas iguales por fila
	e_t, e_v, t = holdout(epsilon, tau, p)
	print e_t, e_v, t 

def holdout(epsilon, tau, p):
	e_t = 1
	e_v = 1
	t = 0
	v = int(p*P)
	while(t<tau and e_t > epsilon):
		print "epoch", t, "   e_training", e_t, "	e_validation", e_v
		e_t = incremental(X[:v+1],Z[:v+1])
		##### CAMBIAR CUANDO SE ARREGLE OUTPUT Y MAIN
		# deberia ser v+1, pero si es vacio revienta
		e_v = testing(X[v:],Z[v:])
		t += 1
	return e_t, e_v, t 

def incremental(X, Z):
	e = 0
	for h in range(1, P+1): 
		activation(X[h])
		e += correction(Z[h])
		adaptation()
	return e/len(X) 	

def activation(X_h):
	global Y
	Y[1] = append(X_h, [-1])[newaxis]
	for j in range(2, L+1):
		if j == L:
			Y[j] = activationFunction(dot(Y[j-1], W[j]))
		else:
			Y[j] = append(activationFunction(dot(Y[j-1], W[j])), [-1])[newaxis]
	return Y[L]

def correction(Z_h):
	global dW
	E = Z_h - Y[L]
	e = sum(E**2)
	for j in range(L, 1, -1):   
		D = E*activationFunction(dot(Y[j-1], W[j]), True)
		dW[j] = ((eta*dot(transpose(Y[j-1]), D)) + momentum*dW[j])
		# El error no tiene sentido q tenga el -1 del final
		E = dot(D, transpose(W[j]))[0][:-1]
	return e

def adaptation():
	global W
	for j in range(2, L+1):
		W[j] += dW[j]

def testing(X, Z):
	e = 0
	for (X_h, Z_h) in zip(X, Z):
		E = activation(X_h)-Z_h
		e += sum(E**2)
	return e/len(X) 

def bipolar(vector, derivative=False):
	B = 1
	if derivative:
		return B * (1 - bipolar(vector)**2)
	else:
		return tanh(B*vector)

# def bipolarSigmoid(vector, derivative=False):
# 	 #f(x) = -1 + 2 / (1 + e^-x)
# 	B = 1
# 	if derivative:
# 		return 0.5 * (1 + bipolarSigmoid(vector)) * (1 - bipolarSigmoid(vector) )
# 	else:
# 		return -1 + 2 / (1 + exp(-vector))

def activationFunction(vector, derivative=False, f=bipolar):
	return f(vector, derivative)
	
if __name__ == "__main__":
	main()