from numpy import *

def activation(X_h):
	global Y
	Y[1] = X_h
	for j in range(2, L+1):
		Y[j] = activationFunction(dot(Y[j-1], W[j]))
	return Y[L]

def adaptation():
	global W
	global dW
	for j in range(2, L+1):
		W[j] += dW[j]
		dW[j] = 0 

def correction(Z_h):
	global dW 
	E = Z_h - Y[L]
	e = square(linalg.norm(E))
	for j in range(L, 1, -1):
		D = E*activationFunction(dot(Y[j-1], W[j]), True)
		dW[j] += eta*dot(D, Y[j-1])
		E = dot(D, transpose(W[j]))
	return e 

def sigmoid(num, derivative=False):
	B = 1
	if derivative:
		return B * sigmoid(num) * (1 - sigmoid(num))
	else:
		return 1 / (1 + math.exp(-B*num))

def activationFunction(num, derivative=False, f=sigmoid):
	return f(num, derivative)

def incremental(X, Z):
	e = 0
	for h in range(1, P+1):
		activation(X[h])
		e += correction(Z[h])
		adaptation()
	return e 

def holdout(epsilon, tau, p):
	e_t = 1
	e_v = 1
	t = 0
	v = int(P-p)
	while(e_t>epsilon and t<tau):
		e_t = incremental(X[:v],Z[:v])
		e_v = testing(X[v:],Z[v:])
		t += 1
	return e_t, e_v, t 	

def testing(X, Z):
	e = 0
	for (X_h, Z_h) in zip(X, Z):
		E = activation(X_h)-Z_h
		e += linalg.norm(E)^2
	return e 

def main():
	global X, Z, L, W, dW, eta, P, Y
	P = 4
	X = [[0,0], [0,1], [1,0], [1,1]]
	Z = [0, 1, 1, 1]
	L = 2
	W = [random.uniform(-1/sqrt(2),1/sqrt(2), (2, 1)) for x in range(L+1)]
	dW = W
	Y = [random.uniform(-1/sqrt(2),1/sqrt(2), (1, 2)) for x in range(L+1)]
	eta = 0.01
	e_t, e_v, t = holdout(0.1, 1000, 0.75)
	print e_t, e_v, t 
	

if __name__ == "__main__":
    main()

def perceptronSimple(trainSet):
	w = random.uniform(0.01, 0.1, (5,25))
	e = 1
	E = 0.1
	i = 0
	I = 1000
	eta = 0.1  #coeficiente de aprendizaje

	while e > E and i < I:
		e = 0
		for x,z in trainSet:
			h = dot(w,x)
			O = activationFunction(h, derivative=True)
			delta = z - O
			dw = eta*dot(delta,transpose(x))
			w += dw
			e += linalg.norm(delta)
		i += 1
	return w