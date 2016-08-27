from numpy import *

def perceptronSimple(trainSet):
	w = [random.uniform(0.1,0.01) for _ in range(2)]
	e = 1
	E = 0.1
	t = 0
	T = 1000
	H = 0.1  #coeficiente de aprendizaje

	while e > E and t < T:
		e = 0
		for x,z in trainSet:
			y = sigmoid(asarray(x).dot(asarray(w)))
			error = z - y
			dw = H*(asarray(x).dot(asarray(error)))
			w += dw
			e += linalg.norm(error)
		t += 1
	return w

def activationFunction(num):
	return sign(num)

def sigmoid(x):
	return 1 / (1 + math.exp(-x))


def main():
	trainOut = [0,1]
	w = perceptronSimple([([0,0],0),([0,1],1),([1,0],1),([1,1],1)])  #[(input,target)]
	print w
	resultado = asarray(w).dot(asarray(trainOut))
	print activationFunction(resultado)

if __name__ == "__main__":
    main()