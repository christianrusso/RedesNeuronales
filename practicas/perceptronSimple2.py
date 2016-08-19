from numpy import *

def perceptronSimple(trainSet):
	w = [[random.uniform(0.1,0.01) for _ in range(25)] for _ in range(5)]
	
	e = 1
	E = 0.1
	t = 0
	T = 1000
	H = 0.1  #coeficiente de aprendizaje

	while e > E and t < T:
		e = 0
		for x,z in trainSet:
			xs = array(x)[newaxis].T
			y = activationFunction(dot(w,xs))
			z = array(z)[newaxis].T
			error = z - y
			dw = H*dot(error,matrix(x))
			print dw
			w += dw
			e += linalg.norm(error)
		t += 1
	return w

def activationFunction(num):
	return sign(num)

def sigmoid(x):
	return 1 / (1 + math.exp(-x))


def main():
	trainOut = [random.uniform(0.1,0.01) for _ in range(25)]
	w = perceptronSimple([([random.uniform(0.1,0.01) for _ in range(25)],[random.uniform(0.1,0.01) for _ in range(5)])])  #[(input,target)]
	
	resultado = dot(w,trainOut)
	print activationFunction(resultado)

if __name__ == "__main__":
    main()