from numpy import *

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
			O = activationFunction(h)
			delta = z - O
			dw = eta*dot(delta,transpose(x))
			w += dw
			e += linalg.norm(delta)
		i += 1
	return w

def activationFunction(num):
	return sign(num)

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

def main():
	w = perceptronSimple([(random.uniform(0.1,0.01,(25,1)),random.uniform(0.1,0.01, (5,1)))])  #[(input,target)]
	
	y = random.uniform(0.1,0.01,(25, 1))
	prediccion = dot(w,y)
	print activationFunction(prediccion)

if __name__ == "__main__":
    main()