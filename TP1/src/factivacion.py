import numpy as np

def fHiper(x):
	return np.tanh(x)

def fDerHiper(x):
	return 1.0-x**2
	
def fSigmoidea(x):
	return 1/(1+np.exp(-x))
	
def fDerSigmoidea(x):
	return fSigmoidea(x)*(1-fSigmoidea(x))