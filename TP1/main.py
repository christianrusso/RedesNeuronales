from new_perceptron import perceptron_Multiple,sigmoidea_bipolar,sigmoidea_logistica
from numpy import *
#import matplotlib.pyplot as plt
from pylab import Line2D, plot, axis, show, pcolor, colorbar, bone, savefig
import pylab as plt
import sys

def test_1(Dataset=None, unitsPorCapa=[15], epsilon=0.1, tau=1000, etha=0.01):
	print "Ejercicio 1"	
	print "Unidades por capa " + str(unitsPorCapa)
	print "Error aceptable " + str(epsilon)
	print "Max Epocas " +str(tau)
	print "Learning Rate " + str(etha)
	Red=perceptron_Multiple(unitsPorCapa,epsilon,tau,etha)
	Red.load_dataset_1(Dataset)
	erroresTraining, erroresTesting, epoch = Red.train()
	print '>> epochAlcanzada: ' + str(epoch)
	print '>> errorTraining: ' + str(erroresTraining[-1])
	print '>> errorTesting: ' + str(erroresTesting[-1])	
	plt.plot(erroresTraining)
	plt.ylabel('error Training Vs epocas')
	plt.show()
	return mean(erroresTraining), mean(erroresTesting)

def test_2(Dataset=None, unitsPorCapa=[15], epsilon=0.1, tau=1000, etha=0.01):
	print "Ejercicio 2"
	print "Unidades por capa " + str(unitsPorCapa)
	print "Error aceptable " + str(epsilon)
	print "Max Epocas " +str(tau)
	print "Learning Rate " + str(etha)
	Red=perceptron_Multiple(unitsPorCapa,epsilon,tau,etha)
	Red.load_dataset_2(Dataset)
	erroresTraining, erroresTesting, epoch = Red.train()
	print '>> epochAlcanzada: ' + str(epoch)
	print '>> errorTraining: ' + str(erroresTraining[-1])
	print '>> errorTesting: ' + str(erroresTesting[-1])	
	plt.plot(erroresTraining)
	plt.ylabel('error Training Vs epocas')
	plt.show()
	return mean(erroresTraining), mean(erroresTesting)

def pruebas():
	print "Perceptron Multiple Mark XLV"
	#test_1('./datasets/tp1_ej1_training.csv')						 #anda mas o menos estable, pero las epocas de aprendizaje varian mucho de 35 a 300 epocas tipicamente
	#test_1('./datasets/tp1_ej1_training.csv',[5], 0.1,1000, 0.05)	  #anda muy mal nunca termina de aprender
	#test_1('./datasets/tp1_ej1_training.csv',[15,10], 0.1,1000, 0.01)#no mejora la de 15, estabiliza alrededor de 300 epocas, aprende en rango de 300 a 1000
	#test_1('./datasets/tp1_ej1_training.csv',[10,5], 0.1,1000, 0.01) #parece razonable, oscila mas pero llega mas rapido. En alrededor de 100 epocas siempre termina
	#test_1('./datasets/tp1_ej1_training.csv',[5,5,5], 0.1,1000, 0.01)#oscila horriblemente y es mas lenta que la anterior
	#test_1('./datasets/tp1_ej1_training.csv',[8,4,3], 0.1,1000, 0.01) #oscila con picos, aprende alrededor de las 300 epocas
	#test_1('./datasets/tp1_ej1_training.csv',[8,4,3], 0.1,1000, 0.05) #nunca termina de aprender, se estanca el error y baja muy lento
	#test_1('./datasets/tp1_ej1_training.csv',[10,5], 0.1,1000, 0.05) # no mejora con rate aumentado
	#test_1('./datasets/tp1_ej1_training.csv',[10,5], 0.1,1000, 0.01)
	test_1('./datasets/tp1_ej1_training.csv',[10,5], 0.1,100, 0.01)

	#test_2('./datasets/tp1_ej2_training.csv')						 
	#test_2('./datasets/tp1_ej2_training.csv',[5], 0.1,1000, 0.05)	  
	#test_2('./datasets/tp1_ej2_training.csv',[15,10], 0.1,1000, 0.01)
	#test_2('./datasets/tp1_ej2_training.csv',[10,5], 0.1,1000, 0.01) 
	#test_2('./datasets/tp1_ej2_training.csv',[5,5,5], 0.1,1000, 0.01)
	#test_2('./datasets/tp1_ej2_training.csv',[8,4,3], 0.1,1000, 0.01)
	#test_2('./datasets/tp1_ej2_training.csv',[8,4,3], 0.1,1000, 0.05)
	#test_2('./datasets/tp1_ej2_training.csv',[10,5], 0.1,1000, 0.05)
	#test_2('./datasets/tp1_ej2_training.csv',[10,5], 0.1,1000, 0.01)

args = sys.argv
message = "\nModo de uso:\n\
python main.py (ej1|ej2) -t nomArchivoData parametros\n\
Con los siguientes parametros en orden:\n\
unidadesXcapa epsilon tau etha\n\n"

print len(args)
if len(args)==1:
	pruebas()
elif len(args) < 5:
	print message
	sys.exit()
else:
	cmdEj1 = args[1] == "ej1"
	cmdEj2 = args[1] == "ej2"
	cmdTrain = args[2] == "-t"
	cmdLoad = args[2] == "-l"

if cmdEj1 and cmdTrain:
	
	if len(args) != 11:
		print "\nIncorrecta cantidad de argumentos para entrenar."
		print message
		sys.exit()
		
	archivoDataset = args[3]
	unidadesCapaOculta = int(args[5])
	errorAceptable = float(args[6])
	maxEpoch = int(args[7])
	holdoutRate = float(args[8])
	learningRate = float(args[9])
	momentum = float(args[10])
	
	ej1(archivoDataset, archivoRed, None, None, [30, unidadesCapaOculta, 1], errorAceptable, maxEpoch, holdoutRate, learningRate, momentum)
	
elif cmdEj1 and cmdLoad:
	
	if len(args) != 5:
		print "\nIncorrecta cantidad de argumentos para entrenar."
		print message
		sys.exit()
		
	archivoRed = args[3]
	archivoDataset = args[4]
	
	ej1(None, None, archivoRed, archivoDataset)
	
elif cmdEj2 and cmdTrain:
	
	if len(args) != 11:
		print "\nIncorrecta cantidad de argumentos para entrenar."
		print message
		sys.exit()
		
	archivoDataset = args[3]
	archivoRed = args[4]
	unidadesCapaOculta = int(args[5])
	errorAceptable = float(args[6])
	maxEpoch = int(args[7])
	holdoutRate = float(args[8])
	learningRate = float(args[9])
	momentum = float(args[10])
	
	ej2(archivoDataset, archivoRed, None, None, [8, unidadesCapaOculta, 2], errorAceptable, maxEpoch, holdoutRate, learningRate, momentum)
	
elif cmdEj2 and cmdLoad:
	
	if len(args) != 5:
		print "\nIncorrecta cantidad de argumentos para entrenar."
		print message
		sys.exit()
		
	archivoRed = args[3]
	archivoDataset = args[4]

	ej2(None, None, archivoRed, archivoDataset)
	
else:
	print message