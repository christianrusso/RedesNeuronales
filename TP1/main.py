from new_perceptron import perceptron_Multiple,sigmoidea_bipolar,sigmoidea_logistica
from numpy import *
#import matplotlib.pyplot as plt
from sklearn.grid_search import ParameterGrid
from pylab import Line2D, plot, axis, show, pcolor, colorbar, bone, savefig
from timeit import default_timer as timer 

import pylab as plt
import sys
import cPickle
import datetime

def imprimirImagen(ejercicio, error_t_hist,error_v_hist, suma, img_name, prnt=False, save=True):
	plt.xlabel('Epoch')
	plt.title("Ejercicio "+str(ejercicio))
	plt.plot(error_t_hist, label="Training")
	if suma:
		img_name += "_sum.png"
		plt.ylabel('Suma de errores')
	else:
		img_name += "_mean.png"
		plt.ylabel('Promedio de errores')
	plt.plot(error_v_hist, label="Validacion")
	plt.grid()
	plt.legend()
	if prnt:
		plt.show()
	if save:
		plt.savefig(img_name)
	plt.close()

def train(ejercicio,Dataset=None, save_file=None, epsilon=0.1, tau=1000, etha=0.01,m=0.6,holdoutRate=0.5, modo=0, early=0,unitsPorCapa=[15], wrapper=False):
	if not wrapper:			
		print "Ejercicio ",ejercicio	
		print "File "+str(Dataset)
		print "Unidades por capa " + str(unitsPorCapa)
		print "Error aceptable " + str(epsilon)
		print "Max Epocas " +str(tau)
		print "Learning Rate " + str(etha)
	Red=perceptron_Multiple(unitsPorCapa,epsilon,tau,etha,m,holdoutRate)
	Red.load_dataset(Dataset, ejercicio)
	i = 1
	best_error = -1
	if wrapper:
		error_t_hist_best = None
		error_v_hist_best = None
		error_v_hist_sum_best = None
		error_t_hist_sum_best = None
		i = 3
	for _ in xrange(i):
		error_t_hist,error_v_hist, error_v_hist_sum, error_t_hist_sum, epoch = Red.train(modo, early)
		if best_error == -1 or (wrapper and error_v_hist_sum[-1] < best_error):
			best_error = error_v_hist_sum[-1]
			error_t_hist_best = error_t_hist
			error_v_hist_best = error_v_hist
			error_v_hist_sum_best = error_v_hist_sum
			error_t_hist_sum_best = error_t_hist_sum
	if not wrapper:
		print '>> epochAlcanzada: ' + str(epoch)
		print '>> error promedio Training:	' + str(error_t_hist_best[-1])
		print '>> error promedio Testing:	' + str(error_v_hist_best[-1])
		print '>> suma de errores Training:	' + str(error_t_hist_sum_best[-1])
		print '>> suma de errores Testing:	' + str(error_v_hist_sum_best[-1])	
	img_name= "ej"+str(ejercicio)+"_"+str(etha)+"_"+str(m)+"_"+str(modo)+"_"+str(unitsPorCapa)	
	imprimirImagen(ejercicio, error_t_hist_best,error_v_hist_best, suma=False, img_name=img_name)
	imprimirImagen(ejercicio, error_v_hist_sum_best, error_t_hist_sum_best, suma=True, img_name=img_name)
	
	if(save_file!=None):
		print "Guardando Red"
		with open(save_file, "wb") as output:
			cPickle.dump(Red, output, cPickle.HIGHEST_PROTOCOL)
	return error_t_hist_sum[-1], error_v_hist_sum[-1]

def load(ej,Dataset, Net, prnt=True):
	print "Cargando Red tipo Ejercicio ",ej
	with open(Net, "rb") as input:
		Red = cPickle.load(input) # protocol version is auto detected
	Red.load_dataset(Dataset, ej)
	lista_error,errorTotal,cantidad_errores = Red.evaluate()
	print '>> error total acumulado: ' + str(errorTotal)+' Numero de equivocaciones: '+str(cantidad_errores)	
	plt.plot(lista_error)
	plt.ylabel('Valor del error')
	if prnt:
		plt.show()
	return errorTotal, cantidad_errores	

def grid_search(param_grid):
	grid = ParameterGrid(param_grid)
	best_error = None
	best_params = None
	print "Ejercicio "+str(param_grid["1"])
	print len(grid), "modelos distintos"
	i = 0
	print "i 	error(training, validacion)"
	start = timer()
	for params in grid:
	    e = train(params['1'], params['2'], params['3'],params['4'],params['5'],params['6'],params['7'],params['8'],params['9'],params['10'],params['11'], True )
	    print i, "	", e
	    i += 1
	    if i%50 == 0:
	    	end = timer()
	    	print int((end-start)/60), " min"
	    if best_error == None or e < best_error:
	    	best_error = e
	    	best_params = [params['1'], params['2'], params['3'],params['4'],params['5'],params['6'],params['7'],params['8'],params['9'],params['10'],params['11']]
	print "MEJOR ERROR Y MODELO EJERCICIO "+str(param_grid["1"])
	print "training, validacion:", best_error
	print "parametros:", best_params
	end = timer()
	print int((end-start)/60), " min"
	
def pruebas():
	print "Grid search de parametros para mejor modelo"	
	# # EJERCICIO 1 
	# param_grid = {'1': [1],"2":['./datasets/tp1_ej1_training.csv'],"3": [None], "4":[0.1], "5": [200], "6":[0.001,0.01,0.1,0.5, 0.005, 0.2, 0.3, 0.4], "7":[0.1,0.3,0.5, 0.7, 0.9], "8": [0.70], "9": [0,1], "10":[0], "11":[[5],[10],[15],[20],[25],[15,15],[5,5],[10,10]]}
	# grid_search(param_grid)
	# EJERCICIO 2
	param_grid = {'1': [2],"2":['./datasets/tp1_ej2_training.csv'],"3": [None], "4":[0.1], "5": [200], "6":[0.005], "7":[0.9], "8": [0.70], "9": [1], "10":[0], "11":[[5]]}
	# param_grid = {'1': [2],"2":['./datasets/tp1_ej2_training.csv'],"3": [None], "4":[0.1], "5": [200], "6":[0.001,0.01,0.1,0.5, 0.005, 0.2, 0.3, 0.4], "7":[0.1,0.3,0.5, 0.7, 0.9], "8": [0.70], "9": [0,1], "10":[0], "11":[[5],[10],[15],[20],[25],[15,15],[5,5],[10,10]]}
	grid_search(param_grid)
	
args = sys.argv
message = "\nModo de uso:\n\
python main.py (ej1|ej2) -t nomDataSet nomFfileout parametros\n\
Con los siguientes parametros en orden:\n\
epsilon tau holdoutRate etha momentum modo_aprendizaje\n\n\
-t es para entrenar\
-l es para testear\
modo_aprendizaje = 0 para batch 1 para incremental\
"

# print len(args)
if len(args)==1:
	pruebas()
	sys.exit()
elif len(args) < 5:
	print message
	sys.exit()

cmdTrain = args[2] == "-t"
cmdLoad = args[2] == "-l"

if(args[1] == "ej1"):
	ejercicio=1
elif(args[1] == "ej2"):
	ejercicio=2
else:
	print message
	sys.exit()

if cmdTrain:
	
	if len(args) < 3:
		print "\nIncorrecta cantidad de argumentos para entrenar."
		print message
		sys.exit()
	archivoDataset = args[3]
	archivoRed=None
	errorAceptable=0.1
	maxEpoch=1000
	learningRate=0.01
	momentum=0.6
	holdoutRate=0.5
	modo=0
	early=0
	unitsPorCapa=[5]
	if len(args) > 4:
		archivoRed=args[4]
	if len(args) > 5:
		errorAceptable = float(args[5])
	if len(args) > 6:
		maxEpoch = int(args[6])
	if len(args) > 7:
		learningRate = float(args[7])
	if len(args) > 8:
		momentum = float(args[8])
	if len(args) > 9:
		holdoutRate = float(args[9])
	if len(args) > 10:
		modo=int(args[10])
	if len(args) > 11:
		early=int(args[11])
	if len(args) > 12:
		#unitsPorCapa=int(args[11])
		str1=str(args[12])
		unitsPorCapa=map(int, str1[1:-1].split(','))
	train(ejercicio,archivoDataset,archivoRed,errorAceptable,maxEpoch,learningRate,momentum,holdoutRate,modo, early, unitsPorCapa)

elif cmdLoad:
	
	if len(args) != 5:
		print "\nIncorrecta cantidad de argumentos para entrenar."
		print message
		sys.exit()
		
	archivoRed = args[3]
	archivoDataset = args[4]
	
	load(ejercicio,archivoRed, archivoDataset)
else:
	print message