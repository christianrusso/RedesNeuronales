from new_perceptron import perceptron_Multiple,sigmoidea_bipolar,sigmoidea_logistica
from numpy import *
#import matplotlib.pyplot as plt
from sklearn.grid_search import ParameterGrid
from pylab import Line2D, plot, axis, show, pcolor, colorbar, bone, savefig
import pylab as plt
import sys
import cPickle
import datetime

def wrapper(ejercicio,Dataset=None, save_file=None, epsilon=0.1, tau=1000, etha=0.01,m=0.6,holdoutRate=0.5, modo=1, early=0, unitsPorCapa=[15]):
	Red=perceptron_Multiple(unitsPorCapa,epsilon,tau,etha,m,holdoutRate)
	if(ejercicio==1):
		Red.load_dataset_1(Dataset)
	else:
		Red.load_dataset_2(Dataset)
	best_error = -1
	error_t_hist_best = None
	error_v_hist_best = None
	error_v_hist_sum_best = None
	error_t_hist_sum_best = None
	for _ in xrange(3):
		error_t_hist,error_v_hist, error_v_hist_sum, error_t_hist_sum, epoch = Red.train(modo, early)
		if best_error == -1 or error_v_hist_sum[-1] < best_error:
			best_error = error_v_hist_sum[-1]
			error_t_hist_best = error_t_hist
			error_v_hist_best = error_v_hist
			error_v_hist_sum_best = error_v_hist_sum
			error_t_hist_sum_best = error_t_hist_sum
	plt.xlabel('Epoch')
	plt.title("Ejercicio "+str(ejercicio))
	plt.plot(error_t_hist_best, label="Training")
	plt.ylabel('Promedio de errores')
	plt.plot(error_v_hist_best, label="Validacion")
	plt.grid()
	plt.legend()
	img_name= "ej"+str(ejercicio)+"_"+str(etha)+"_"+str(m)+"_"+str(modo)+"_"+str(unitsPorCapa)+"_mean.png"
	plt.savefig("./grid_search/"+img_name)
	plt.close()
	plt.title("Ejercicio "+str(ejercicio))
	plt.xlabel('Epoch')
	plt.plot(error_t_hist_sum_best, label="Training")
	plt.ylabel('Suma de errores')
	plt.plot(error_v_hist_sum_best, label="Validacion")
	plt.grid()
	plt.legend()
	img_name= "ej"+str(ejercicio)+"_"+str(etha)+"_"+str(m)+"_"+str(modo)+"_"+str(unitsPorCapa)+"_sum.png"
	plt.savefig("./grid_search/"+img_name)
	plt.close()


	return best_error

def train(ejercicio,Dataset=None, save_file=None, epsilon=0.1, tau=1000, etha=0.01,m=0.6,holdoutRate=0.5, modo=0, early=0,unitsPorCapa=[15]):
	print "Ejercicio ",ejercicio	
	print "File "+str(Dataset)
	print "Unidades por capa " + str(unitsPorCapa)
	print "Error aceptable " + str(epsilon)
	print "Max Epocas " +str(tau)
	print "Learning Rate " + str(etha)
	Red=perceptron_Multiple(unitsPorCapa,epsilon,tau,etha,m,holdoutRate)
	if(ejercicio==1):
		Red.load_dataset_1(Dataset)
	else:
		Red.load_dataset_2(Dataset)
	error_t_hist,error_v_hist, error_v_hist_sum, error_t_hist_sum, epoch = Red.train(modo, early)
	print '>> epochAlcanzada: ' + str(epoch)
	print '>> errorTraining: ' + str(error_t_hist[-1])
	print '>> errorTesting: ' + str(error_v_hist[-1])	
	plt.xlabel('Epoch')
	plt.title("Ejercicio "+str(ejercicio))
	plt.plot(error_t_hist, label="Training")
	plt.ylabel('Promedio de errores')
	#plt.show()
	plt.plot(error_v_hist, label="Validacion")
	#plt.ylabel('error Testing Vs epocas')
	plt.grid()
	plt.legend()
	r=Dataset.rfind('.')
	current_time = datetime.datetime.now()
	img_name= Dataset[0:r]+ "_" +str(current_time)+"_mean.png"
	plt.savefig(img_name)
	# plt.show()
	plt.close()
	plt.title("Ejercicio "+str(ejercicio))
	plt.xlabel('epoch')
	plt.plot(error_t_hist_sum, label="Training")
	plt.ylabel('Suma de errores')
	#plt.show()
	plt.plot(error_v_hist_sum, label="Validacion")
	#plt.ylabel('error Testing Vs epocas')
	plt.grid()
	plt.legend()
	img_name= Dataset[0:r]+ " " +str(current_time)+"_sum.png"
	plt.savefig(img_name)
	# plt.show()
	plt.close()

	if(save_file!=None):
		print "Guardando Red"
		with open(save_file, "wb") as output:
			cPickle.dump(Red, output, cPickle.HIGHEST_PROTOCOL)
	return mean(error_t_hist), mean(error_v_hist)

def load(ej,Net=None,Dataset=None):
	print "Cargando Red tipo Ejercicio ",ej
	with open(Net, "rb") as input:
		Red = cPickle.load(input) # protocol version is auto detected
	if(ej==1):
		Red.load_dataset_1(Dataset)
	else:
		Red.load_dataset_2(Dataset)
	lista_error,errorTotal,cantidad_errores = Red.evaluate()
	print '>> error total acumulado: ' + str(errorTotal)+' Numero de equivocaciones: '+str(cantidad_errores)	
	plt.plot(lista_error)
	plt.ylabel('Valor del error')
	plt.show()
	return mean(lista_error)


def pruebas():
	print "Grid search de parametros para mejor modelo"
	
	# EJERCICIO 1 
	# param_grid = {'1': [1],"2":['./datasets/tp1_ej1_training.csv'],"3": [None], "4":[0.1], "5": [200], "6":[0.001], "7":[0.1], "8": [0.70], "9": [0], "10":[0], "11":[[5]]}
	param_grid = {'1': [1],"2":['./datasets/tp1_ej1_training.csv'],"3": [None], "4":[0.1], "5": [200], "6":[0.001,0.01,0.1,0.5, 0.005, 0.2, 0.3, 0.4], "7":[0.1,0.3,0.5, 0.7, 0.9], "8": [0.70], "9": [0,1], "10":[0], "11":[[5],[10],[15],[20],[25],[15,15],[5,5],[10,10]]}
	grid = ParameterGrid(param_grid)
	best_error = None
	best_params = None
	print "Ejercicio 1"
	print len(grid), "modelos distintos"
	i = 0
	print "i 	error"
	for params in grid:
	    e = wrapper(params['1'], params['2'], params['3'],params['4'],params['5'],params['6'],params['7'],params['8'],params['9'],params['10'],params['11'] )
	    print i, e
	    i += 1
	    if best_error == None or e < best_error:
	    	best_error = e
	    	best_params = [params['1'], params['2'], params['3'],params['4'],params['5'],params['6'],params['7'],params['8'],params['9'],params['10'],params['11']]
	print "MEJOR ERROR Y MODELO EJERCICIO 1"
	print best_error, best_params
	# EJERCICIO 2
	param_grid = {'1': [2],"2":['./datasets/tp1_ej2_training.csv'],"3": [None], "4":[0.1], "5": [200], "6":[0.001,0.01,0.1,0.5, 0.005, 0.2, 0.3, 0.4], "7":[0.1,0.3,0.5, 0.7, 0.9], "8": [0.70], "9": [0,1], "10":[0], "11":[[5],[10],[15],[20],[25],[15,15],[5,5],[10,10]]}
	grid = ParameterGrid(param_grid)
	best_error = None
	best_params = None
	print "Ejercicio 2"
	print len(grid), "modelos distintos"
	i = 0
	print "i 	error"
	for params in grid:
	    e = wrapper(params['1'], params['2'], params['3'],params['4'],params['5'],params['6'],params['7'],params['8'],params['9'],params['10'],params['11'] )
	    print i, e
	    i += 1
	    if best_error == None or e < best_error:
	    	best_error = e
	    	best_params = [params['1'], params['2'], params['3'],params['4'],params['5'],params['6'],params['7'],params['8'],params['9'],params['10'],params['11']]
	print "MEJOR ERROR Y MODELO EJERCICIO 2"
	print best_error, best_params

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
	
	if len(args) != 12:
		print "\nIncorrecta cantidad de argumentos para entrenar."
		print message
		sys.exit()
		
	archivoDataset = args[3]
	archivoRed=args[4]
	errorAceptable = float(args[5])
	maxEpoch = int(args[6])
	holdoutRate = float(args[7])
	learningRate = float(args[8])
	momentum = float(args[9])
	modo=int(args[10])
	early=int(args[11])
	train(ejercicio,archivoDataset,archivoRed,errorAceptable,maxEpoch,learningRate,momentum,holdoutRate,modo, early)

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