from hebbian import red_hebbiana
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import Line2D, plot, axis, show, pcolor, colorbar, bone, savefig
import pylab as plt
import sys
import os
import time
import cPickle

#Ejercicio 1
def test_y_graficar(red,resultados, metodo=0, load=False):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
	
	for i in xrange(len(resultados)):
		data = resultados[i]
		res=red.activate(np.array(data[1:]).reshape((1, 856)))
		if not load:
			if i >= 600:
				ax.scatter([res[0][0]],[res[0][1]],[res[0][2]], marker="o", c=colors[data[0] - 1], s=20*4)
			else:
				ax.scatter([res[0][0]],[res[0][1]],[res[0][2]], marker="v", c=colors[data[0] - 1], s=20*4)
		else:
			ax.scatter([res[0][0]],[res[0][1]],[res[0][2]], marker="o", c=colors[data[0] - 1], s=20*4)
			
	timestr = time.strftime("%Y%m%d-%H%M%S")
	dirpath = "imgs/ej1/"+str(metodo)+"/"+timestr
	if not os.path.exists(dirpath):
		os.makedirs(dirpath)
	
	for ii in xrange(0,360, 40):
		ax.azim = ii 		
		savefig(dirpath+"/slice%d.png" %ii)

	# plt.show()


def load_Ej1(Dataset,Net):
	print "Cargando Red"
	with open(Net, "rb") as input:
		red = cPickle.load(input)
	dataset=(np.genfromtxt(Dataset,dtype=int, delimiter=',',usecols=range(0,857))).tolist()
	test_y_graficar(red,dataset, load=True)


def train_Ej1(Dataset,save_file,out_space,lrate,max_epochs,metodo):
	print "Entrenando " + str(max_epochs) + ' epocas y ' +' lrate: ' + str(lrate) +'\nreduciendo a '+str(out_space)+' dimensiones'
	if(metodo):
		print "Usando Oja"
	else:
		print "Usando Sanger"
	dataset=(np.genfromtxt(Dataset,dtype=int, delimiter=',',usecols=range(0,857))).tolist()

	traindataset = [ np.array(data[1:]).reshape((1, 856)) for data in dataset ] #quito la informacion sobre el tipo de dato

	hnn=red_hebbiana(856, out_space,lrate,metodo)
	convergio = hnn.train(traindataset[:600], max_epochs)

	if convergio:
		test_y_graficar(hnn,dataset, metodo)
	if(save_file!=None):
		print "Guardando Red"
		with open(save_file, "wb") as output:
			cPickle.dump(hnn, output, cPickle.HIGHEST_PROTOCOL)
	return convergio

def pruebas(dataset,save_file,max_epochs=5000):
	best_params_oja = []
	best_params_sanger = -1
	for m in [0, 1]:
		for lrate in np.linspace(0.001, 0.1, 1):
			convergio = train_Ej1(dataset, save_file, 3, lrate, max_epochs,m)
			if convergio and not m:
				best_params_sanger = lrate
				break
			if convergio and m:
				best_params_oja.append(lrate)
	return best_params_sanger, best_params_oja





args = sys.argv
usage1 = "\nPara entrenar desde un dataset y guardar la red:\n\
python main.py nomDataset nomRedOut -train lrate epochs metodo out_space \n\
metodo -o para usar oja -s para usar sanger"
usage2= "\nPara cargar una red entrenada y testearla contra un dataset:\n\
python main.py nomDataset normRedIn -load\n"
usage3="Asume el dataset sigue la forma 'categoria, valor1, ... , valor856'"

if(len(args)<4):
	print usage1
	print usage2
	print usage3
	sys.exit()
nomDataset = args[1]
nomRed = args[2]
operacion= str(args[3])
#default value
epochs=int(1000)
lrate=float(0.0001)
metodo=1
out_space=int(3)
if operacion == "-train":
	# Entrenar
	if(len(args)>8):
		print usage1
		print usage3
		sys.exit()	
	if(len(args)>4):
		lrate = float(args[4])
	if(len(args)>5):
		epochs = int(args[5])
	if(len(args)>6):
		if(str(args[6])=="-o"):
			metodo=1
		elif(str(args[6])=="-s"):
			metodo=0
		else:
			print usage1
			print usage3
			sys.exit()
	if(len(args)>7):
		out_space=int(args[7])

	train_Ej1(nomDataset,nomRed,out_space,lrate,epochs,metodo)

elif operacion == "-load":
	# Cargar y testear.
	if(len(args)!=4):
		print usage2
		print usage3
		sys.exit()
	load_Ej1(nomDataset,nomRed)
elif operacion == "-pruebas":
	print pruebas(nomDataset, nomRed)
else:
	print usage1
	print usage2
	print usage3


