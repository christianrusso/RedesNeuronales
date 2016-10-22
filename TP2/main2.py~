import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from kohonen import som
import numpy as np
import matplotlib.cm as mmc
from pylab import Line2D, plot, axis, show, pcolor, colorbar, bone, savefig
import pylab as plt
import sys
import os
import cPickle

#Ejercicio 2
def train_Ej2(Dataset,save_file,sigma0,lrateInicial,M1,M2,max_epochs):
	print "Entrenando con " + str(epochs) + ' epochs - ' + ' sigma: ' + str(sigmaInicial) + ' lrate: ' + str(lrateInicial) 
	data = np.genfromtxt(Dataset, delimiter=',',usecols=range(1,857))
	s = som(M1, M2, lrateInicial,sigma0)
	s.train(data, epochs)
	if(save_file!=None):
		print "Guardando red"
		with open(save_file, "wb") as output:
			cPickle.dump(s, output, cPickle.HIGHEST_PROTOCOL)


def load_Ej2(file,Net):
	print "Cargando red"
	with open(Net, "rb") as input:
		red = cPickle.load(input)

	if not os.path.exists("imgs/ej2"):
		os.makedirs("imgs/ej2")

	print "Generando mapa de caracteristicas"
	
	dataset = np.genfromtxt(file, delimiter=',',usecols=range(0,857))
	color = [[dict() for _ in xrange(red.M2)] for _ in xrange(red.M1)]

	for data in dataset:
		pos=red.test(data[1:].reshape((1,856)))

		d = color[pos[0]][pos[1]]
		if (int(data[0])-1) in d:
			d[int(data[0])-1] += 1
		else:
			d[int(data[0])-1] = 1
	

	for i in xrange(red.M1):
		for j in xrange(red.M2):
			d = color[i][j]
			if d == {}:
				color[i][j] = 9
			else:
				most_repeated = max(d.iterkeys(), key=(lambda key: d[key]))
				color[i][j] = most_repeated

	cmap = mmc.get_cmap(name="nipy_spectral", lut=10)
	plt.pcolor(color, cmap=cmap)
	plt.colorbar()
	plt.show()
	print "Listo! Los resultados se encuentran en la carpeta 'imgs'."

args = sys.argv
usage1 = "\nPara entrenar desde un dataset y guardar la red:\n\
python main.py nomDataset nomRedOut -train sigmaInicial lrateInicial dimX dimY epochs\n"
usage2= "\nPara cargar una red entrenada y testearla contra un dataset:\n\
python main.py nomDataset normRedIn -load\n"
usage3="Asume el dataset sigue la forma 'categoria, valor1, ... , valor856'"

#default value
lrate=float(0.5)
sigmaInicial=float(0.3)
epochs=100
X=20
Y=20

if(len(args)<4):
	print usage1
	print usage2
	print usage3
	sys.exit()
nomDataset = args[1]
nomRed = args[2]
operacion= str(args[3])
if operacion == "-train":
	# Entrenar
	if(len(args)>9):
		print usage1
		print usage3
		sys.exit()
	if(len(args)>4):
		sigmaInicial = float(args[4])
	if(len(args)>5):
		lrate = float(args[5])
	if(len(args)>6):
		X = int(args[6])
	if(len(args)>7):
		Y = int(args[7])
	if(len(args)>8):
		epochs = int(args[8])
	train_Ej2(nomDataset,nomRed,sigmaInicial,lrate,X,Y,epochs)

elif operacion == "-load":
	# Cargar y testear.
	if(len(args)!=4):
		print usage1
		print usage3
		sys.exit()
	load_Ej2(nomDataset,nomRed)
else:
	print usage1
	print usage2
	print usage3

""" Testeos.
N = 856
M1 = 20
M2 = 20

for epochs in [5, 20, 100]:
	
	for sigma in [0.5, 1.0, 5.0, 10.0]:
		for lrate in [0.5, 1.0, 2.0]:
"""