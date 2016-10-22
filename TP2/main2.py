import matplotlib.pyplot as plt
from kohonen import som
import numpy as np
import matplotlib.cm as mmc
import sys
import os
import cPickle

def prueba():
	file="tp2_training_dataset.csv"
	train_data = np.genfromtxt(file, delimiter=',',usecols=range(1,857))
	train_data=train_data[:600]
	test_data  = np.genfromtxt(file, delimiter=',',usecols=range(0,857))
	test_data=test_data[600:]
	if not os.path.exists("imgs/ej2"):
		os.makedirs("imgs/ej2")
	sigma=0.001
	max_epoca=100
	for M in [3,5]:
		for lrate in np.linspace(0.001, 0.1, 20):
			red = som(M, M, lrate,sigma)
			red.train(train_data, max_epoca)
			img_name="imgs/ej2/train M "+str(M)+" lrate "+str(lrate)+" sigma"+str(sigma)+".jpg"
			graficador(red,train_data,img_name)
			img_name="imgs/ej2/test M "+str(M)+" lrate "+str(lrate)+" sigma"+str(sigma)+".jpg"
			graficador(red,test_data,img_name)

def graficador(red,dataset,save_img=None):
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
	color=np.asarray(color)
	plt.pcolor(color, cmap=cmap)
	plt.title('Resultado')

	# y, x = np.mgrid[slice(0, red.M2),slice(0, red.M1)]
	# plt.subplot(2, 2, 1)
	# plt.pcolor(x, y, color, cmap=cmap)
	# plt.axis([x.min(), x.max(), y.min(), y.max()])

	plt.colorbar()

	if(save_img==None):
		plt.show()
	else:
		plt.savefig(save_img)

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
	print "Generando mapa de caracteristicas"
	dataset = np.genfromtxt(file, delimiter=',',usecols=range(0,857))
	graficador(red,dataset)


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

if(len(args)==2 and args[1]=="-prueba"):
	prueba()
	sys.exit()
elif(len(args)<4):
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