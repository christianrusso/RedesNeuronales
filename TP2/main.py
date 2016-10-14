from hebbian import red_hebbiana
import numpy as np
from utils import BOWDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from kohonen import som
import numpy as np
from pylab import Line2D, plot, axis, show, pcolor, colorbar, bone, savefig
import pylab as plt
import sys
import os
import cPickle

#Ejercicio 1

def load_Ej1(Dataset,Net):
	print "Cargando Red"
	with open(Net, "rb") as input:
		Red = cPickle.load(input)
	return Red

def train_Ej1(Dataset,save_file,sigma0,rateInicial,M1,M2,max_epochs):
	dataset = BOWDataset(filename='tp2_training_dataset.csv')
	tdataset = dataset.uncategorized_dataset()
	EPOCHS = max_epochs

	#hnn = GHANeuralNetwork(len(tdataset[0]), 3, 0.0001, 0.1)
	hnn=red_hebbiana(len(tdataset[0]), 3, 0.0001, 0.1)
	#hnn.train(tdataset[:600], EPOCHS, callback=call)
	hnn.train(tdataset[:600], EPOCHS)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	markers = [u'o', u'v', u'^', u'<', u'>', u'8', u's', u'p', u'*', u'h', u'H', u'D', u'd']
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']

	# fout = open('reduced_50e.csv', 'w');

	reduced_dataset = dataset.activate_neural_network(hnn)

	for data in reduced_dataset.dataset[600:]:
	    ax.scatter([data[1]], [data[2]], [data[3]], marker=markers[data[0] - 1], c=colors[data[0] - 1])
	    pass
	plt.show()
	print();
	if(save_file!=None):
		print "Guardando Red"
		with open(save_file, "wb") as output:
			cPickle.dump(hnn, output, cPickle.HIGHEST_PROTOCOL)


#Ejercicio 2
def train_Ej2(Dataset,save_file,sigma0,rateInicial,M1,M2,max_epochs):
	print "Entrenando con " + str(epochs) + ' epochs - ' + ' sigma: ' + str(sigmaInicial) + ' lrate: ' + str(lrateInicial) 
	data = np.genfromtxt(nomDataset, delimiter=',',usecols=range(1,857))
	#s = som(M1, M2, N, sigmaInicial, lrateInicial, False)
	s = som(M1, M2, lrateInicial,sigma0)
	s.trainRandom(data, epochs)
	np.save(nomRed, s.W)


def load_Ej2(Dataset,Net,M1,M2):
	data = np.genfromtxt(Dataset, delimiter=',',usecols=range(1,857))
	s = som(M1, M2, N)
	s.W = np.load(Net)
	if not os.path.exists("imgs"):
		os.makedirs("imgs")
	
	print "Procesando... por favor espere"
	markers =  ['o','s','D','o','s','D','o','s','D']
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'm', 'b']
	target = np.genfromtxt(Dataset,delimiter=',',usecols=(0),dtype=str)
	t = np.zeros(len(target),dtype=int)
	t[target == '1'] = 0
	t[target == '2'] = 1
	t[target == '3'] = 2
	t[target == '4'] = 3
	t[target == '5'] = 4
	t[target == '6'] = 5
	t[target == '7'] = 6
	t[target == '8'] = 7
	t[target == '9'] = 8

	maps = []
	for i in range(9):
		maps.append([])

	for cnt, xx in enumerate(data):
		w = s.mapear(xx)
		maps[t[cnt]].append(w)
		plot(w[0]+.5,w[1]+.5,markers[t[cnt]],markerfacecolor='None',
		
	markeredgecolor=colors[t[cnt]],markersize=12,markeredgewidth=2)
	axis([0,M1,0,M2])
	savefig('imgs/total.png')
	plt.close()

	for mapaIndex in range(len(maps)):
		
		mapa = maps[mapaIndex]
		color = colors[mapaIndex]
		marker = markers[mapaIndex]

		bone()
		for tupla in mapa:
			plot(tupla[0]+.5,tupla[1]+.5,marker,markerfacecolor='None',
		markeredgecolor=color,markersize=10,markeredgewidth=2)
		axis([0,M1,0,M2])
		savefig('imgs/parcialCat' + str(mapaIndex) + '.png')
		plt.close()
	print "Listo! Los resultados se encuentran en la carpeta 'imgs'."


args = sys.argv
usage = "\nPara entrenar desde un dataset y guardar la red:\n\
python main.py nomDataset nomRedOut -train sigmaInicial lrateInicial M1 M2 epochs\n\
Para cargar una red entrenada y testearla contra un dataset:\n\
python main.py nomDataset normRedIn -load M1 M2\n\
Asume el dataset sigue la forma 'categoria, valor1, ... , valor856'"

N = 856
if(len(args)<4):
	print usage
	sys.exit()
nomDataset = args[1]
nomRed = args[2]
operacion= str(args[3])
epochs=100
if operacion == "-train":
	# Entrenar
	if(len(args)!=9):
		print usage
		sys.exit()	
	sigmaInicial = float(args[4])
	lrateInicial = float(args[5])
	M1 = int(args[6])
	M2 = int(args[7])
	epochs = int(args[8])
	train_Ej1(nomDataset,nomRed,sigmaInicial,lrateInicial,M1,M2,epochs)
	#train_Ej2(nomDataset,nomRed,sigmaInicial,lrateInicial,M1,M2,epochs)

elif operacion == "-load":
	# Cargar y testear.
	print "Cargando"
	if(len(args)!=5):
		print usage
		sys.exit()
	M1 = int(args[4])
	M2 = int(args[5])
	load_Ej1(nomDataset,nomRed)
	#load_Ej2(nomDataset,nomRed,M1,M2)
else:
	print usage
	
""" Testeos.
N = 856
M1 = 20
M2 = 20

for epochs in [5, 20, 100]:
	
	for sigma in [0.5, 1.0, 5.0, 10.0]:
		for lrate in [0.5, 1.0, 2.0]:
		
			plt.close()


			nadaDeRandomW = False
			s = som(M1, M2, N, sigma, lrate, nadaDeRandomW)

			data = np.genfromtxt('tp2.csv', delimiter=',',usecols=range(1,857))

			print "Entrenando con " + str(epochs) + ' epochs - ' + ' sigma ' + str(sigma) + ' lrate ' + str(lrate) 

			if True:
				s.trainRandom(data, epochs)
				np.save('pesos', s.W)
			else:
				s.W = np.load('pesos.npy')
			print "Listo"


			prefijo = 'imgs/' + str(epochs) + 'sig' + str(sigma) + 'lr' + str(lrate)

			markers =  ['o','s','D','o','s','D','o','s','D']
			colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'm', 'b']
			target = np.genfromtxt('tp2.csv',delimiter=',',usecols=(0),dtype=str)
			t = np.zeros(len(target),dtype=int)
			t[target == '1'] = 0
			t[target == '2'] = 1
			t[target == '3'] = 2
			t[target == '4'] = 3
			t[target == '5'] = 4
			t[target == '6'] = 5
			t[target == '7'] = 6
			t[target == '8'] = 7
			t[target == '9'] = 8

			maps = []
			for i in range(9):
				maps.append([])

			for cnt, xx in enumerate(data):
				w = s.mapear(xx)
				maps[t[cnt]].append(w)
				plot(w[0]+.5,w[1]+.5,markers[t[cnt]],markerfacecolor='None',
				
			markeredgecolor=colors[t[cnt]],markersize=12,markeredgewidth=2)
			axis([0,M1,0,M2])
			savefig(prefijo + 'total.png')
			plt.close()

			for mapaIndex in range(len(maps)):
				
				mapa = maps[mapaIndex]
				color = colors[mapaIndex]
				marker = markers[mapaIndex]

				bone()
				for tupla in mapa:
					plot(tupla[0]+.5,tupla[1]+.5,marker,markerfacecolor='None',
				markeredgecolor=color,markersize=10,markeredgewidth=2)
				axis([0,M1,0,M2])
				savefig(prefijo + 'parcial' + str(mapaIndex) + '.png')
				plt.close()
"""