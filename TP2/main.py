from redneu.hebb import GHANeuralNetwork
import numpy as np
from redneu.utils import BOWDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

dataset = BOWDataset(filename='tp2_training_dataset.csv')
tdataset = dataset.uncategorized_dataset()

EPOCHS = 100

hnn = GHANeuralNetwork(len(tdataset[0]), 3, 0.0001, 0.1)
hnn.train(tdataset[:600], EPOCHS, callback=call)

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

from som import SOM
import numpy as np
from pylab import Line2D, plot, axis, show, pcolor, colorbar, bone, savefig
import pylab as plt
import sys
import os

args = sys.argv
usage = "\nPara entrenar desde un dataset y guardar la red:\n\
python tp2.py nomArchivoData sigmaInicial lrateInicial M1 M2 epochs\n\
Guarda W en 'pesos.npy'\n\n\
Para cargar un 'pesos.npy' guardado y testearla contra un dataset:\n\
python tp2.py M1 M2 nomArchivoData\n\
Asume el dataset sigue la forma 'categoria, valor1, ... , valor856'"

N = 856

if len(args) == 7:
	# Entrenar
	nomArchivoData = args[1]
	sigmaInicial = float(args[2])
	lrateInicial = float(args[3])
	M1 = int(args[4])
	M2 = int(args[5])
	epochs = int(args[6])
	
	print "Entrenando con " + str(epochs) + ' epochs - ' + ' sigma ' + str(sigmaInicial) + ' lrate ' + str(lrateInicial) 
	data = np.genfromtxt(nomArchivoData, delimiter=',',usecols=range(1,857))
	s = SOM(M1, M2, N, sigmaInicial, lrateInicial, False)
	s.trainRandom(data, epochs)
	np.save('pesos', s.W)

elif len(args) == 4:
	# Cargar y testear.
	M1 = int(args[1])
	M2 = int(args[2])
	nomArchivoData = args[3]
	data = np.genfromtxt(nomArchivoData, delimiter=',',usecols=range(1,857))
	s = SOM(M1, M2, N)
	s.W = np.load('pesos.npy')
	if not os.path.exists("imgs"):
		os.makedirs("imgs")
	
	print "Procesando... por favor espere"
	markers =  ['o','s','D','o','s','D','o','s','D']
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'm', 'b']
	target = np.genfromtxt(nomArchivoData,delimiter=',',usecols=(0),dtype=str)
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
			s = SOM(M1, M2, N, sigma, lrate, nadaDeRandomW)

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