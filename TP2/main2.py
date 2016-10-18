import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from kohonen import som
import numpy as np
from pylab import Line2D, plot, axis, show, pcolor, colorbar, bone, savefig
import pylab as plt
import sys
import os
import cPickle

#Ejercicio 2
def train_Ej2(Dataset,save_file,sigma0,lrateInicial,M1,M2,max_epochs):
	print "Entrenando con " + str(epochs) + ' epochs - ' + ' sigma: ' + str(sigmaInicial) + ' lrate: ' + str(lrateInicial) 
	data = np.genfromtxt(Dataset, delimiter=',',usecols=range(1,857))
	#s = som(M1, M2, N, sigmaInicial, lrateInicial, False)
	#s = som(M1, M2, lrateInicial,sigma0)
	s = som(len(data[0]), M2, lrateInicial,sigma0)
	s.train(data, epochs)
	#np.save(nomRed, s.W)
	if(save_file!=None):
		print "Guardando Red"
		with open(save_file, "wb") as output:
			cPickle.dump(s, output, cPickle.HIGHEST_PROTOCOL)


def load_Ej2(Dataset,Net):#,M1,M2):
	data = np.genfromtxt(Dataset, delimiter=',',usecols=range(1,857))
	#s = som(M1, M2, N)
	#s.W = np.load(Net)
	print "Cargando Red"
	with open(Net, "rb") as input:
		s = cPickle.load(input)

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
		w = s.activate(xx)
		maps[t[cnt]].append(w)
		plot(w[0]+.5,w[1]+.5,markers[t[cnt]],markerfacecolor='None',
		
	markeredgecolor=colors[t[cnt]],markersize=12,markeredgewidth=2)
	axis([0,s.ninputs,0,s.output_size[0]])
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
		axis([0,s.ninputs,0,s.output_size[0]])
		savefig('imgs/parcialCat' + str(mapaIndex) + '.png')
		plt.close()
	print "Listo! Los resultados se encuentran en la carpeta 'imgs'."


args = sys.argv
usage1 = "\nPara entrenar desde un dataset y guardar la red:\n\
python main.py nomDataset nomRedOut -train sigmaInicial lrateInicial M1 [X,Y] epochs\n"
usage2= "\nPara cargar una red entrenada y testearla contra un dataset:\n\
python main.py nomDataset normRedIn -load\n"
usage3="Asume el dataset sigue la forma 'categoria, valor1, ... , valor856'"

N = 856
#default value
lrate=float(0.001)
sigmaInicial=float(0.1)
epochs=1000
M1=300
M2=[3,3]

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
		M1 = int(args[6])
	if(len(args)>7):
		M2 = list(map(int, args[7].split(',')))
	print M2
	if(len(args)>8):
		epochs = int(args[8])
	train_Ej2(nomDataset,nomRed,sigmaInicial,lrate,M1,M2,epochs)

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