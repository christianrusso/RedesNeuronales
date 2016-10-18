from hebbian import red_hebbiana
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import Line2D, plot, axis, show, pcolor, colorbar, bone, savefig
import pylab as plt
import sys
import os
import cPickle

#Ejercicio 1
def  test_y_graficar(red,resultados):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	markers = [u'o', u'v', u'^', u'<', u'>', u'8', u's', u'p', u'*', u'h', u'H', u'D', u'd']
	colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k']
	
	for data in resultados:
		res=red.activate(np.array(data[1:]).reshape((1, 856)))
		ax.scatter([res[0][0]],[res[0][1]],[res[0][2]], marker=markers[data[0] - 1], c=colors[data[0] - 1])
		pass
	plt.show()


def load_Ej1(Dataset,Net):
	print "Cargando Red"
	with open(Net, "rb") as input:
		red = cPickle.load(input)
	dataset=(np.genfromtxt(Dataset,dtype=int, delimiter=',',usecols=range(0,857))).tolist()
	test_y_graficar(red,dataset)


def train_Ej1(Dataset,save_file,out_space,lrate,max_epochs,metodo):
	print "Entrenando " + str(max_epochs) + ' epocas y ' +' lrate: ' + str(lrate) +'\nreduciendo a '+str(out_space)+' dimensiones'
	if(metodo):
		print "Usando Oja"
	else:
		print "Usando Sanger"
	dataset=(np.genfromtxt(Dataset,dtype=int, delimiter=',',usecols=range(0,857))).tolist()

	traindataset = [ np.array(data[1:]).reshape((1, 856)) for data in dataset ] #quito la informacion sobre el tipo de dato

	hnn=red_hebbiana(856, out_space,lrate,metodo)
	hnn.train(traindataset[:600], max_epochs)

	test_y_graficar(hnn,dataset[600:])
	if(save_file!=None):
		print "Guardando Red"
		with open(save_file, "wb") as output:
			cPickle.dump(hnn, output, cPickle.HIGHEST_PROTOCOL)


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
metodo=True
out_space=int(3)
if operacion == "-train":
	# Entrenar
	if(len(args)>7):
		print usage1
		print usage3
		sys.exit()	
	if(len(args)>4):
		lrate = float(args[4])
	if(len(args)>5):
		epochs = int(args[5])
	if(len(args)>6):
		if(str(args[6])=="-o"):
			metodo=True
		elif(str(args[6])=="-s"):
			metodo=False
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