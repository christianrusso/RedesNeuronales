from random import random
import numpy, Image
max_h_size=5
max_w_size=5


def invert(array):
	for x in range(len(array)):
		if array[x] == 0:
			array[x]=255
		else:
			array[x]=0

	return array

def noise(pixels, level, intensity):    # Para agregar ruido a la imagen
    level *= 0.01                       # Se obtiene la cantidad de ruido 
    intensity *= 13                     # Se obtiene la intensidad del ruido                  
    for a, pixel in enumerate(pixels):  # Se recorren los pixeles
        if(random() < level):           # Jugamos la probabilidad de modificar el pixel por ruido 
            color = (0+intensity) if random() < 0.5 else (255-intensity) # Se juega si el ruido sera un pixel blanco o negro
            pixel = (color, color, color) # Armamos el nuevo pixel
       	pixels[a] = pixel              # Lo modificamos en la lista
    return pixels                      # Regresamos los pixeles con el ruido eliminado o reducido.


def main():
	arch='amy.jpg'
	im=Image.open(arch)									#abro la imagen
	width, height = im.size
	print(arch)
	print 'Tamanio',width,height
	#im = im.crop((0, 0, max_w_size, max_h_size)) 	#aplico la funcion de crop para que no se vaya de rango
	im=im.convert("1")								#aplico la conversion a 1 bit de color
	for x in range(0,10):
		for y in range(0,10):
			matrix=list(im.getdata())						#obtengo el arreglo de la imagen
			#matrix=invert(matrix)							#proceso la imagen invirtiendola
			matrix=noise(matrix,x,y)
			im_reconstruida = Image.new(im.mode, im.size)	#construyo la imagen final
			im_reconstruida.putdata(matrix)					#construyo la imagen final con el arreglo procesado
			name='modified %d.jpg' % (x*10+y)
			im_reconstruida.save(name)				#guardo la imagen

if __name__ == "__main__":
    main()