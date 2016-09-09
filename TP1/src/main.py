import numpy as np
import sys

def ej1():
	print "ejercicio 1"


def ej2():
	print "ejercicio 2"

# Main.
args = sys.argv
usage = "ACA PONER LOS COMANDOS PARA USAR"

if len(args) < 1:
	print usage
	sys.exit()
else:
	cmdEj1 = args[1] == "ej1"
	cmdEj2 = args[1] == "ej2"

if cmdEj1:
	ej1()
elif cmdEj2:
	ej2()
else:
	print usage