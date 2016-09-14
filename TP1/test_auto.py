import sys
import os
if __name__ == "__main__": 
	args = sys.argv
	if len(args)==2:
		file=str(args[1])
		print file
	else:
		print "falta el nombre de archivo de tests"
		sys.exit()
	f = open(file)
	for line in f:
		if len(line)>1 and line[0]!='#':
			print line
			os.system(line) 