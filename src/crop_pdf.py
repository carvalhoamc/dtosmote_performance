import os


for filename in os.listdir('./../output/cdfiles/'):
	arquivo = './../output/cdfiles/'+filename
	print(arquivo)
	os.system('pdf-crop-margins -v -s -u ' + arquivo)