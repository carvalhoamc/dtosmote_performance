import os


for filename in os.listdir('./../rank/artigo/multiclasse/'):
	arquivo = './../rank/artigo/multiclasse/'+filename
	print(arquivo)
	os.system('pdf-crop-margins -v -s -u ' + arquivo)