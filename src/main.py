from parameters import releases, input_dir
from performance import Performance
import pandas as pd



def main():
	analisys = Performance()
	'''for r in releases:
		analisys.average_results(input_dir+'dto_biclass_results_'+r+'.csv',kind='biclass',release=r)
		analisys.run_rank(input_dir+'average_results_biclass_'+r+'.csv',kind='biclass',release=r)
		analisys.grafico_variacao_alpha(kind='biclass',release=r)
	'''
	analisys.best_alpha(kind='biclass')


if __name__ == '__main__':
	main()
