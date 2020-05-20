from parameters import releases, input_dir
from performance import Performance
import pandas as pd



def main():
	analisys = Performance()
	#for r in releases:
	#	analisys.average_results(input_dir+'dto_multiclass_results_'+r+'.csv',kind='multiclass',release=r)
	#	analisys.run_rank_choose_parameters(input_dir+'average_results_multiclass_'+r+'.csv',kind='multiclass',release=r)
	#	analisys.grafico_variacao_alpha(kind='multiclass',release=r)
	
	#analisys.best_alpha(kind='multiclass')
	#r = 'v1'
	#analisys.run_global_rank(input_dir+'average_results_multiclass_'+r+'.csv',kind='multiclass',release=r)
	
	#analisys.overall_rank('auc','biclass',7.0)
	#analisys.overall_rank('geo', 'biclass', 7.0)
	#analisys.overall_rank('iba', 'biclass', 7.0)
	#analisys.overall_rank('geo', 'multiclass', 7.5)
	#analisys.overall_rank('iba', 'multiclass', 7.5)
	
	df = pd.read_csv('./../output/overall_rank_results_multiclass_7.5_iba.csv')
	analisys.cd_graphics(df,datasetlen=7,kind='multiclass')
	
	df = pd.read_csv('./../output/overall_rank_results_multiclass_7.5_geo.csv')
	analisys.cd_graphics(df, datasetlen=7, kind='multiclass')
	
	df = pd.read_csv('./../output/overall_rank_results_biclass_7.0_auc.csv')
	analisys.cd_graphics(df, datasetlen=61, kind='biclass')
	
	df = pd.read_csv('./../output/overall_rank_results_biclass_7.0_geo.csv')
	analisys.cd_graphics(df, datasetlen=61, kind='biclass')
	
	df = pd.read_csv('./../output/overall_rank_results_biclass_7.0_iba.csv')
	analisys.cd_graphics(df, datasetlen=61, kind='biclass')








if __name__ == '__main__':
	main()
#TODO TESTAR OUTRAS GEOMETRIAS E OUTROS ALPHAS