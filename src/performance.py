import Orange
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from parameters import output_dir, rank_dir, input_dir
from classifiers import classifiers_list
from datasets import dataset_biclass, dataset_multiclass

# geometry
order = ['area',
         'volume',
         'area_volume_ratio',
         'edge_ratio',
         'radius_ratio',
         'aspect_ratio',
         'max_solid_angle',
         'min_solid_angle',
         'solid_angle']

# Dirichlet Distribution alphas
alphas = np.arange(1, 10, 0.5)


class Performance:
	
	def __init__(self):
		pass
	
	def average_results(self, rfile, kind, release):
		'''
		Calculates average results
		:param rfile: filename with results
		:param kind: biclass or multiclass
		:return: avarege_results in another file
		'''
		
		df = pd.read_csv(rfile)
		t = pd.Series(data=np.arange(0, df.shape[0], 1))
		if kind == 'biclass':
			dfr = pd.DataFrame(columns=['MODE', 'DATASET', 'PREPROC', 'ALGORITHM', 'ORDER',
			                            'ALPHA', 'PRE', 'REC', 'SPE', 'F1', 'GEO', 'IBA', 'AUC'],
			                   index=np.arange(0, int(t.shape[0] / 5)))
		else:
			dfr = pd.DataFrame(columns=['MODE', 'DATASET', 'PREPROC', 'ALGORITHM', 'ORDER',
			                            'ALPHA', 'PRE', 'REC', 'SPE', 'F1', 'GEO', 'IBA'],
			                   index=np.arange(0, int(t.shape[0] / 5)))
		
		df_temp = df.groupby(by=['MODE', 'DATASET', 'PREPROC', 'ALGORITHM'])
		idx = dfr.index.values
		i = idx[0]
		for name, group in df_temp:
			group = group.reset_index()
			dfr.at[i, 'MODE'] = group.loc[0, 'MODE']
			dfr.at[i, 'DATASET'] = group.loc[0, 'DATASET']
			dfr.at[i, 'PREPROC'] = group.loc[0, 'PREPROC']
			dfr.at[i, 'ALGORITHM'] = group.loc[0, 'ALGORITHM']
			dfr.at[i, 'ORDER'] = group.loc[0, 'ORDER']
			dfr.at[i, 'ALPHA'] = group.loc[0, 'ALPHA']
			dfr.at[i, 'PRE'] = group['PRE'].mean()
			dfr.at[i, 'REC'] = group['REC'].mean()
			dfr.at[i, 'SPE'] = group['SPE'].mean()
			dfr.at[i, 'F1'] = group['F1'].mean()
			dfr.at[i, 'GEO'] = group['GEO'].mean()
			dfr.at[i, 'IBA'] = group['IBA'].mean()
			if kind == 'biclass':
				dfr.at[i, 'AUC'] = group['AUC'].mean()
			i = i + 1
		
		print('Total lines in a file: ', i)
		dfr.to_csv(input_dir + 'average_results_' + kind + '_' + str(release) + '.csv', index=False)
	
	def rank_by_algorithm(self, df, kind, order, alpha, release, smote=False):
		'''
		Calcula rank
		:param df:
		:param tipo:
		:param wd:
		:param delaunay_type:
		:return:
		'''
		biclass_measures = ['PRE', 'REC', 'SPE', 'F1', 'GEO', 'IBA', 'AUC']
		multiclass_measures = ['PRE', 'REC', 'SPE', 'F1', 'GEO', 'IBA']
		
		df_table = pd.DataFrame(
				columns=['DATASET', 'ALGORITHM', 'ORIGINAL', 'RANK_ORIGINAL', 'SMOTE', 'RANK_SMOTE', 'SMOTE_SVM',
				         'RANK_SMOTE_SVM', 'BORDERLINE1', 'RANK_BORDERLINE1', 'BORDERLINE2', 'RANK_BORDERLINE2',
				         'GEOMETRIC_SMOTE', 'RANK_GEOMETRIC_SMOTE', 'DELAUNAY', 'RANK_DELAUNAY', 'DELAUNAY_TYPE',
				         'ALPHA', 'unit'])
		
		df_temp = df.groupby(by=['ALGORITHM'])
		for name, group in df_temp:
			
			group = group.reset_index()
			group.drop('index', axis=1, inplace=True)
			if smote == False:
				df.to_csv(rank_dir + release + '_' + kind + '_' + order + '_' + str(alpha) + '.csv', index=False)
			else:
				df.to_csv(rank_dir + release + '_smote_' + kind + '_' + order + '_' + str(alpha) + '.csv', index=False)
			
			j = 0
			if kind == 'biclass':
				dataset = dataset_biclass
				measures = biclass_measures
			else:
				dataset = dataset_multiclass
				measures = multiclass_measures
			
			for d in dataset:
				for m in measures:
					aux = group[group['DATASET'] == d]
					aux = aux.reset_index()
					df_table.at[j, 'DATASET'] = d
					df_table.at[j, 'ALGORITHM'] = name
					indice = aux.PREPROC[aux.PREPROC == '_train'].index.tolist()[0]
					df_table.at[j, 'ORIGINAL'] = aux.at[indice, m]
					indice = aux.PREPROC[aux.PREPROC == '_SMOTE'].index.tolist()[0]
					df_table.at[j, 'SMOTE'] = aux.at[indice, m]
					indice = aux.PREPROC[aux.PREPROC == '_smoteSVM'].index.tolist()[0]
					df_table.at[j, 'SMOTE_SVM'] = aux.at[indice, m]
					indice = aux.PREPROC[aux.PREPROC == '_Borderline1'].index.tolist()[0]
					df_table.at[j, 'BORDERLINE1'] = aux.at[indice, m]
					indice = aux.PREPROC[aux.PREPROC == '_Borderline2'].index.tolist()[0]
					df_table.at[j, 'BORDERLINE2'] = aux.at[indice, m]
					indice = aux.PREPROC[aux.PREPROC == '_Geometric_SMOTE'].index.tolist()[0]
					df_table.at[j, 'GEOMETRIC_SMOTE'] = aux.at[indice, m]
					indice = aux.PREPROC[aux.ORDER == order].index.tolist()[0]
					df_table.at[j, 'DELAUNAY'] = aux.at[indice, m]
					df_table.at[j, 'DELAUNAY_TYPE'] = order
					df_table.at[j, 'ALPHA'] = alpha
					df_table.at[j, 'unit'] = m
					j += 1
			
			df_pre = df_table[df_table['unit'] == 'PRE']
			df_rec = df_table[df_table['unit'] == 'REC']
			df_spe = df_table[df_table['unit'] == 'SPE']
			df_f1 = df_table[df_table['unit'] == 'F1']
			df_geo = df_table[df_table['unit'] == 'GEO']
			df_iba = df_table[df_table['unit'] == 'IBA']
			if kind == 'biclass':
				df_auc = df_table[df_table['unit'] == 'AUC']
			
			pre = df_pre[
				['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', 'GEOMETRIC_SMOTE', 'DELAUNAY']]
			rec = df_rec[
				['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', 'GEOMETRIC_SMOTE', 'DELAUNAY']]
			spe = df_spe[
				['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', 'GEOMETRIC_SMOTE', 'DELAUNAY']]
			f1 = df_f1[['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', 'GEOMETRIC_SMOTE', 'DELAUNAY']]
			geo = df_geo[
				['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', 'GEOMETRIC_SMOTE', 'DELAUNAY']]
			iba = df_iba[
				['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', 'GEOMETRIC_SMOTE', 'DELAUNAY']]
			if kind == 'biclass':
				auc = df_auc[['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', 'GEOMETRIC_SMOTE',
				              'DELAUNAY']]
			
			pre = pre.reset_index()
			pre.drop('index', axis=1, inplace=True)
			rec = rec.reset_index()
			rec.drop('index', axis=1, inplace=True)
			spe = spe.reset_index()
			spe.drop('index', axis=1, inplace=True)
			f1 = f1.reset_index()
			f1.drop('index', axis=1, inplace=True)
			geo = geo.reset_index()
			geo.drop('index', axis=1, inplace=True)
			iba = iba.reset_index()
			iba.drop('index', axis=1, inplace=True)
			if kind == 'biclass':
				auc = auc.reset_index()
				auc.drop('index', axis=1, inplace=True)
			
			# calcula rank linha a linha
			pre_rank = pre.rank(axis=1, ascending=False)
			rec_rank = rec.rank(axis=1, ascending=False)
			spe_rank = spe.rank(axis=1, ascending=False)
			f1_rank = f1.rank(axis=1, ascending=False)
			geo_rank = geo.rank(axis=1, ascending=False)
			iba_rank = iba.rank(axis=1, ascending=False)
			if kind == 'biclass':
				auc_rank = auc.rank(axis=1, ascending=False)
			
			df_pre = df_pre.reset_index()
			df_pre.drop('index', axis=1, inplace=True)
			df_pre['RANK_ORIGINAL'] = pre_rank['ORIGINAL']
			df_pre['RANK_SMOTE'] = pre_rank['SMOTE']
			df_pre['RANK_SMOTE_SVM'] = pre_rank['SMOTE_SVM']
			df_pre['RANK_BORDERLINE1'] = pre_rank['BORDERLINE1']
			df_pre['RANK_BORDERLINE2'] = pre_rank['BORDERLINE2']
			df_pre['RANK_GEOMETRIC_SMOTE'] = pre_rank['GEOMETRIC_SMOTE']
			df_pre['RANK_DELAUNAY'] = pre_rank['DELAUNAY']
			
			df_rec = df_rec.reset_index()
			df_rec.drop('index', axis=1, inplace=True)
			df_rec['RANK_ORIGINAL'] = rec_rank['ORIGINAL']
			df_rec['RANK_SMOTE'] = rec_rank['SMOTE']
			df_rec['RANK_SMOTE_SVM'] = rec_rank['SMOTE_SVM']
			df_rec['RANK_BORDERLINE1'] = rec_rank['BORDERLINE1']
			df_rec['RANK_BORDERLINE2'] = rec_rank['BORDERLINE2']
			df_rec['RANK_GEOMETRIC_SMOTE'] = rec_rank['GEOMETRIC_SMOTE']
			df_rec['RANK_DELAUNAY'] = rec_rank['DELAUNAY']
			
			df_spe = df_spe.reset_index()
			df_spe.drop('index', axis=1, inplace=True)
			df_spe['RANK_ORIGINAL'] = spe_rank['ORIGINAL']
			df_spe['RANK_SMOTE'] = spe_rank['SMOTE']
			df_spe['RANK_SMOTE_SVM'] = spe_rank['SMOTE_SVM']
			df_spe['RANK_BORDERLINE1'] = spe_rank['BORDERLINE1']
			df_spe['RANK_BORDERLINE2'] = spe_rank['BORDERLINE2']
			df_spe['RANK_GEOMETRIC_SMOTE'] = spe_rank['GEOMETRIC_SMOTE']
			df_spe['RANK_DELAUNAY'] = spe_rank['DELAUNAY']
			
			df_f1 = df_f1.reset_index()
			df_f1.drop('index', axis=1, inplace=True)
			df_f1['RANK_ORIGINAL'] = f1_rank['ORIGINAL']
			df_f1['RANK_SMOTE'] = f1_rank['SMOTE']
			df_f1['RANK_SMOTE_SVM'] = f1_rank['SMOTE_SVM']
			df_f1['RANK_BORDERLINE1'] = f1_rank['BORDERLINE1']
			df_f1['RANK_BORDERLINE2'] = f1_rank['BORDERLINE2']
			df_f1['RANK_GEOMETRIC_SMOTE'] = f1_rank['GEOMETRIC_SMOTE']
			df_f1['RANK_DELAUNAY'] = f1_rank['DELAUNAY']
			
			df_geo = df_geo.reset_index()
			df_geo.drop('index', axis=1, inplace=True)
			df_geo['RANK_ORIGINAL'] = geo_rank['ORIGINAL']
			df_geo['RANK_SMOTE'] = geo_rank['SMOTE']
			df_geo['RANK_SMOTE_SVM'] = geo_rank['SMOTE_SVM']
			df_geo['RANK_BORDERLINE1'] = geo_rank['BORDERLINE1']
			df_geo['RANK_BORDERLINE2'] = geo_rank['BORDERLINE2']
			df_geo['RANK_GEOMETRIC_SMOTE'] = geo_rank['GEOMETRIC_SMOTE']
			df_geo['RANK_DELAUNAY'] = geo_rank['DELAUNAY']
			
			df_iba = df_iba.reset_index()
			df_iba.drop('index', axis=1, inplace=True)
			df_iba['RANK_ORIGINAL'] = iba_rank['ORIGINAL']
			df_iba['RANK_SMOTE'] = iba_rank['SMOTE']
			df_iba['RANK_SMOTE_SVM'] = iba_rank['SMOTE_SVM']
			df_iba['RANK_BORDERLINE1'] = iba_rank['BORDERLINE1']
			df_iba['RANK_BORDERLINE2'] = iba_rank['BORDERLINE2']
			df_iba['RANK_GEOMETRIC_SMOTE'] = iba_rank['GEOMETRIC_SMOTE']
			df_iba['RANK_DELAUNAY'] = iba_rank['DELAUNAY']
			
			if kind == 'biclass':
				df_auc = df_auc.reset_index()
				df_auc.drop('index', axis=1, inplace=True)
				df_auc['RANK_ORIGINAL'] = auc_rank['ORIGINAL']
				df_auc['RANK_SMOTE'] = auc_rank['SMOTE']
				df_auc['RANK_SMOTE_SVM'] = auc_rank['SMOTE_SVM']
				df_auc['RANK_BORDERLINE1'] = auc_rank['BORDERLINE1']
				df_auc['RANK_BORDERLINE2'] = auc_rank['BORDERLINE2']
				df_auc['RANK_GEOMETRIC_SMOTE'] = auc_rank['GEOMETRIC_SMOTE']
				df_auc['RANK_DELAUNAY'] = auc_rank['DELAUNAY']
			
			# avarege rank
			media_pre_rank = pre_rank.mean(axis=0)
			media_rec_rank = rec_rank.mean(axis=0)
			media_spe_rank = spe_rank.mean(axis=0)
			media_f1_rank = f1_rank.mean(axis=0)
			media_geo_rank = geo_rank.mean(axis=0)
			media_iba_rank = iba_rank.mean(axis=0)
			if kind == 'biclass':
				media_auc_rank = auc_rank.mean(axis=0)
			
			media_pre_rank_file = media_pre_rank.reset_index()
			media_pre_rank_file = media_pre_rank_file.sort_values(by=0)
			
			media_rec_rank_file = media_rec_rank.reset_index()
			media_rec_rank_file = media_rec_rank_file.sort_values(by=0)
			
			media_spe_rank_file = media_spe_rank.reset_index()
			media_spe_rank_file = media_spe_rank_file.sort_values(by=0)
			
			media_f1_rank_file = media_f1_rank.reset_index()
			media_f1_rank_file = media_f1_rank_file.sort_values(by=0)
			
			media_geo_rank_file = media_geo_rank.reset_index()
			media_geo_rank_file = media_geo_rank_file.sort_values(by=0)
			
			media_iba_rank_file = media_iba_rank.reset_index()
			media_iba_rank_file = media_iba_rank_file.sort_values(by=0)
			
			if kind == 'biclass':
				media_auc_rank_file = media_auc_rank.reset_index()
				media_auc_rank_file = media_auc_rank_file.sort_values(by=0)
			
			if smote == False:
				
				# Grava arquivos importantes
				df_pre.to_csv(
						rank_dir + release + '_' + kind + '_total_rank_' + order + '_' + str(
								alpha) + '_' + name + '_pre.csv', index=False)
				df_rec.to_csv(
						rank_dir + release + '_' + kind + '_total_rank_' + order + '_' + str(
								alpha) + '_' + name + '_rec.csv', index=False)
				df_spe.to_csv(
						rank_dir + release + '_' + kind + '_total_rank_' + order + '_' + str(
								alpha) + '_' + name + '_spe.csv', index=False)
				df_f1.to_csv(
						rank_dir + release + '_' + kind + '_total_rank_' + order + '_' + str(
								alpha) + '_' + name + '_f1.csv', index=False)
				df_geo.to_csv(
						rank_dir + release + '_' + kind + '_total_rank_' + order + '_' + str(
								alpha) + '_' + name + '_geo.csv', index=False)
				df_iba.to_csv(
						rank_dir + release + '_' + kind + '_total_rank_' + order + '_' + str(
								alpha) + '_' + name + '_iba.csv', index=False)
				if kind == 'biclass':
					df_auc.to_csv(
							rank_dir + release + '_' + kind + '_total_rank_' + order + '_' + str(
									alpha) + '_' + name + '_auc.csv',
							index=False)
				
				media_pre_rank_file.to_csv(
						rank_dir + release + '_' + 'media_rank_' + kind + '_' + order + '_' + str(
								alpha) + '_' + name + '_pre.csv',
						index=False)
				media_rec_rank_file.to_csv(
						rank_dir + release + '_media_rank_' + kind + '_' + order + '_' + str(
								alpha) + '_' + name + '_rec.csv',
						index=False)
				media_spe_rank_file.to_csv(
						rank_dir + release + '_media_rank_' + kind + '_' + order + '_' + str(
								alpha) + '_' + name + '_spe.csv',
						index=False)
				media_f1_rank_file.to_csv(
						rank_dir + release + '_media_rank_' + kind + '_' + order + '_' + str(
								alpha) + '_' + name + '_f1.csv',
						index=False)
				media_geo_rank_file.to_csv(
						rank_dir + release + '_media_rank_' + kind + '_' + order + '_' + str(
								alpha) + '_' + name + '_geo.csv',
						index=False)
				media_iba_rank_file.to_csv(
						rank_dir + release + '_media_rank_' + kind + '_' + order + '_' + str(
								alpha) + '_' + name + '_iba.csv',
						index=False)
				if kind == 'biclass':
					media_auc_rank_file.to_csv(
							rank_dir + release + '_media_rank_' + kind + '_' + order + '_' + str(
									alpha) + '_' + name + '_auc.csv',
							index=False)
				
				delaunay_type = order + '_' + str(alpha)
				
				# grafico CD
				identificadores = ['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', 'GEOMETRIC_SMOTE',
				                   delaunay_type]
				avranks = list(media_pre_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(dataset))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(
						rank_dir + release + 'cd_' + '_' + kind + '_' + delaunay_type + '_' + name + '_pre.pdf')
				plt.close()
				
				avranks = list(media_rec_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(dataset))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(
						rank_dir + release + 'cd_' + '_' + kind + '_' + delaunay_type + '_' + name + '_rec.pdf')
				plt.close()
				
				avranks = list(media_spe_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(dataset))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(
						rank_dir + release + 'cd_' + '_' + kind + '_' + delaunay_type + '_' + name + '_spe.pdf')
				plt.close()
				
				avranks = list(media_f1_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(dataset))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(rank_dir + release + 'cd_' + '_' + kind + '_' + delaunay_type + '_' + name + '_f1.pdf')
				plt.close()
				
				avranks = list(media_geo_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(dataset))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(
						rank_dir + release + 'cd_' + '_' + kind + '_' + delaunay_type + '_' + name + '_geo.pdf')
				plt.close()
				
				avranks = list(media_iba_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(dataset))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(
						rank_dir + release + 'cd_' + '_' + kind + '_' + delaunay_type + '_' + name + '_iba.pdf')
				plt.close()
				
				if kind == 'biclass':
					avranks = list(media_auc_rank)
					cd = Orange.evaluation.compute_CD(avranks, len(dataset))
					Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
					plt.savefig(
							rank_dir + release + 'cd_' + '_' + kind + '_' + delaunay_type + '_' + name + '_auc.pdf')
					plt.close()
				
				print('Delaunay Type= ', delaunay_type)
				print('Algorithm= ', name)
			
			
			else:
				# Grava arquivos importantes
				df_pre.to_csv(
						rank_dir + release + '_smote_' + kind + '_total_rank_' + order + '_' + str(
								alpha) + '_' + name + '_pre.csv', index=False)
				df_rec.to_csv(
						rank_dir + release + '_smote_' + kind + '_total_rank_' + order + '_' + str(
								alpha) + '_' + name + '_rec.csv', index=False)
				df_spe.to_csv(
						rank_dir + release + '_smote_' + kind + '_total_rank_' + order + '_' + str(
								alpha) + '_' + name + '_spe.csv', index=False)
				df_f1.to_csv(
						rank_dir + release + '_smote_' + kind + '_total_rank_' + order + '_' + str(
								alpha) + '_' + name + '_f1.csv', index=False)
				df_geo.to_csv(
						rank_dir + release + '_smote_' + kind + '_total_rank_' + order + '_' + str(
								alpha) + '_' + name + '_geo.csv', index=False)
				df_iba.to_csv(
						rank_dir + release + '_smote_' + kind + '_total_rank_' + order + '_' + str(
								alpha) + '_' + name + '_iba.csv', index=False)
				if kind == 'biclass':
					df_auc.to_csv(
							rank_dir + release + '_smote_' + kind + '_total_rank_' + order + '_' + str(
									alpha) + '_' + name + '_auc.csv',
							index=False)
				
				media_pre_rank_file.to_csv(
						rank_dir + release + '_smote_media_rank_' + kind + '_' + order + '_' + str(
								alpha) + '_' + name + '_pre.csv',
						index=False)
				media_rec_rank_file.to_csv(
						rank_dir + release + '_smote__media_rank_' + kind + '_' + order + '_' + str(
								alpha) + '_' + name + '_rec.csv',
						index=False)
				media_spe_rank_file.to_csv(
						rank_dir + release + 'smote__media_rank_' + kind + '_' + order + '_' + str(
								alpha) + '_' + name + '_spe.csv',
						index=False)
				media_f1_rank_file.to_csv(
						rank_dir + release + 'smote__media_rank_' + kind + '_' + order + '_' + str(
								alpha) + '_' + name + '_f1.csv',
						index=False)
				media_geo_rank_file.to_csv(
						rank_dir + release + 'smote__media_rank_' + kind + '_' + order + '_' + str(
								alpha) + '_' + name + '_geo.csv',
						index=False)
				media_iba_rank_file.to_csv(
						rank_dir + release + 'smote__media_rank_' + kind + '_' + order + '_' + str(
								alpha) + '_' + name + '_iba.csv',
						index=False)
				if kind == 'biclass':
					media_auc_rank_file.to_csv(
							rank_dir + release + 'smote__media_rank_' + kind + '_' + order + '_' + str(
									alpha) + '_' + name + '_auc.csv',
							index=False)
				
				delaunay_type = order + '_' + str(alpha)
				
				# grafico CD
				identificadores = ['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', 'GEOMETRIC_SMOTE',
				                   delaunay_type]
				avranks = list(media_pre_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(dataset))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(
						rank_dir + release + 'cd_smote' + '_' + kind + '_' + delaunay_type + '_' + name + '_pre.pdf')
				plt.close()
				
				avranks = list(media_rec_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(dataset))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(
						rank_dir + release + 'cd_smote' + '_' + kind + '_' + delaunay_type + '_' + name + '_rec.pdf')
				plt.close()
				
				avranks = list(media_spe_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(dataset))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(
						rank_dir + release + 'cd_smote' + '_' + kind + '_' + delaunay_type + '_' + name + '_spe.pdf')
				plt.close()
				
				avranks = list(media_f1_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(dataset))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(rank_dir + release + 'cd_smote' + '_' + kind + '_' + delaunay_type + '_' + name + '_f1.pdf')
				plt.close()
				
				avranks = list(media_geo_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(dataset))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(
						rank_dir + release + 'cd_smote' + '_' + kind + '_' + delaunay_type + '_' + name + '_geo.pdf')
				plt.close()
				
				avranks = list(media_iba_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(dataset))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(
						rank_dir + release + 'cd_smote' + '_' + kind + '_' + delaunay_type + '_' + name + '_iba.pdf')
				plt.close()
				
				if kind == 'biclass':
					avranks = list(media_auc_rank)
					cd = Orange.evaluation.compute_CD(avranks, len(dataset))
					Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
					plt.savefig(
							rank_dir + release + 'cd_smote' + '_' + kind + '_' + delaunay_type + '_' + name + '_auc.pdf')
					plt.close()
				
				print('SMOTE Delaunay Type= ', delaunay_type)
				print('SMOTE Algorithm= ', name)
	
	def rank_dto_by(self, geometry, kind, release, smote=False):
		if kind == 'biclass':
			M = ['_pre.csv', '_rec.csv', '_spe.csv', '_f1.csv', '_geo.csv', '_iba.csv', '_auc.csv']
		else:
			M = ['_pre.csv', '_rec.csv', '_spe.csv', '_f1.csv', '_geo.csv', '_iba.csv']
		
		df_media_rank = pd.DataFrame(columns=['ALGORITHM', 'RANK_ORIGINAL', 'RANK_SMOTE',
		                                      'RANK_SMOTE_SVM', 'RANK_BORDERLINE1', 'RANK_BORDERLINE2',
		                                      'RANK_GEOMETRIC_SMOTE', 'RANK_DELAUNAY', 'unit'])
		
		if smote == False:
			name = rank_dir + release + '_' + kind + '_total_rank_' + geometry + '_'
		else:
			name = rank_dir + release + '_smote_' + kind + '_total_rank_' + geometry + '_'
		
		for m in M:
			i = 0
			for c in classifiers_list:
				df = pd.read_csv(name + c + m)
				rank_original = df.RANK_ORIGINAL.mean()
				rank_smote = df.RANK_SMOTE.mean()
				rank_smote_svm = df.RANK_SMOTE_SVM.mean()
				rank_b1 = df.RANK_BORDERLINE1.mean()
				rank_b2 = df.RANK_BORDERLINE2.mean()
				rank_geo_smote = df.RANK_GEOMETRIC_SMOTE.mean()
				rank_dto = df.RANK_DELAUNAY.mean()
				df_media_rank.loc[i, 'ALGORITHM'] = df.loc[0, 'ALGORITHM']
				df_media_rank.loc[i, 'RANK_ORIGINAL'] = rank_original
				df_media_rank.loc[i, 'RANK_SMOTE'] = rank_smote
				df_media_rank.loc[i, 'RANK_SMOTE_SVM'] = rank_smote_svm
				df_media_rank.loc[i, 'RANK_BORDERLINE1'] = rank_b1
				df_media_rank.loc[i, 'RANK_BORDERLINE2'] = rank_b2
				df_media_rank.loc[i, 'RANK_GEOMETRIC_SMOTE'] = rank_geo_smote
				df_media_rank.loc[i, 'RANK_DELAUNAY'] = rank_dto
				df_media_rank.loc[i, 'unit'] = df.loc[0, 'unit']
				i += 1
			
			dfmediarank = df_media_rank.copy()
			dfmediarank = dfmediarank.sort_values('RANK_DELAUNAY')
			
			dfmediarank.loc[i, 'ALGORITHM'] = 'avarage'
			dfmediarank.loc[i, 'RANK_ORIGINAL'] = df_media_rank['RANK_ORIGINAL'].mean()
			dfmediarank.loc[i, 'RANK_SMOTE'] = df_media_rank['RANK_SMOTE'].mean()
			dfmediarank.loc[i, 'RANK_SMOTE_SVM'] = df_media_rank['RANK_SMOTE_SVM'].mean()
			dfmediarank.loc[i, 'RANK_BORDERLINE1'] = df_media_rank['RANK_BORDERLINE1'].mean()
			dfmediarank.loc[i, 'RANK_BORDERLINE2'] = df_media_rank['RANK_BORDERLINE2'].mean()
			dfmediarank.loc[i, 'RANK_GEOMETRIC_SMOTE'] = df_media_rank['RANK_GEOMETRIC_SMOTE'].mean()
			dfmediarank.loc[i, 'RANK_DELAUNAY'] = df_media_rank['RANK_DELAUNAY'].mean()
			dfmediarank.loc[i, 'unit'] = df.loc[0, 'unit']
			i += 1
			dfmediarank.loc[i, 'ALGORITHM'] = 'std'
			dfmediarank.loc[i, 'RANK_ORIGINAL'] = df_media_rank['RANK_ORIGINAL'].std()
			dfmediarank.loc[i, 'RANK_SMOTE'] = df_media_rank['RANK_SMOTE'].std()
			dfmediarank.loc[i, 'RANK_SMOTE_SVM'] = df_media_rank['RANK_SMOTE_SVM'].std()
			dfmediarank.loc[i, 'RANK_BORDERLINE1'] = df_media_rank['RANK_BORDERLINE1'].std()
			dfmediarank.loc[i, 'RANK_BORDERLINE2'] = df_media_rank['RANK_BORDERLINE2'].std()
			dfmediarank.loc[i, 'RANK_GEOMETRIC_SMOTE'] = df_media_rank['RANK_GEOMETRIC_SMOTE'].std()
			dfmediarank.loc[i, 'RANK_DELAUNAY'] = df_media_rank['RANK_DELAUNAY'].std()
			dfmediarank.loc[i, 'unit'] = df.loc[0, 'unit']
			
			dfmediarank['RANK_ORIGINAL'] = pd.to_numeric(dfmediarank['RANK_ORIGINAL'], downcast="float").round(2)
			dfmediarank['RANK_SMOTE'] = pd.to_numeric(dfmediarank['RANK_SMOTE'], downcast="float").round(2)
			dfmediarank['RANK_SMOTE_SVM'] = pd.to_numeric(dfmediarank['RANK_SMOTE_SVM'], downcast="float").round(2)
			dfmediarank['RANK_BORDERLINE1'] = pd.to_numeric(dfmediarank['RANK_BORDERLINE1'], downcast="float").round(2)
			dfmediarank['RANK_BORDERLINE2'] = pd.to_numeric(dfmediarank['RANK_BORDERLINE2'], downcast="float").round(2)
			dfmediarank['RANK_GEOMETRIC_SMOTE'] = pd.to_numeric(dfmediarank['RANK_GEOMETRIC_SMOTE'],
			                                                    downcast="float").round(2)
			dfmediarank['RANK_DELAUNAY'] = pd.to_numeric(dfmediarank['RANK_DELAUNAY'], downcast="float").round(2)
			
			if smote == False:
				dfmediarank.to_csv(output_dir + release + '_' + kind + '_results_media_rank_' + geometry + m,
				                   index=False)
			else:
				dfmediarank.to_csv(output_dir + release + '_smote_' + kind + '_results_media_rank_' + geometry + m,
				                   index=False)
	
	def grafico_variacao_alpha(self, kind, release):
		if kind == 'biclass':
			M = ['_geo', '_iba', '_auc']
		else:
			M = ['_geo', '_iba']
		
		order = ['area', 'volume', 'area_volume_ratio', 'edge_ratio', 'radius_ratio', 'aspect_ratio', 'max_solid_angle',
		         'min_solid_angle', 'solid_angle']
		
		# Dirichlet Distribution alphas
		alphas = np.arange(1, 10, 0.5)
		
		df_alpha_variations_rank = pd.DataFrame()
		df_alpha_variations_rank['alphas'] = alphas
		df_alpha_variations_rank.index = alphas
		
		df_alpha_all = pd.DataFrame()
		df_alpha_all['alphas'] = alphas
		df_alpha_all.index = alphas
		
		for m in M:
			for o in order:
				for a in alphas:
					filename = output_dir + release + '_' + kind + '_results_media_rank_' + o + '_' + str(
							a) + m + '.csv'
					print(filename)
					df = pd.read_csv(filename)
					mean = df.loc[8, 'RANK_DELAUNAY']
					df_alpha_variations_rank.loc[a, 'AVARAGE_RANK'] = mean
				
				if m == '_geo':
					measure = 'GEO'
				if m == '_iba':
					measure = 'IBA'
				if m == '_auc':
					measure = 'AUC'
				
				df_alpha_all[o + '_' + measure] = df_alpha_variations_rank['AVARAGE_RANK'].copy()
				
				fig, ax = plt.subplots()
				ax.set_title('DTO AVARAGE RANK\n ' + 'GEOMETRY = ' + o + '\nMEASURE = ' + measure, fontsize=10)
				ax.set_xlabel('Alpha')
				ax.set_ylabel('Rank')
				ax.plot(df_alpha_variations_rank['AVARAGE_RANK'], marker='d', label='Avarage Rank')
				ax.legend(loc="upper right")
				plt.xticks(range(11))
				fig.savefig(output_dir + release + '_' + kind + '_pic_' + o + '_' + measure + '.png', dpi=125)
				plt.show()
				plt.close()
		
		# figure(num=None, figsize=(10, 10), dpi=800, facecolor='w', edgecolor='k')
		
		fig, ax = plt.subplots(figsize=(10, 7))
		ax.set_title('DTO AVARAGE RANK\n ' + '\nMEASURE = GEO', fontsize=5)
		ax.set_xlabel('Alpha')
		ax.set_ylabel('Rank')
		t1 = df_alpha_all['alphas']
		t2 = df_alpha_all['alphas']
		t3 = df_alpha_all['alphas']
		t4 = df_alpha_all['alphas']
		t5 = df_alpha_all['alphas']
		t6 = df_alpha_all['alphas']
		t7 = df_alpha_all['alphas']
		t8 = df_alpha_all['alphas']
		t9 = df_alpha_all['alphas']
		
		ft1 = df_alpha_all['area_GEO']
		ft2 = df_alpha_all['volume_GEO']
		ft3 = df_alpha_all['area_volume_ratio_GEO']
		ft4 = df_alpha_all['edge_ratio_GEO']
		ft5 = df_alpha_all['radius_ratio_GEO']
		ft6 = df_alpha_all['aspect_ratio_GEO']
		ft7 = df_alpha_all['max_solid_angle_GEO']
		ft8 = df_alpha_all['min_solid_angle_GEO']
		ft9 = df_alpha_all['solid_angle_GEO']
		
		ax.plot(t1, ft1, color='tab:blue', marker='o', label='area')
		ax.plot(t2, ft2, color='tab:red', marker='o', label='volume')
		ax.plot(t3, ft3, color='tab:green', marker='o', label='area_volume_ratio')
		ax.plot(t4, ft4, color='tab:orange', marker='o', label='edge_ratio')
		ax.plot(t5, ft5, color='tab:olive', marker='o', label='radius_ratio')
		ax.plot(t6, ft6, color='tab:purple', marker='o', label='aspect_ratio')
		ax.plot(t7, ft7, color='tab:brown', marker='o', label='max_solid_angle')
		ax.plot(t8, ft8, color='tab:pink', marker='o', label='min_solid_angle')
		ax.plot(t9, ft9, color='tab:gray', marker='o', label='solid_angle')
		
		leg = ax.legend(loc='upper right')
		leg.get_frame().set_alpha(0.5)
		plt.xticks(range(12))
		plt.savefig(output_dir + release + '_' + kind + '_pic_all_geo.png', dpi=800)
		plt.show()
		plt.close()
		df_alpha_all.to_csv(output_dir + release + '_' + kind + '_pic_all_geo.csv', index=False)
		
		###################
		fig, ax = plt.subplots(figsize=(10, 7))
		ax.set_title('DTO AVARAGE RANK\n ' + '\nMEASURE = IBA', fontsize=5)
		ax.set_xlabel('Alpha')
		ax.set_ylabel('Rank')
		t1 = df_alpha_all['alphas']
		t2 = df_alpha_all['alphas']
		t3 = df_alpha_all['alphas']
		t4 = df_alpha_all['alphas']
		t5 = df_alpha_all['alphas']
		t6 = df_alpha_all['alphas']
		t7 = df_alpha_all['alphas']
		t8 = df_alpha_all['alphas']
		t9 = df_alpha_all['alphas']
		
		ft1 = df_alpha_all['area_IBA']
		ft2 = df_alpha_all['volume_IBA']
		ft3 = df_alpha_all['area_volume_ratio_IBA']
		ft4 = df_alpha_all['edge_ratio_IBA']
		ft5 = df_alpha_all['radius_ratio_IBA']
		ft6 = df_alpha_all['aspect_ratio_IBA']
		ft7 = df_alpha_all['max_solid_angle_IBA']
		ft8 = df_alpha_all['min_solid_angle_IBA']
		ft9 = df_alpha_all['solid_angle_IBA']
		
		ax.plot(t1, ft1, color='tab:blue', marker='o', label='area')
		ax.plot(t2, ft2, color='tab:red', marker='o', label='volume')
		ax.plot(t3, ft3, color='tab:green', marker='o', label='area_volume_ratio')
		ax.plot(t4, ft4, color='tab:orange', marker='o', label='edge_ratio')
		ax.plot(t5, ft5, color='tab:olive', marker='o', label='radius_ratio')
		ax.plot(t6, ft6, color='tab:purple', marker='o', label='aspect_ratio')
		ax.plot(t7, ft7, color='tab:brown', marker='o', label='max_solid_angle')
		ax.plot(t8, ft8, color='tab:pink', marker='o', label='min_solid_angle')
		ax.plot(t9, ft9, color='tab:gray', marker='o', label='solid_angle')
		
		leg = ax.legend(loc='upper right')
		leg.get_frame().set_alpha(0.5)
		plt.xticks(range(12))
		plt.savefig(output_dir + release + '_' + kind + '_pic_all_iba.png', dpi=800)
		plt.show()
		plt.close()
		df_alpha_all.to_csv(output_dir + release + '_' + kind + '_pic_all_iba.csv', index=False)
		
		if kind == 'biclass':
			fig, ax = plt.subplots(figsize=(10, 7))
			ax.set_title('DTO AVARAGE RANK\n ' + '\nMEASURE = AUC', fontsize=5)
			ax.set_xlabel('Alpha')
			ax.set_ylabel('Rank')
			t1 = df_alpha_all['alphas']
			t2 = df_alpha_all['alphas']
			t3 = df_alpha_all['alphas']
			t4 = df_alpha_all['alphas']
			t5 = df_alpha_all['alphas']
			t6 = df_alpha_all['alphas']
			t7 = df_alpha_all['alphas']
			t8 = df_alpha_all['alphas']
			t9 = df_alpha_all['alphas']
			
			ft1 = df_alpha_all['area_AUC']
			ft2 = df_alpha_all['volume_AUC']
			ft3 = df_alpha_all['area_volume_ratio_AUC']
			ft4 = df_alpha_all['edge_ratio_AUC']
			ft5 = df_alpha_all['radius_ratio_AUC']
			ft6 = df_alpha_all['aspect_ratio_AUC']
			ft7 = df_alpha_all['max_solid_angle_AUC']
			ft8 = df_alpha_all['min_solid_angle_AUC']
			ft9 = df_alpha_all['solid_angle_AUC']
			
			ax.plot(t1, ft1, color='tab:blue', marker='o', label='area')
			ax.plot(t2, ft2, color='tab:red', marker='o', label='volume')
			ax.plot(t3, ft3, color='tab:green', marker='o', label='area_volume_ratio')
			ax.plot(t4, ft4, color='tab:orange', marker='o', label='edge_ratio')
			ax.plot(t5, ft5, color='tab:olive', marker='o', label='radius_ratio')
			ax.plot(t6, ft6, color='tab:purple', marker='o', label='aspect_ratio')
			ax.plot(t7, ft7, color='tab:brown', marker='o', label='max_solid_angle')
			ax.plot(t8, ft8, color='tab:pink', marker='o', label='min_solid_angle')
			ax.plot(t9, ft9, color='tab:gray', marker='o', label='solid_angle')
			
			leg = ax.legend(loc='upper right')
			leg.get_frame().set_alpha(0.5)
			plt.xticks(range(12))
			plt.savefig(output_dir + release + '_' + kind + '_pic_all_auc.png', dpi=800)
			plt.show()
			plt.close()
			df_alpha_all.to_csv(output_dir + release + '_' + kind + '_pic_all_auc.csv', index=False)
	
	def best_alpha(self, kind):
		# Best alpha calculation
		# GEO
		df1 = pd.read_csv(output_dir + 'v1' + '_' + kind + '_pic_all_geo.csv')
		df2 = pd.read_csv(output_dir + 'v2' + '_' + kind + '_pic_all_geo.csv')
		df3 = pd.read_csv(output_dir + 'v3' + '_' + kind + '_pic_all_geo.csv')
		
		if kind == 'biclass':
			col = ['area_GEO', 'volume_GEO', 'area_volume_ratio_GEO',
			       'edge_ratio_GEO', 'radius_ratio_GEO', 'aspect_ratio_GEO',
			       'max_solid_angle_GEO', 'min_solid_angle_GEO', 'solid_angle_GEO',
			       'area_IBA', 'volume_IBA', 'area_volume_ratio_IBA', 'edge_ratio_IBA',
			       'radius_ratio_IBA', 'aspect_ratio_IBA', 'max_solid_angle_IBA',
			       'min_solid_angle_IBA', 'solid_angle_IBA', 'area_AUC', 'volume_AUC',
			       'area_volume_ratio_AUC', 'edge_ratio_AUC', 'radius_ratio_AUC',
			       'aspect_ratio_AUC', 'max_solid_angle_AUC', 'min_solid_angle_AUC',
			       'solid_angle_AUC']
		else:
			col = ['area_GEO', 'volume_GEO',
			       'area_volume_ratio_GEO', 'edge_ratio_GEO', 'radius_ratio_GEO',
			       'aspect_ratio_GEO', 'max_solid_angle_GEO', 'min_solid_angle_GEO',
			       'solid_angle_GEO', 'area_IBA', 'volume_IBA', 'area_volume_ratio_IBA',
			       'edge_ratio_IBA', 'radius_ratio_IBA', 'aspect_ratio_IBA',
			       'max_solid_angle_IBA', 'min_solid_angle_IBA', 'solid_angle_IBA']
		df_mean = pd.DataFrame()
		df_mean['alphas'] = df1.alphas
		for c in col:
			for i in np.arange(0, df1.shape[0]):
				df_mean.loc[i, c] = (df1.loc[i, c] + df2.loc[i, c] + df3.loc[i, c]) / 3.0
		
		fig, ax = plt.subplots(figsize=(10, 7))
		ax.set_title('DTO AVARAGE RANK\n ' + '\nMEASURE = GEO', fontsize=5)
		ax.set_xlabel('Alpha')
		ax.set_ylabel('Rank')
		t1 = df_mean['alphas']
		t2 = df_mean['alphas']
		t3 = df_mean['alphas']
		t4 = df_mean['alphas']
		t5 = df_mean['alphas']
		t6 = df_mean['alphas']
		t7 = df_mean['alphas']
		t8 = df_mean['alphas']
		t9 = df_mean['alphas']
		
		ft1 = df_mean['area_GEO']
		ft2 = df_mean['volume_GEO']
		ft3 = df_mean['area_volume_ratio_GEO']
		ft4 = df_mean['edge_ratio_GEO']
		ft5 = df_mean['radius_ratio_GEO']
		ft6 = df_mean['aspect_ratio_GEO']
		ft7 = df_mean['max_solid_angle_GEO']
		ft8 = df_mean['min_solid_angle_GEO']
		ft9 = df_mean['solid_angle_GEO']
		
		ax.plot(t1, ft1, color='tab:blue', marker='o', label='area')
		ax.plot(t2, ft2, color='tab:red', marker='o', label='volume')
		ax.plot(t3, ft3, color='tab:green', marker='o', label='area_volume_ratio')
		ax.plot(t4, ft4, color='tab:orange', marker='o', label='edge_ratio')
		ax.plot(t5, ft5, color='tab:olive', marker='o', label='radius_ratio')
		ax.plot(t6, ft6, color='tab:purple', marker='o', label='aspect_ratio')
		ax.plot(t7, ft7, color='tab:brown', marker='o', label='max_solid_angle')
		ax.plot(t8, ft8, color='tab:pink', marker='o', label='min_solid_angle')
		ax.plot(t9, ft9, color='tab:gray', marker='o', label='solid_angle')
		
		leg = ax.legend(loc='upper right')
		leg.get_frame().set_alpha(0.5)
		plt.xticks(range(12))
		plt.savefig(output_dir + kind + '_pic_average_geo.png', dpi=800)
		plt.show()
		plt.close()
		df_mean.to_csv(output_dir + kind + '_pic_average_geo.csv', index=False)
		
		###################
		fig, ax = plt.subplots(figsize=(10, 7))
		ax.set_title('DTO AVARAGE RANK\n ' + '\nMEASURE = IBA', fontsize=5)
		ax.set_xlabel('Alpha')
		ax.set_ylabel('Rank')
		t1 = df_mean['alphas']
		t2 = df_mean['alphas']
		t3 = df_mean['alphas']
		t4 = df_mean['alphas']
		t5 = df_mean['alphas']
		t6 = df_mean['alphas']
		t7 = df_mean['alphas']
		t8 = df_mean['alphas']
		t9 = df_mean['alphas']
		
		ft1 = df_mean['area_IBA']
		ft2 = df_mean['volume_IBA']
		ft3 = df_mean['area_volume_ratio_IBA']
		ft4 = df_mean['edge_ratio_IBA']
		ft5 = df_mean['radius_ratio_IBA']
		ft6 = df_mean['aspect_ratio_IBA']
		ft7 = df_mean['max_solid_angle_IBA']
		ft8 = df_mean['min_solid_angle_IBA']
		ft9 = df_mean['solid_angle_IBA']
		
		ax.plot(t1, ft1, color='tab:blue', marker='o', label='area')
		ax.plot(t2, ft2, color='tab:red', marker='o', label='volume')
		ax.plot(t3, ft3, color='tab:green', marker='o', label='area_volume_ratio')
		ax.plot(t4, ft4, color='tab:orange', marker='o', label='edge_ratio')
		ax.plot(t5, ft5, color='tab:olive', marker='o', label='radius_ratio')
		ax.plot(t6, ft6, color='tab:purple', marker='o', label='aspect_ratio')
		ax.plot(t7, ft7, color='tab:brown', marker='o', label='max_solid_angle')
		ax.plot(t8, ft8, color='tab:pink', marker='o', label='min_solid_angle')
		ax.plot(t9, ft9, color='tab:gray', marker='o', label='solid_angle')
		
		leg = ax.legend(loc='upper right')
		leg.get_frame().set_alpha(0.5)
		plt.xticks(range(12))
		plt.savefig(output_dir + kind + '_pic_average_iba.png', dpi=800)
		plt.show()
		plt.close()
		df_mean.to_csv(output_dir + kind + '_pic_average_iba.csv', index=False)
		
		if kind == 'biclass':
			fig, ax = plt.subplots(figsize=(10, 7))
			ax.set_title('DTO AVARAGE RANK\n ' + '\nMEASURE = AUC', fontsize=5)
			ax.set_xlabel('Alpha')
			ax.set_ylabel('Rank')
			t1 = df_mean['alphas']
			t2 = df_mean['alphas']
			t3 = df_mean['alphas']
			t4 = df_mean['alphas']
			t5 = df_mean['alphas']
			t6 = df_mean['alphas']
			t7 = df_mean['alphas']
			t8 = df_mean['alphas']
			t9 = df_mean['alphas']
			
			ft1 = df_mean['area_AUC']
			ft2 = df_mean['volume_AUC']
			ft3 = df_mean['area_volume_ratio_AUC']
			ft4 = df_mean['edge_ratio_AUC']
			ft5 = df_mean['radius_ratio_AUC']
			ft6 = df_mean['aspect_ratio_AUC']
			ft7 = df_mean['max_solid_angle_AUC']
			ft8 = df_mean['min_solid_angle_AUC']
			ft9 = df_mean['solid_angle_AUC']
			
			ax.plot(t1, ft1, color='tab:blue', marker='o', label='area')
			ax.plot(t2, ft2, color='tab:red', marker='o', label='volume')
			ax.plot(t3, ft3, color='tab:green', marker='o', label='area_volume_ratio')
			ax.plot(t4, ft4, color='tab:orange', marker='o', label='edge_ratio')
			ax.plot(t5, ft5, color='tab:olive', marker='o', label='radius_ratio')
			ax.plot(t6, ft6, color='tab:purple', marker='o', label='aspect_ratio')
			ax.plot(t7, ft7, color='tab:brown', marker='o', label='max_solid_angle')
			ax.plot(t8, ft8, color='tab:pink', marker='o', label='min_solid_angle')
			ax.plot(t9, ft9, color='tab:gray', marker='o', label='solid_angle')
			
			leg = ax.legend(loc='upper right')
			leg.get_frame().set_alpha(0.5)
			plt.xticks(range(12))
			plt.savefig(output_dir + kind + '_pic_average_auc.png', dpi=800)
			plt.show()
			plt.close()
			df_mean.to_csv(output_dir + kind + '_pic_average_auc.csv', index=False)
	
	def run_rank_choose_parameters(self, filename, kind, release):
		df_best_dto = pd.read_csv(filename)
		df_B1 = df_best_dto[df_best_dto['PREPROC'] == '_Borderline1'].copy()
		df_B2 = df_best_dto[df_best_dto['PREPROC'] == '_Borderline2'].copy()
		df_GEO = df_best_dto[df_best_dto['PREPROC'] == '_Geometric_SMOTE'].copy()
		df_SMOTE = df_best_dto[df_best_dto['PREPROC'] == '_SMOTE'].copy()
		df_SMOTEsvm = df_best_dto[df_best_dto['PREPROC'] == '_smoteSVM'].copy()
		df_original = df_best_dto[df_best_dto['PREPROC'] == '_train'].copy()
		
		for o in order:
			for a in alphas:
				GEOMETRY = '_delaunay_' + o + '_' + str(a)
				df_dto = df_best_dto[df_best_dto['PREPROC'] == GEOMETRY].copy()
				df = pd.concat([df_B1, df_B2, df_GEO, df_SMOTE, df_SMOTEsvm, df_original, df_dto])
				self.rank_by_algorithm(df, kind, o, str(a), release)
				self.rank_dto_by(o + '_' + str(a), kind, release)
	
	def run_global_rank(self, filename, kind, release):
		df_best_dto = pd.read_csv(filename)
		df_B1 = df_best_dto[df_best_dto['PREPROC'] == '_Borderline1'].copy()
		df_B2 = df_best_dto[df_best_dto['PREPROC'] == '_Borderline2'].copy()
		df_GEO = df_best_dto[df_best_dto['PREPROC'] == '_Geometric_SMOTE'].copy()
		df_SMOTE = df_best_dto[df_best_dto['PREPROC'] == '_SMOTE'].copy()
		df_SMOTEsvm = df_best_dto[df_best_dto['PREPROC'] == '_smoteSVM'].copy()
		df_original = df_best_dto[df_best_dto['PREPROC'] == '_train'].copy()
		o = 'solid_angle'
		if kind == 'biclass':
			a = 7.0
		else:
			a = 7.5
		
		GEOMETRY = '_delaunay_' + o + '_' + str(a)
		df_dto = df_best_dto[df_best_dto['PREPROC'] == GEOMETRY].copy()
		df = pd.concat([df_B1, df_B2, df_GEO, df_SMOTE, df_SMOTEsvm, df_original, df_dto])
		self.rank_by_algorithm(df, kind, o, str(a), release, smote=True)
		self.rank_dto_by(o + '_' + str(a), kind, release, smote=True)
	
	def overall_rank(self, ext, kind, alpha):
		df1 = pd.read_csv(
				output_dir + 'v1_smote_' + kind + '_results_media_rank_solid_angle_' + str(alpha) + '_' + ext + '.csv')
		df2 = pd.read_csv(
				output_dir + 'v2_smote_' + kind + '_results_media_rank_solid_angle_' + str(alpha) + '_' + ext + '.csv')
		df3 = pd.read_csv(
				output_dir + 'v3_smote_' + kind + '_results_media_rank_solid_angle_' + str(alpha) + '_' + ext + '.csv')
		
		col = ['RANK_ORIGINAL', 'RANK_SMOTE', 'RANK_SMOTE_SVM', 'RANK_BORDERLINE1'
			, 'RANK_BORDERLINE2', 'RANK_GEOMETRIC_SMOTE', 'RANK_DELAUNAY']
		
		df_mean = pd.DataFrame()
		df_mean['ALGORITHM'] = df1.ALGORITHM
		df_mean['unit'] = df1.unit
		for c in col:
			for i in np.arange(0, df1.shape[0]):
				df_mean.loc[i, c] = (df1.loc[i, c] + df2.loc[i, c] + df3.loc[i, c]) / 3.0
		
		df_mean['RANK_ORIGINAL'] = pd.to_numeric(df_mean['RANK_ORIGINAL'], downcast="float").round(2)
		df_mean['RANK_SMOTE'] = pd.to_numeric(df_mean['RANK_SMOTE'], downcast="float").round(2)
		df_mean['RANK_SMOTE_SVM'] = pd.to_numeric(df_mean['RANK_SMOTE_SVM'], downcast="float").round(2)
		df_mean['RANK_BORDERLINE1'] = pd.to_numeric(df_mean['RANK_BORDERLINE1'], downcast="float").round(2)
		df_mean['RANK_BORDERLINE2'] = pd.to_numeric(df_mean['RANK_BORDERLINE2'], downcast="float").round(2)
		df_mean['RANK_GEOMETRIC_SMOTE'] = pd.to_numeric(df_mean['RANK_GEOMETRIC_SMOTE'], downcast="float").round(2)
		df_mean['RANK_DELAUNAY'] = pd.to_numeric(df_mean['RANK_DELAUNAY'], downcast="float").round(2)
		
		df_mean.to_csv(output_dir + 'overall_rank_results_' + kind + '_' + str(alpha) + '_' + ext + '.csv', index=False)
	
	def cd_graphics(self, df, datasetlen, kind):  # TODO
		# grafico CD
		names = ['ORIGINAL', 'SMOTE', 'SMOTE_SVM', 'BORDERLINE1', 'BORDERLINE2', 'GEOMETRIC_SMOTE', 'DTO']
		algorithms = classifiers_list
		
		for i in np.arange(0, len(algorithms)):
			avranks = list(df.loc[i])
			algorithm = avranks[0]
			measure = avranks[1]
			avranks = avranks[2:]
			cd = Orange.evaluation.compute_CD(avranks, datasetlen)
			Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=len(algorithms), textspace=3)
			plt.savefig(output_dir + kind + '_cd_' + algorithm + '_' + measure + '.pdf')
			plt.close()
