import Orange
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datasets import dataset_biclass, dataset_multiclass

output_dir = './../output/'
rank_dir = './../rank/'
# geometry
order = ['area',  # ok
         'volume',  # ok
         'area_volume_ratio',  # ok
         'edge_ratio',  # ok
         'radius_ratio',  # ok
         'aspect_ratio',  # ok
         'max_solid_angle',
         'min_solid_angle',
         'solid_angle']

# Dirichlet Distribution alphas
alphas = np.arange(1, 10, 0.5)


class Performance:
	
	def __init__(self):
		pass
	
	def average_results(self, rfile, kind, version):
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
			mode = group.loc[0, 'MODE']
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
			print(i)
		
		if kind == 'biclass':
			dfr.to_csv(output_dir + 'average_results_biclass_' + str(version) + '.csv', index=False)
		else:
			dfr.to_csv(output_dir + 'average_results_multiclass_' + str(version) + '.csv', index=False)
	
	def rank_by_algorithm(self, df, kind, order, alpha):
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
			df.to_csv(rank_dir + '_' + kind + '_' + order + '_' + str(alpha) + '.csv')
			
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
			
			# Grava arquivos importantes
			df_pre.to_csv(
					rank_dir + '_total_rank_' + kind + '_' + order + '_' + str(
							alpha) + '_' + name + '_pre.csv',
					index=False)
			df_rec.to_csv(
					rank_dir + '_total_rank_' + kind + '_' + order + '_' + str(
							alpha) + '_' + name + '_rec.csv',
					index=False)
			df_spe.to_csv(
					rank_dir + '_total_rank_' + kind + '_' + order + '_' + str(
							alpha) + '_' + name + '_spe.csv',
					index=False)
			df_f1.to_csv(rank_dir + '_total_rank_' + kind + '_' + order + '_' + str(
					alpha) + '_' + name + '_f1.csv',
			             index=False)
			df_geo.to_csv(
					rank_dir + '_total_rank_' + kind + '_' + order + '_' + str(
							alpha) + '_' + name + '_geo.csv',
					index=False)
			df_iba.to_csv(
					rank_dir + '_total_rank_' + kind + '_' + order + '_' + str(
							alpha) + '_' + name + '_iba.csv',
					index=False)
			if kind == 'biclass':
				df_auc.to_csv(
						rank_dir + '_total_rank_' + kind + '_' + order + '_' + str(
								alpha) + '_' + name + '_auc.csv',
						index=False)
			
			media_pre_rank_file.to_csv(
					rank_dir + '_media_rank_' + kind + '_' + order + '_' + str(
							alpha) + '_' + name + '_pre.csv',
					index=False)
			media_rec_rank_file.to_csv(
					rank_dir + '_media_rank_' + kind + '_' + order + '_' + str(
							alpha) + '_' + name + '_rec.csv',
					index=False)
			media_spe_rank_file.to_csv(
					rank_dir + '_media_rank_' + kind + '_' + order + '_' + str(
							alpha) + '_' + name + '_spe.csv',
					index=False)
			media_f1_rank_file.to_csv(
					rank_dir + '_media_rank_' + kind + '_' + order + '_' + str(
							alpha) + '_' + name + '_f1.csv',
					index=False)
			media_geo_rank_file.to_csv(
					rank_dir + '_media_rank_' + kind + '_' + order + '_' + str(
							alpha) + '_' + name + '_geo.csv',
					index=False)
			media_iba_rank_file.to_csv(
					rank_dir + '_media_rank_' + kind + '_' + order + '_' + str(
							alpha) + '_' + name + '_iba.csv',
					index=False)
			if kind == 'biclass':
				media_auc_rank_file.to_csv(
						rank_dir + '_media_rank_' + kind + '_' + order + '_' + str(
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
					rank_dir + 'cd_' + '_' + kind + '_' + delaunay_type + '_' + name + '_pre.pdf')
			plt.close()
			
			avranks = list(media_rec_rank)
			cd = Orange.evaluation.compute_CD(avranks, len(dataset))
			Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
			plt.savefig(
					rank_dir + 'cd_' + '_' + kind + '_' + delaunay_type + '_' + name + '_rec.pdf')
			plt.close()
			
			avranks = list(media_spe_rank)
			cd = Orange.evaluation.compute_CD(avranks, len(dataset))
			Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
			plt.savefig(
					rank_dir + 'cd_' + '_' + kind + '_' + delaunay_type + '_' + name + '_spe.pdf')
			plt.close()
			
			avranks = list(media_f1_rank)
			cd = Orange.evaluation.compute_CD(avranks, len(dataset))
			Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
			plt.savefig(rank_dir + 'cd_' + '_' + kind + '_' + delaunay_type + '_' + name + '_f1.pdf')
			plt.close()
			
			avranks = list(media_geo_rank)
			cd = Orange.evaluation.compute_CD(avranks, len(dataset))
			Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
			plt.savefig(
					rank_dir + 'cd_' + '_' + kind + '_' + delaunay_type + '_' + name + '_geo.pdf')
			plt.close()
			
			avranks = list(media_iba_rank)
			cd = Orange.evaluation.compute_CD(avranks, len(dataset))
			Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
			plt.savefig(
					rank_dir + 'cd_' + '_' + kind + '_' + delaunay_type + '_' + name + '_iba.pdf')
			plt.close()
			
			if kind == 'biclass':
				avranks = list(media_auc_rank)
				cd = Orange.evaluation.compute_CD(avranks, len(dataset))
				Orange.evaluation.graph_ranks(avranks, identificadores, cd=cd, width=9, textspace=3)
				plt.savefig(
						rank_dir + 'cd_' + '_' + kind + '_' + delaunay_type + '_' + name + '_auc.pdf')
				plt.close()
			
			print('Delaunay Type= ', delaunay_type)
			print('Algorithm= ', name)
	
	def run_rank(self, filename, kind):
		df = pd.read_csv(filename)
		df_B1 = df[df['PREPROC'] == '_Borderline1'].copy()
		df_B2 = df[df['PREPROC'] == '_Borderline2'].copy()
		df_GEO = df[df['PREPROC'] == '_Geometric_SMOTE'].copy()
		df_SMOTE = df[df['PREPROC'] == '_SMOTE'].copy()
		df_SMOTESVM = df[df['PREPROC'] == '_smoteSVM'].copy()
		df_original = df[df['PREPROC'] == '_train'].copy()
		
		for o in order:
			for a in alphas:
				GEOMETRY = '_delaunay_' + o + '_' + str(a)
				df_dto = df[df['PREPROC'] == GEOMETRY].copy()
				df = pd.concat([df_B1, df_B2, df_GEO, df_SMOTE, df_SMOTESVM, df_original, df_dto])
				self.rank_by_algorithm(df, kind, o, str(a))
