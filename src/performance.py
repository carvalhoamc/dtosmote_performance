import pandas as pd
import numpy as np

output_dir = './../output/'
#geometry
order = ['area',#ok
         'volume',#ok
         'area_volume_ratio',#ok
         'edge_ratio',#ok
         'radius_ratio',#ok
         'aspect_ratio',#ok
         'max_solid_angle',
         'min_solid_angle',
         'solid_angle']

# Dirichlet Distribution alphas
alphas = np.arange(1,10,0.5)

class Performance:
	
	def __init__(self):
		pass
	
	def average_results(self, rfile, kind,version):
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
			dfr.to_csv(output_dir + 'average_results_biclass_'+str(version)+'.csv', index=False)
		else:
			dfr.to_csv(output_dir + 'average_results_multiclass_'+str(version)+'.csv', index=False)
			
			
	def verfify_alpha_performance(self,filename):
		
		df = pd.read_csv(filename)
		df_B1 = df[df['PREPROC']=='_Borderline1'].copy()
		df_B2 = df[df['PREPROC'] == '_Borderline2'].copy()
		df_GEO = df[df['PREPROC'] == '_Geometric_SMOTE'].copy()
		df_SMOTE = df[df['PREPROC'] == '_SMOTE'].copy()
		df_SMOTEsvm = df[df['PREPROC'] == '_smoteSVM'].copy()
		df_original = df[df['PREPROC'] == '_train'].copy()
		
		for o in order:
			for a in alphas:
				GEOMETRY = '_delaunay_'+ o + '_'+str(a)
				df_dto = df_best_dto[df_best_dto['PREPROC'] == GEOMETRY].copy()
				df = pd.concat([df_B1,df_B2,df_GEO,df_SMOTE,df_SMOTEsvm,df_original,df_dto])
				diag.rank_by_algorithm(df, 'multiclasse', './../output_dir/multiclass/', 'pca', o, str(a))
				diag.rank_dto_by(o + '_'+ str(a))
