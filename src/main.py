from parameters import releases, input_dir
from performance import Performance
import pandas as pd



def main():
    analisys = Performance()
    for r in releases:
        analisys.average_results(input_dir+'dto_multiclass_results_'+r+'.csv',kind='multiclass',release=r)
        #analisys.run_rank('./../input/average_results_multiclass_'+r+'.csv',kind='multiclass',release=r)
        #analisys.grafico_variacao_alpha(kind='multiclass',release=r)

    
if __name__ == '__main__':
    main()