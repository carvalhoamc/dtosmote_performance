from performance import Performance
import pandas as pd

results_biclass_files = ['avarage_results_biclass_1.csv',
                         'avarage_results_biclass_2.csv',
                         'avarage_results_biclass_3.csv']

results_multiclass_files = ['avarage_results_multiclass_1.csv']


def main():
    analisys = Performance()
    #analisys.average_results('./../input/dto_multiclass_results_v1.csv',kind='multiclass',version=1)
    analisys.run_rank('./../input/average_results_multiclass_1.csv',kind='multiclass')

    
if __name__ == '__main__':
    main()