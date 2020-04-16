from performance import Performance


def main():
    analisys = Performance()
    analisys.average_results('./../input/dto_multiclass_results_1.csv',kind='multiclass',version=1)
    analisys.verfify_alpha_performance('./../output/avarage_results_multiclass_1.csv')
    
    
    
    

if __name__ == '__main__':
    main()