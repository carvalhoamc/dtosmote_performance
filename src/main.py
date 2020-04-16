from performance import Performance


def main():
    analisys = Performance()
    analisys.average_results('./../input/dto_multiclass_results_1.csv',kind='multiclass')
    
    
    
    
    

if __name__ == '__main__':
    main()