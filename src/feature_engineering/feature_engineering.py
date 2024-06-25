import pandas as pd


#temporal
data = pd.read_csv('./datasets/merge.csv') 

def feature_engineering(data):
    
    # Descomponer start date en columnas mes y año
    # Transformar end date en 0,1
    # mandar distribution type y method al final por tener 3 respuestas
    # Transformar columnas con OHE
    