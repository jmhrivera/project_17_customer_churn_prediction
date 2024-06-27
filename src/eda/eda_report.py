import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import sys

def eda_report(data):
    # Resumen de datos ___________________

    # Creamos un reporte para describe 
    describe_result = data.describe()

    # Exportamos describe
    with open('./src/eda/files/describe.txt', 'w') as f:
        f.write(describe_result.to_string())

    # Exportamos info
    with open('./src/eda/files/info.txt','w') as f:
        sys.stdout = f
        data.info()
        sys.stdout = sys.__stdout__
       
    # Grafico de variables ___________________
    ## Mostrar variables numéricas

    numeric = data.select_dtypes(include='number')
    num_height = math.ceil(len(numeric.columns)/3)
    fig1, axes = plt.subplots(num_height, 3, figsize=(12,12))

    for column, ax in zip(numeric.columns, axes.flatten()): 
        ax.hist(x=data[column])
        ax.set_title(f'{column} Histogram')
     
    plt.tight_layout()
    fig1.savefig('./src/eda/files/numeric.png')

    # Mostrar variables categóricas
    categoric = data.select_dtypes(exclude='number')
    categoric = categoric.iloc[:,2:]
    cat_height = math.ceil(len(categoric.columns)/3)
    
    # Graficamos    
    fig2, axes = plt.subplots(cat_height, 3, figsize=(12,12))

    for column, ax in zip(categoric.columns, axes.flatten()):
        sns.countplot(x=column,data=categoric,ax=ax)
        ax.set_title(f'Distribución de {column}')
        ax.set_xlabel('Categoría')
        ax.set_ylabel('Conteo')

    plt.tight_layout()
    fig2.savefig('./src/eda/files/categoric.png')

    print('EDA report created at route: /src/eda/files')
