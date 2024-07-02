import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import sys

def eda_report(data):
    '''Te EDA report will create some files to analyze the in deep the variables of the table.
     The elements will be divided by categoric and numeric and some extra info will printed'''

    # Data Summary ___________________

    # Creating a describing report
    describe_result = data.describe()

    eda_path = './results/eda_results/'

    # Exporting the file
    with open(eda_path+'describe.txt', 'w') as f:
        f.write(describe_result.to_string())

    # Exporting general info
    with open(eda_path+'info.txt','w') as f:
        sys.stdout = f
        data.info()
        sys.stdout = sys.__stdout__
       
    # Variables Chart ___________________
    ## Plotting numeric chart
    numeric = data.select_dtypes(include='number')
    num_height = math.ceil(len(numeric.columns)/3)
    fig1, axes = plt.subplots(num_height, 3, figsize=(12,12))

    for column, ax in zip(numeric.columns, axes.flatten()): 
        ax.hist(x=data[column])
        ax.set_title(f'{column} Histogram')
     
    plt.tight_layout()
    fig1.savefig(eda_path+'numeric.png')

    # Plotting categoric chart
    categoric = data.select_dtypes(exclude='number')
    categoric = categoric.iloc[:,2:]
    cat_height = math.ceil(len(categoric.columns)/3)
        
    fig2, axes = plt.subplots(cat_height, 3, figsize=(18,18))

    for column, ax in zip(categoric.columns, axes.flatten()):
        sns.countplot(x=column,data=categoric,ax=ax)
        ax.set_title(f'Distribución de {column}', fontsize=16)
        ax.set_xlabel('Categoría')
        ax.set_ylabel('Conteo')
        
        # Adding rotation to avoid overlay text
        if column in ('EndDate','PaymentMethod'):
            ax.tick_params(axis='x', labelrotation=45)

            
    plt.tight_layout()
    fig2.savefig(eda_path+'categoric.png')

    print(f'EDA report created at route: {eda_path}')
