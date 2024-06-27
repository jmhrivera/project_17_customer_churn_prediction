from sklearn.preprocessing import StandardScaler
from feature_engineering.ml_imputation import ml_imputation, OHE
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Funciones

def scaler(columns):
    scaler = StandardScaler()
    scaled_cols = scaler.fit_transform(columns)
    return scaled_cols

def correlation(data):
    output_path = './src/feature_engineering/results/'
    corr = (data).corr()
    plt.figure(figsize=(12,12))
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.savefig(output_path+'corr_heatmap.png')
    # plt.show()
    corr_results = (corr['EndDate']*100).sort_values(ascending=False)
    # print(abs(corr_results))
    results = pd.DataFrame(corr_results)
    results.to_csv(output_path+'corr_results.csv')
    # selected_columns = corr_results[corr_results>25].index #Modificar a criterio
    # selected_columns = corr_results[corr_results<-25].index #Modificar a criterio
    selected_columns = corr_results[abs(corr_results)>17].index #Modificar a criterio


    return selected_columns


def ft_engineering(data):
    
    # Eliminar columna customerID
    data = data.drop(columns=['CustomerID', 'BeginDate'])
    # Descomponer start date en columnas mes y año

    # Transformar end date en 0,1
    data['EndDate'] = np.where(data['EndDate'] =='No',0,1)

    # Transformamos las columnas, No,Yes en 0,1
    yn_columns= ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Partner',
       'Dependents', 'MultipleLines']

    for col in data[yn_columns]:
        data[col] = data[col].map({'Yes': 1, 'No': 0})

    ## Imputación de valores __________________________
    # Imputando por medio de ML
    data = ml_imputation(data)

    # # Imputación por Media _______
    # En caso de querer imputar por media habilitar este espacio y deshabilitar imputación por ML

    # numeric = data.select_dtypes(include='number').columns
    # categoric = data.select_dtypes(exclude='number').columns
    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # impute_numeric = pd.DataFrame(imp.fit_transform(data[numeric]), columns=numeric)
    # imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    # impute_categoric = pd.DataFrame(imp.fit_transform(data[categoric]), columns=categoric)

    ## Transformación de datos _____________________
    ## Pasando valores categóricos a numéricos
    numeric = data.select_dtypes(include='number')
    categoric = data.select_dtypes(exclude='number')

    numeric_encoded = OHE(categoric)
    imputed_merge = pd.concat([numeric,numeric_encoded], axis=1)
 
    ## Análisis de correlación
    selected_columns = correlation(imputed_merge)

    # return imputed_merge[selected_columns]
    # Exportando df
    output_path = './src/feature_engineering/results/'
    imputed_merge.to_csv(output_path + 'imputed_df',index=False)

    return imputed_merge[selected_columns]

