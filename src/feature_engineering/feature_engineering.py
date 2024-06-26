from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

# Funciones
def OHE(df):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_columns = encoder.fit_transform(df)
    return pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(), index=df.index)

def scaler(columns):
    scaler = StandardScaler()
    scaled_cols = scaler.fit_transform(columns)
    return scaled_cols

def correlation(data):

    output_path = './src/feature_engineering/correlation_results/'
    corr = (data).corr()
    plt.figure(figsize=(12,12))
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.savefig(output_path+'corr_heatmap.png')
    # plt.show()
    corr_results = (corr['EndDate']*100).sort_values(ascending=False)
    # print(abs(corr_results))
    results = pd.DataFrame(corr_results)
    results.to_csv(output_path+'corr_results.csv')
    selected_columns = corr_results[abs(corr_results)>19].index #Modificar a criterio
    return selected_columns

def feature_engineering(data):
    
    # Eliminar columna customerID
    data = data.drop(columns=['CustomerID', 'BeginDate'])
    # Descomponer start date en columnas mes y año

    # Transformar end date en 0,1
    data['EndDate'] = np.where(data['EndDate'] =='No',0,1)

    ## Imputación de valores __________________________

    ##
    # Imputando por medio de ML
    # Transformar columnas con OHE
    # PENDIENTE
    ##

    # Imputación por Media 
    #######TEMPORAL 
    numeric = data.select_dtypes(include='number').columns
    categoric = data.select_dtypes(exclude='number').columns
    
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    impute_numeric = pd.DataFrame(imp.fit_transform(data[numeric]), columns=numeric)
    
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    impute_categoric = pd.DataFrame(imp.fit_transform(data[categoric]), columns=categoric)

    ## Transformación de datos _____________________
    
    ## Pasando valores categóricos a numéricos
    numeric_encoded = OHE(impute_categoric)
    merge2= pd.concat([impute_numeric,numeric_encoded], axis=1)

    ## Análisis de correlación
    selected_columns = correlation(merge2)

    return merge2[selected_columns]
    # return merge2

