from sklearn.preprocessing import OneHotEncoder, StandardScaler,
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt


#temporal
data = pd.read_csv('./datasets/merge.csv') 

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
    corr = (data).corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.show()
    corr_results = (corr['EndDate']*100).sort_values(ascending=False)
    corr_results
#   selected_columns = corr_results[corr_results>19].index




def feature_engineering(data):
    
    # Eliminar columna customerID
    data = data.drop(columns=['CustomerID', 'BeginDate'])
    # Descomponer start date en columnas mes y año

    # Transformar end date en 0,1
    data['EndDate'] = np.where(data['EndDate'] =='No',0,1)

    # Mandar distribution type y method al final por tener 3 respuestas

    # Transformar columnas con OHE

    ## Imputación de valores __________________________
    # Imputando por medio de ML

    #######TEMPORAL 
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imputed_merge = pd.DataFrame(imp.fit_transform(data), columns=data.columns)

    ## Transformación de datos _____________________

    ## Pasando valores categóricos a numéricos
    numeric = imputed_merge.select_dtypes(include='number')
    categoric = imputed_merge.select_dtypes(exclude='number').columns

    OHE(categoric)

x = feature_engineering(data)
print(x)

