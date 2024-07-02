from sklearn.preprocessing import StandardScaler
from .ml_imputation import ml_imputation, OHE
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def scaler(columns):
    '''Function to normalize numeric values'''
    scaler = StandardScaler()
    scaled_cols = scaler.fit_transform(columns)
    return scaled_cols

def correlation(data):
    '''This function assists in selecting the columns for modeling 
    by identifying the columns that have the highest positive and
    negative correlations with the `EndDate'.'''
    
    output_path = './results/correlation_results/'
    corr = (data).corr()
    
    plt.figure(figsize=(12,12))
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.savefig(output_path+'corr_heatmap.png')
    
    corr_results = (corr['EndDate']*100).sort_values(ascending=False)
    results = pd.DataFrame(corr_results)
    results.to_csv(output_path+'corr_results.csv')
    selected_columns = corr_results[abs(corr_results)>10].index #Modify ad criteria
    return selected_columns


def ft_engineering(data):
    
    # Removing CustomerID and BeginDate
    data = data.drop(columns=['CustomerID', 'BeginDate'])

    # Transforming EndDate into binary
    data['EndDate'] = np.where(data['EndDate'] =='No',0,1)

    # Transforming columns into binary.
    yn_columns= ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Partner',
       'Dependents', 'MultipleLines']

    for col in data[yn_columns]:
        data[col] = data[col].map({'Yes': 1, 'No': 0})

    # # Imputation by mean _______
    # Method 1 Imputing by SimpleImputer
    
    # Enable at criteria, if you do, please disable method 2

    # numeric = data.select_dtypes(include='number').columns
    # categoric = data.select_dtypes(exclude='number').columns
    # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    # impute_numeric = pd.DataFrame(imp.fit_transform(data[numeric]), columns=numeric)
    # imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    # impute_categoric = pd.DataFrame(imp.fit_transform(data[categoric]), columns=categoric)

    ## Impute Values __________________________
    # Method 2 Imputing by Machine Learning Model
    data = ml_imputation(data)

    ## Feature Engineering _____________________
    ## Tranforming categoric to numeric values
    numeric = data.select_dtypes(include='number')
    categoric = data.select_dtypes(exclude='number')

    numeric_encoded = OHE(categoric)
    imputed_merge = pd.concat([numeric,numeric_encoded], axis=1)
 
    ## Correlation Analysis
    selected_columns = correlation(imputed_merge)

    # Exporting DataFrame

    output_path = './datasets/output/'
    imputed_merge.to_csv(output_path + 'imputed_df',index=False)

    return imputed_merge[selected_columns]

