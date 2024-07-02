import pandas as pd
import os

def preprocess_data(data):
    '''This function will clean the data by setting removing duplicates, 
    formatting the column types, names and removing incoherent data. The datasets
    will be merged in one joined by the CustomerID'''
    contract_df = data[0] 
    internet_df = data[1] 
    personal_df = data[2] 
    phone_df    = data[3] 
     
    # Contract preprocessing
    contract_df.rename(columns={'customerID': 'CustomerID'}, inplace=True)
    contract_df['BeginDate'] = pd.to_datetime(contract_df['BeginDate'], format='%Y-%m-%d')
    contract_df['Type'] = contract_df['Type'].astype('category')
    contract_df['PaymentMethod'] = contract_df['PaymentMethod'].astype('category')
    # Convert TotalCharges to float
    contract_df['TotalCharges'] = pd.to_numeric(contract_df['TotalCharges'], errors='coerce')
    
    # Internet preprocessing
    internet_df.rename(columns={'customerID': 'CustomerID'}, inplace=True)
    # Personal preprocessing
    personal_df.rename(columns={'customerID': 'CustomerID', 'gender': 'Gender'}, inplace=True)
    # Phone preprocessing
    phone_df.rename(columns={'customerID': 'CustomerID'}, inplace=True)
    
    # Merging data
    merged_df = pd.merge(contract_df, internet_df, how='outer', on='CustomerID')
    merged_df = pd.merge(merged_df, personal_df, how='outer', on='CustomerID')
    merged_df = pd.merge(merged_df, phone_df, how='outer', on='CustomerID')

    
    path = './datasets/output/'

    if not os.path.exists(path):
        os.makedirs(path)

    merged_df.to_csv(path+'merge.csv', index=False)

    print(f'Dataframe created at route: {path}merge.csv ')

    return merged_df

