import pandas as pd
import numpy as np

def preprocess_data(data):
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
    return merged_df

