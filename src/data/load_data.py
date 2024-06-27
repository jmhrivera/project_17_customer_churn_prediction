import pandas as pd

def load_datasets():
    '''This function will upload the necessary datasets
    to perform the project.'''
    contract_df = pd.read_csv('datasets/contract.csv')
    internet_df = pd.read_csv('datasets/internet.csv')
    personal_df = pd.read_csv('datasets/personal.csv')
    phone_df = pd.read_csv('datasets/phone.csv')
    return contract_df, internet_df, personal_df, phone_df
