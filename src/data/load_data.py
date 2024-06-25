import pandas as pd

def load_datasets():
    contract_df = pd.read_csv('datasets/contract.csv')
    internet_df = pd.read_csv('datasets/internet.csv')
    personal_df = pd.read_csv('datasets/personal.csv')
    phone_df = pd.read_csv('datasets/phone.csv')
    return contract_df, internet_df, personal_df, phone_df
