import os
from src.data.load_data import load_datasets
from src.data.preprocess import preprocess_data
from src.eda.eda_report import eda_report


def main():
    data = load_datasets()
    preprocessed_data = preprocess_data(data)
    preprocessed_data.to_csv('./datasets/merge.csv')
    eda_report(preprocessed_data)

    
x = main()


