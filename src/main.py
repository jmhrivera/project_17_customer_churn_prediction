import os
from src.data.load_data import load_datasets
from src.data.preprocess import preprocess_data
from src.eda.eda_report import eda_report
from src.feature_engineering.feature_engineering import feature_engineering
from src.models.build_models import model_structure, iterative_modeling

import pandas as pd

def main():

    data = load_datasets()
    preprocessed_data = preprocess_data(data)
    preprocessed_data.to_csv('./datasets/merge.csv', index=False)
    eda_report(preprocessed_data)
    processed_data = feature_engineering(preprocessed_data)
    iterative_modeling(processed_data)
    


x = main()


