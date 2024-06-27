import os
from data.load_data import load_datasets
from data.preprocess import preprocess_data
from eda.eda_report import eda_report
from feature_engineering.feature_engineering import ft_engineering
from models.build_models import iterative_modeling


import pandas as pd

def main():

    data = load_datasets()
    preprocessed_data = preprocess_data(data)
    eda_report(preprocessed_data)

    processed_data = ft_engineering(preprocessed_data)

    results = iterative_modeling(processed_data)
    df_results = pd.DataFrame(results, columns=['model','best_estimator','best_train_score','validation_score'])
    return df_results

results = main()



