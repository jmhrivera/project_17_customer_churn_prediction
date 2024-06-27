from data.load_data import load_datasets
from data.preprocess import preprocess_data
from eda.eda_report import eda_report
from feature_engineering.feature_engineering import ft_engineering
from models.build_models import iterative_modeling
import pandas as pd

def main():
    '''This main function progresses through various stages to process data, 
    evaluate variables, and create a robust model for predicting churned users. 
    For more detailed information, please refer to the README.md file. '''

    data = load_datasets() #Loading stage
    preprocessed_data = preprocess_data(data) #Preprocessing stage
    eda_report(preprocessed_data) # Analysis stage
    processed_data = ft_engineering(preprocessed_data) # Feature engineering stage
    results = iterative_modeling(processed_data) # Modeling stage
    df_results = pd.DataFrame(results, columns=['model','best_estimator','best_train_score','validation_score'])
    return df_results

results = main()



