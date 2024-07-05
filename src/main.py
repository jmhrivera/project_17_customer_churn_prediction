# In case of debugging enable and select your path
import sys
# path = 'path_of_the_project'
# sys.path.append('path_of_the_project')
# path= '/home/carnivurus/Documents/Tripleten/project17_final'
# sys.path.append(path)


from src.data_loader.load_data import load_datasets
from src.data_loader.preprocess import preprocess_data
from src.eda.eda_report import eda_report
from src.feature_engineering.feature_engineering import ft_engineering
from src.models.build_models import iterative_modeling
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
    return results

results = main()

print(results)

