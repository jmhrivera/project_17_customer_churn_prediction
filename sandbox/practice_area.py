import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.impute import SimpleImputer
# import numpy as np
# import statsmodels.api as sm


path = './src/feature_engineering/results/imputed_df'
path2 = './datasets/merge.csv'
data = pd.read_csv(path)

data.columns
