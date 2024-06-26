import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, root_mean_squared_error, classification_report, accuracy_score, mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.compose import ColumnTransformer

## Logistic Regression Model

def all_models():

    lr_pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=10000))
    ])

    lr_param_grid = {
            'logreg__penalty': [ 'l1', 'l2', None],  # Regularización
            # 'logreg__C': [0.01, 0.1, 1, 10, 100],  # Fuerza de la regularización
            'logreg__solver': ['saga'], # ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # Algoritmo de optimización
            # 'logreg__l1_ratio': np.linspace(0, 1, 10)  # Solo si el solver es 'saga' y penalty es 'elasticnet'
        }

    lr = ['Logreg',lr_pipeline,lr_param_grid]

    xg_pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('xgboost', XGBClassifier(random_state=1234))
    ])

    xg_param_grid = {
        'xgboost__max_depth': [3, 5, 7],  # Profundidad máxima del árbol
        'xgboost__learning_rate': [0.1, 0.01, 0.001],  # Tasa de aprendizaje
        'xgboost__n_estimators': [100, 500, 1000],  # Número de árboles en el bosque
    }

    xg = ['XGboost',xg_pipeline,xg_param_grid]
    # models = [lr,xg]
    models = [lr]
    return models
