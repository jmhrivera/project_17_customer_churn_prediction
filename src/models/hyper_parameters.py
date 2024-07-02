from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

def imputation_params(model):
    '''Function exclusivo for Imputing, will select the parameters for the type
    of model selected'''

    if model == 'linreg':
        lr_pipeline = Pipeline([('linear_reg', LinearRegression())])
        lr_param_grid = {}

    if model == 'logreg':
        lr_pipeline = Pipeline([('logreg', LogisticRegression(max_iter=10000))])
        lr_param_grid = {
        'logreg__penalty': [ 'l1', 'l2', 'none'],  # Regularización
        # 'logreg__C': [0.01, 0.1, 1, 10, 100],  # Fuerza de la regularización
        'logreg__solver': ['saga'], # ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # Algoritmo de optimización
        # 'logreg__l1_ratio': np.linspace(0, 1, 10)  # Solo si el solver es 'saga' y penalty es 'elasticnet'
        }

    return lr_pipeline, lr_param_grid

## Logistic Regression Model
def all_models():
    '''This function will host all the model parameters, can be used to iterate the
    grid search '''

    lr_pipeline = Pipeline([
        ('scale', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=10000))
    ])

    lr_param_grid = {
            'logreg__penalty': [ 'l1', 'l2', 'none'],  # Regularización
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
    
    lgbm_pipeline = Pipeline([
        ('scale', StandardScaler()),
        # ('preprocessor', preprocessor), 80
        ('lightgbm', LGBMClassifier())
    ])

    lgbm_param_grid = {
        'lightgbm__max_depth': [3, 5, 7],  # Profundidad máxima del árbol
        'lightgbm__learning_rate': [0.1, 0.01, 0.001],  # Tasa de aprendizaje
        'lightgbm__n_estimators': [100, 500, 1000],  # Número de árboles en el bosque
        
    }
    lgbm = ['LGBM',lgbm_pipeline,lgbm_param_grid]
        
    
    rf_pipeline = Pipeline([
    ('scale', StandardScaler()),
    # ('preprocessor', preprocessor),
    ('random_forest', RandomForestClassifier(random_state=1234))])

    # Crear el grid de parámetros para Random Forest
    rf_param_grid = {
        'random_forest__n_estimators': [100, 500, 1000],  # Número de árboles en el bosque
        'random_forest__max_depth': ['none', 10, 20, 30],  # Profundidad máxima del árbol
        'random_forest__min_samples_split': [2, 5, 10],  # Número mínimo de muestras requeridas para dividir un nodo interno
        'random_forest__min_samples_leaf': [1, 2, 4],  # Número mínimo de muestras requeridas para estar en un nodo hoja
    }

    # Evaluar el modelo con la función model_evaluation
    rf = ['Random_Forest',rf_pipeline,rf_param_grid]
        
    # models = [lr,xg,lgbm,rf] Activate to run all the models
    # models = [lr,xg,lgbm]
    models = [lr]

    return models

