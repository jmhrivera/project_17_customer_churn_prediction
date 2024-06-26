import pandas as pd    
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, root_mean_squared_error, classification_report, accuracy_score, mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from src.models.hyper_parameters import all_models




def iterative_modeling(data):
    models = all_models() 
    output_path = './src/models/results/model_report.csv'
    results = []
    for model in models:
        best_estimator, best_score, val_score = model_structure(data, model[1], model[2]) #data, pipeline, param_grid
        results.append([model[0],best_estimator,best_score, val_score])
        
    results_df = pd.DataFrame(results, columns=['model','best_estimator','best_train_score','validation_score'])
    results_df.to_csv(output_path,index=False)
    return results



def model_structure(data, pipeline, param_grid):
    X = data.drop(columns='EndDate')
    y = data['EndDate']
    X_test = data[data['EndDate']==0]

    X_train, X_val, y_train, y_val = train_test_split(X,y, random_state=1234, test_size=0.20)

    smote = SMOTE(random_state=1234)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train,y_train) 

    gs = GridSearchCV(pipeline, param_grid, cv=2, scoring='roc_auc', n_jobs=-1, verbose=2)
    gs.fit(X_train_resampled,y_train_resampled)
    best_score = gs.best_score_
    best_estimator = gs.best_estimator_
    
    pred_val = gs.predict(X_val)
    score_val = evaluate_model(pred_val,y_val)

    results = best_estimator, best_score, score_val 

    return results
    
def evaluate_model(y,y_pred):
    roc_auc = roc_auc_score(y,y_pred)
    # print("AUC-ROC: ", roc_auc)
    return roc_auc