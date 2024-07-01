from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from src.models.hyper_parameters import all_models
import pandas as pd 


def iterative_modeling(data):
    '''This function will bring the hyper parameters from all_model() 
    and wil create a complete report of the best model, estimator, 
    score and validation score'''

    models = all_models() 
    output_path = './src/models/results/model_report.csv'
    results = []

    # Iterating the models
    for model in models:
        best_estimator, best_score, val_score = model_structure(data, model[1], model[2]) #data, pipeline, param_grid
        results.append([model[0],best_estimator,best_score, val_score])
        
    results_df = pd.DataFrame(results, columns=['model','best_estimator','best_train_score','validation_score'])
    results_df.to_csv(output_path,index=False)
    return results


def model_structure(data, pipeline, param_grid):
    '''This function will host the structure to run all the models, splitting the
    dataset, oversampling the data and returning the scores'''

    X = data.drop(columns='EndDate')
    y = data['EndDate']
    # X_test = data[data['EndDate']==0]

    X_train, X_val, y_train, y_val = train_test_split(X,y, random_state=1234, test_size=0.20)

    # Oversampling
    smote = SMOTE(random_state=1234)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train,y_train) 

    # Training the model
    gs = GridSearchCV(pipeline, param_grid, cv=2, scoring='roc_auc', n_jobs=-1, verbose=2)
    gs.fit(X_train_resampled,y_train_resampled)

    # Scores
    best_score = gs.best_score_
    best_estimator = gs.best_estimator_
    pred_val = gs.predict_proba(X_val)[:,1]
    score_val = evaluate_model(y_val,pred_val)
    print(f'AU-ROC: {score_val}')
    results = best_estimator, best_score, score_val 
    return results
    
def evaluate_model(y,y_pred):
    roc_auc = roc_auc_score(y,y_pred)
    return roc_auc