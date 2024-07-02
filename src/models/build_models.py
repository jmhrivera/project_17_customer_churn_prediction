import pandas as pd 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from src.models.hyper_parameters import all_models


def iterative_modeling(data):
    '''This function will bring the hyper parameters from all_model() 
    and wil create a complete report of the best model, estimator, 
    score and validation score'''

    models = all_models() 
    output_path = './results/model_results/model_report.csv'
    results = []

    # Iterating the models
    for model in models:
        best_estimator, best_score, val_score = model_structure(data, model[1], model[2]) #data, pipeline, param_grid
        results.append([model[0],best_estimator,best_score, val_score])

    results_df = pd.DataFrame(results, columns=['model','best_estimator','best_train_score','validation_score'])
    
    tf_results = tens_flow(data)    
    
    # Concatening logistic models and neuronal network
    final_rev = pd.concat([results_df,tf_results])
    final_rev.to_csv(output_path,index=False)

    return final_rev[['model','validation_score']]


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

## Network Model Structure

def build_model(X_train):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),  # Dropout para regularización
        Dense(32, activation='relu'),
        Dropout(0.2),  # Más dropout para regularización
        Dense(1, activation='sigmoid')
    ])
    return model

def tens_flow(data):
    
    # Defining element and objective
    X_train = data.drop(columns='EndDate')
    y_train = data['EndDate']
    
    # Scaling TotalCharges and MonthlyCharges
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    

    # Splitting into X_train, X_val, y_train, y_val
    X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)
    X_val_scaled = scaler.transform(X_val)

    # Compiling the model
    model = build_model(X_train)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Training the model using GPU if available
    with tf.device('/GPU:0'):  
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, 
                            validation_data=(X_val, y_val), callbacks=[early_stopping])

    # Evaluating the model
    y_pred = model.predict(X_val)
    auc_score = roc_auc_score(y_val, y_pred)
    print(f"AU-ROC Score: {auc_score}")
    results = ['Keras',auc_score]
    results_df = pd.DataFrame({'model':[results[0]],'validation_score':[results[1]]})

    return results_df