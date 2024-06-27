
import pandas as pd
import numpy as np
from models.hyper_parameters import imputation_params
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import  accuracy_score, root_mean_squared_error
from sklearn.impute import SimpleImputer

def OHE(df):
    '''Function to encode categoric values'''
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_columns = encoder.fit_transform(df)
    return pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(), index=df.index)

def categoric_to_num(df):
    '''This function will divide the DataFrame into categoric and numeric values,
    categoric will be encoded to numeric and joined to the original numeric values'''
    numeric = df.select_dtypes(include='number')
    categoric = df.select_dtypes(exclude='number')
    numeric_encoded = OHE(categoric)
    imputed_merge = pd.concat([numeric,numeric_encoded], axis=1)
    return imputed_merge

def training_imp_model(X,y,type,pipeline,param_grid,X_test=None):
    '''A model created to impute linear and logistic values if the
      respective metric aligns with the threshold.'''
    
    X_train, X_val, y_train, y_val = train_test_split(X,y, random_state=1234, test_size=0.20)
    print(f'Columna: {y.name}')
    
    if type=='linreg':
        print('Model: Linear')
        grid_search = GridSearchCV(pipeline, param_grid, cv=2, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)
        grid_search.fit(X_train,y_train)
        print(grid_search.best_params_)
        y_pred = grid_search.predict(X_val)
        rmse = root_mean_squared_error(y_val,y_pred)
        print("RMSE: ", rmse)

        ## Threshold

        if X_test is not None and rmse<100:
            pred = grid_search.predict(X_test)
            return pred
        elif X_test is not None and rmse>100:
             print(f'Not proceeding, It is recommended to use SimpleImputer instead for {y.name}')

    elif type=='logreg':
            print('Model: Logistic')
            smote = SMOTE(random_state=1234)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train,y_train) 

            grid_search = GridSearchCV(pipeline, param_grid, cv=2, scoring='accuracy', n_jobs=-1, verbose=2)
            grid_search.fit(X_train_resampled,y_train_resampled)
            print(grid_search.best_params_)
            
            y_pred = grid_search.predict(X_val)
            acc = accuracy_score(y_val,y_pred)
            print(f'Accuracy: : {acc*100:.2f}%') 

            if X_test is not None and acc>0.65:
                print(f'Accepted changes for {y.name}, imputing null values')
                pred = grid_search.predict(X_test)
                return pred

    else:
        print('No model has been selected')
        return


def ml_imputation(data):
    '''This function will encode, model and predict the null values'''

    # # Splitting by rows without nulls for training
    rows_without_nan = data.dropna()

    # Defining the training and prediction columns for train (X_train, y_train)
    cols_without_nan = data.columns[~data.isna().any()]
    impute_df = rows_without_nan[cols_without_nan]
    
    # Defining the training and prediction columns for test (X_test, y_test)
    rows_with_nan = data[data.isna().any(axis=1)]
    test_df = rows_with_nan[cols_without_nan]

    # DataFrame for training without null values and normalized (X_train)
    cols_with_nan = data.columns[data.isna().any()]

    X_train = categoric_to_num(impute_df)
    y_train = rows_without_nan[cols_with_nan]

    X_test = categoric_to_num(test_df)

    # Column for linear prediction
    pipeline ,param_grid = imputation_params('linreg')
    rmse = training_imp_model(X_train,y_train.iloc[:,0],'linreg', pipeline, param_grid, X_test)
 
    # Imputing Linear columns (TotalCharges)
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    data['TotalCharges'] = imp.fit_transform(np.array(data['TotalCharges']).reshape(-1,1))

    # Creating an empty data frame with the null values index
    imputated_nulls= pd.DataFrame(index=X_test.index)

    # Columns for logistic prediction
    pipeline ,param_grid = imputation_params('logreg')
    logistic_columns = y_train.columns[1:]

    # Imputing logistic columns
    for col in logistic_columns:
         imputated_nulls[col] = training_imp_model(X_train,y_train[col],'logreg', pipeline, param_grid, X_test)     
         print('')

    indices = imputated_nulls.index
    columns = imputated_nulls.columns
    data.loc[indices,columns] = imputated_nulls

    return data