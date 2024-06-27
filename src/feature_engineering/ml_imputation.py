
import pandas as pd
import numpy as np
from models.hyper_parameters import imputation_params
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import  accuracy_score, root_mean_squared_error
from sklearn.impute import SimpleImputer

# file = './datasets/merge.csv'
# data= pd.read_csv(file)
def OHE(df):
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_columns = encoder.fit_transform(df)
    return pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(), index=df.index)

def categoric_to_num(df):
    numeric = df.select_dtypes(include='number')
    categoric = df.select_dtypes(exclude='number')
    numeric_encoded = OHE(categoric)
    imputed_merge = pd.concat([numeric,numeric_encoded], axis=1)
    return imputed_merge

def training_imp_model(X,y,type,pipeline,param_grid,X_test=None):

    X_train, X_val, y_train, y_val = train_test_split(X,y, random_state=1234, test_size=0.20)
    print(f'Columna: {y.name}')
    
    if type=='linreg':
        print('Modelo: Linear')
        grid_search = GridSearchCV(pipeline, param_grid, cv=2, scoring='neg_root_mean_squared_error', n_jobs=-1, verbose=2)
        grid_search.fit(X_train,y_train)
        print(grid_search.best_params_)
        y_pred = grid_search.predict(X_val)
        rmse = root_mean_squared_error(y_val,y_pred)
        print("RMSE: ", rmse)

        if X_test is not None and rmse<100:
            pred = grid_search.predict(X_test)
            return pred
        elif X_test is not None and rmse>100:
             print(f'Se recomienda usar SimpleImputer en la columna {y.name}')

    elif type=='logreg':
            print('Modelo: Logístico')
            smote = SMOTE(random_state=1234)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train,y_train) 

            grid_search = GridSearchCV(pipeline, param_grid, cv=2, scoring='accuracy', n_jobs=-1, verbose=2)
            grid_search.fit(X_train_resampled,y_train_resampled)
            print(grid_search.best_params_)
            
            y_pred = grid_search.predict(X_val)
            acc = accuracy_score(y_val,y_pred)
            print(f'Accuracy: : {acc*100:.2f}%') 

            if X_test is not None and acc>0.65:
                print(f'modelo aceptado para {y.name}, sustiyendo valores nulos')
                pred = grid_search.predict(X_test)
                return pred

    else:
        print('No haz seleccionado un modelo valido')
        return


def ml_imputation(data):

    # # Depurando columnas innecesarias 
    # data= data.drop(columns=['CustomerID','BeginDate']) #Eliminar cuando se vincule todo el código
    # data['EndDate'] = np.where(data['EndDate'] =='No',0,1)  #Eliminar cuando se vincule todo el código

    # Dividiento por filas sin nulos para el entrenamiento
    rows_without_nan = data.dropna()

    # Definiendo las columnas de entrenamiento y predicción para train (X_train,y_train)
    cols_without_nan = data.columns[~data.isna().any()]
    impute_df = rows_without_nan[cols_without_nan]
    
    # Definiendo las columnas de entrenamiento y predicción para test (X_test, y_test)
    rows_with_nan = data[data.isna().any(axis=1)]
    test_df = rows_with_nan[cols_without_nan]

    # Dataframe de entrenamiento sin valores nulos y normalizado X_train
    cols_with_nan = data.columns[data.isna().any()]

    X_train = categoric_to_num(impute_df)
    y_train = rows_without_nan[cols_with_nan]

    X_test = categoric_to_num(test_df)

    # Columna para predicción linear
    pipeline ,param_grid = imputation_params('linreg')
    rmse = training_imp_model(X_train,y_train.iloc[:,0],'linreg', pipeline, param_grid, X_test)
 
    # Imputando columnas lineares (TotalCharges)
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    data['TotalCharges'] = imp.fit_transform(np.array(data['TotalCharges']).reshape(-1,1))

    # Creamos un df con el mismo número de filas con valores nulos
    imputated_nulls= pd.DataFrame(index=X_test.index)

    # Columnas para predicción logística 
    pipeline ,param_grid = imputation_params('logreg')
    logistic_columns = y_train.columns[1:]

    # # Imputando columnas logísticas
    for col in logistic_columns:
         imputated_nulls[col] = training_imp_model(X_train,y_train[col],'logreg', pipeline, param_grid, X_test)     
         print('')

    indices = imputated_nulls.index
    columns = imputated_nulls.columns
    data.loc[indices,columns] = imputated_nulls

    

    return data
 
# data = ml_imputation(data)