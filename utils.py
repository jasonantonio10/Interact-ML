from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from flask import jsonify
import pandas as pd
import numpy as np


#df = pd.read_csv('bank-data.csv',sep=',')
#col = df.columns

def check_duplicated(df):
    return df.duplicated().sum()

def check_null_values(df, list_col):
    cols_with_null = []
    for col in list_col:
        if df[col].isnull().sum() > 0:
            cols_with_null.append(col)
    
    #print(cols_with_null)
    #print(df.isnull().any())
    #print(df.isnull().sum())
    print(cols_with_null)
    return cols_with_null

def extractFeaturesTarget(df, target_column, columns_ignored):
    df = df.drop(columns=columns_ignored)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    return X, y

def split_train_test(X, y, test_size=0.25, random_state=42, shuffle=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state, shuffle)
    return X_train, X_test, y_train, y_test 
    
def extractNumerical(data, numerical_cols):
    numerical_data = data[numerical_cols]

    return numerical_data

def impute_numerical(numerical_data, strategy='mean'):
    imputer = SimpleImputer(missing_values=np.nan, strategy=strategy)
    imputer.fit(numerical_data)

    imputed_data_raw = imputer.transform(numerical_data)
    numerical_imputed_data = pd.DataFrame(imputed_data_raw, columns=numerical_data.columns, index=numerical_data.index)

    return numerical_imputed_data, imputer



def extractCategorical(data, categorical_cols):
    categorical_data = data[categorical_cols]

    return categorical_data

def impute_categorical(categorical_data):
    categorical_imputed_data = categorical_data.fillna(value="UNKNOWN")

    return categorical_imputed_data
def one_hot_encoding(categorical_data, dtype=int):
    cat_ohe_data = pd.get_dummies(data=categorical_data, dtype=dtype)
    return cat_ohe_data, cat_ohe_data.columns

def standardizer_data(data):
    data_columns = data.columns
    data_index = data.index

    standardizer = StandardScaler()
    standardizer.fit(data)


    standardized_data_raw = standardizer.transform(data)
    standardized_data = pd.DataFrame(standardized_data_raw, columns=data_columns, index=data_index)

    return standardized_data, standardizer






#print(check_null_values(df, col))
#print(check_duplicated(df))
#def preprocess_data(df, features, target):
