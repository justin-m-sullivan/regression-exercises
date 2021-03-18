import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from env import host, user, password

# Establish a connection
def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the CodeUp db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

# Acquire Data
def get_telco_data():
    '''
    This function reads in telco data from Codeup database.
    It returns a single dataframe containing the columns:
    customer_id, monthly_charges, tenure, and total_charges.
    '''
    df = pd.read_sql('''
        SELECT customer_id, monthly_charges, tenure, total_charges
        FROM customers
        WHERE contract_type_id = 3;
        ''', 
        get_connection('telco_churn'))
    return df

def clean_telco_data(df):
    '''
    This function takes in the dataframe from the get_telco_data function
    and prepares the df for analysis by:
    1) Replacing empty strings in total_charges with 0.00
    2) Converting total_charges to float64 datatype
    '''
    #Fill empty values in total_charges column where tenure is less than 1 month
    df.total_charges = df.total_charges.replace(' ', 0.0)

    #Convert total_charges to float
    df.total_charges = df.total_charges.astype('float64')

    #Reset index
    df.set_index('customer_id', drop=True, inplace=True)

    return df

def wrangle_telco():
    '''
    This function acquires the teclo data set from the the codeup database
    and prepares it for analysis by:     
    1) Replacing empty strings in total_charges with 0.00
    2) Converting total_charges to float64 datatype
    '''
    df = get_telco_data()
    df = clean_telco_data(df)

    return df

def train_validate_test_split(df, target):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=123)

    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=123)

    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test

def minmax_scale_data(X_train, X_validate, X_test):
    scaler = MinMaxScaler(copy=True).fit(X_train)

    #scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train. 
    # 
    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled, 
                                  columns=X_train.columns).\
                                  set_index([X_train.index.values])

    X_validate_scaled = pd.DataFrame(X_validate_scaled, 
                                     columns=X_validate.columns).\
                                     set_index([X_validate.index.values])

    
    X_test_scaled = pd.DataFrame(X_test_scaled, 
                                 columns=X_test.columns).\
                                 set_index([X_test.index.values])                              

    return X_train_scaled, X_validate_scaled, X_test_scaled

def select_kbest(X_train_scaled, y_train, k):

    f_selector = SelectKBest(f_regression, k=k)

    # find the top 2 X's correlated with y
    f_selector.fit(X_train_scaled, y_train)

    # boolean mask of whether the column was selected or not. 
    feature_mask = f_selector.get_support()

    # get list of top K features. 
    f_feature = X_train_scaled.iloc[:,feature_mask].columns.tolist()

    print(f'The top best features for predicting the target are: {f_feature}')

def rfe(X_train_scaled, y_train, k):
    # initialize the ML algorithm
    lm = LinearRegression()

    # create the rfe object, indicating the ML object (lm) and the number of features I want to end up with. 
    rfe = RFE(lm, k)

    # fit the data using RFE
    rfe.fit(X_train_scaled,y_train)  

    # get the mask of the columns selected
    feature_mask = rfe.support_

    # get list of the column names. 
    rfe_feature = X_train_scaled.iloc[:,feature_mask].columns.tolist()

    print(f'The best features for predicting the target are: {rfe_feature}')
    