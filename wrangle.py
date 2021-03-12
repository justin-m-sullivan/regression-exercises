import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
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

def train_validate_test_split(df, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed)

    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed)
    return train, validate, test
