import numpy as np
import pandas as pd
import os
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

