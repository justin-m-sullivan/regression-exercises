import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer

import wrangle

def scale_telco(df):
    '''
    Scale_telco wrangles the telco dataframe from the codeup database,
    splits the df into three data sets (train, validate, test), and scales
    the data using SKLEARN's Min Max Scaler. It returns three datasets:
    train_scaled, validate_scaled, test_scaled
    '''
    df = wrangle.wrangle_telco()

    train, validate, test = wrangle.train_validate_test_split(df)

    scaler = sklearn.preprocessing.MinMaxScaler()

    scaler.fit(train)

    train_scaled = scaler.transform(train)
    validate_scaled = scaler.transform(validate)
    test_scaled = scaler.transform(test)

    train_scaled = pd.DataFrame(train_scaled, columns=train.columns)
    validate_scaled = pd.DataFrame(validate_scaled, columns=train.columns)
    test_scaled = pd.DataFrame(test_scaled, columns=train.columns)

    return train_scaled, validate_scaled, test_scaled