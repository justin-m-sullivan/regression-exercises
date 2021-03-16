import pandas as pd
import numpy as np

from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression 
from math import sqrt
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from statsmodels.formula.api import ols

def plot_residuals(df, target, feature):

    residual = df.residual
    baseline_residual = df.baseline_residual
    
    sns.set(style="white", palette="muted", color_codes=True) 
    
    f, axes = plt.subplots(1, 2, figsize=(12, 7), sharex=True)
    sns.despine(left=True)

    sns.regplot(x=feature, y=residual, data=df, ax=axes[0])
    plt.title("Residual")
    plt.tight_layout()

    sns.scatterplot(x=feature, y=baseline_residual, data=df, ax=axes[1])
    plt.axhline(target.mean(), ls = ':')
    plt.title("Baseline Residual")

    plt.tight_layout()

def regression_errors(df):
    
    df['residual^2'] = df.residual**2
    df['baseline_residual^2'] = df.baseline_residual**2
    
    #Return sum of squared errors
    SSE = df['residual^2'].sum()
    print("SSE = ", SSE)
    
    #Mean Squared Error
    MSE = SSE/len(df)
    print("MSE = ", MSE)
    

    #Root mean squared error
    RMSE = sqrt(MSE)
    print("RMSE = ", RMSE)
    
    # Total Sum of Squares = SSE for baseline
    TSS = df['baseline_residual^2'].sum() 
    print("TSS = ", TSS)
    
    #Explained Sum of Squares
    ESS = TSS - SSE
    
    print("ESS =", ESS)