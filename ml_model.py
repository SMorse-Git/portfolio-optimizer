# ml_model.py
import pandas as pd
from sklearn.linear_model import LinearRegression

def compute_rolling_volatility(returns, window=5):
    """
    Compute rolling volatility of returns.
    """
    return returns.rolling(window=window).std().dropna()

def train_risk_model(volatility_series, window=5):
    """
    Train a linear regression model to predict next-period volatility.
    """
    X = []
    y = []
    for i in range(len(volatility_series) - window):
        X.append(volatility_series.iloc[i:i+window].values)
        y.append(volatility_series.iloc[i+window])
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model

def predict_volatility(model, last_vols):
    """
    Predict next-period volatility using trained model.
    """
    return model.predict([last_vols.values])[0]
