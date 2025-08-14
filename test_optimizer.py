import pandas as pd
from optimizer import optimize_portfolio
from ml_model import compute_rolling_volatility, train_risk_model, predict_volatility

# Load CSV
data = pd.read_csv("data/AAPL_stock_data.csv", parse_dates=[0], index_col=0)

# Ensure 'Close' column is numeric
data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
data = data.dropna(subset=['Close'])

# Compute daily returns
returns = data["Close"].pct_change().dropna().to_frame()

# Optimize portfolio
weights, expected_return, risk = optimize_portfolio(returns)
print("Optimal Weights:", weights)
print("Expected Portfolio Return:", expected_return)
print("Portfolio Risk (Variance):", risk)

# Phase 4: Risk Forecasting
vol = compute_rolling_volatility(returns["Close"], window=5)
print("Rolling Volatility (last 5 days):\n", vol.tail())

model = train_risk_model(vol)
forecast = predict_volatility(model, vol[-5:])
print(f"Forecasted Next-Period Volatility: {forecast:.4f}")
