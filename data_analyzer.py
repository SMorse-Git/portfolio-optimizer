import pandas as pd

# Read the CSV file with proper Date column
data = pd.read_csv("data/AAPL_stock_data.csv", parse_dates=[0], index_col=0)

# Ensure numeric columns
numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Calculate 20-day moving average
data["MA20"] = data["Close"].rolling(window=20).mean()

# Print summary statistics
print("Summary statistics:")
print(data.describe())

# Optional: save the analyzed data
data.to_csv("data/AAPL_stock_data_analyzed.csv")
print("Analyzed data saved to data/AAPL_stock_data_analyzed.csv")
