import yfinance as yf
import pandas as pd

def fetch_data(ticker, start, end):
    # Download data from Yahoo Finance
    data = yf.download(ticker, start=start, end=end, progress=False)
    
    # Keep only required columns
    data = data[["Open", "High", "Low", "Close", "Volume"]]
    
    # Save to CSV with proper Date column
    data.to_csv(f"data/{ticker}_stock_data.csv", index=True, index_label="Date")
    print(f"Data for {ticker} saved to data/{ticker}_stock_data.csv")
    return data

if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    fetch_data(ticker, start_date, end_date)
