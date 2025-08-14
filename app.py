import streamlit as st
import pandas as pd
from data_fetcher import fetch_data
from optimizer import optimize_portfolio
from ml_model import compute_rolling_volatility, train_risk_model, predict_volatility
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder
import plotly.express as px
import time

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="📊 Portfolio Optimizer", layout="wide")

# =========================
# DARK THEME + UI ANIMATIONS
# =========================
st.markdown(
    """
    <style>
    /* Dark background for the app */
    .main {
        background-color: #0E0E0E;
        color: #FFFFFF;
    }

    /* Center the Optimize button */
    div.stButton > button {
        display: block;
        margin: 0 auto;
        background-color: #1E1E1E;
        color: #FFFFFF;
        border-radius: 12px;
        padding: 0.6em 2em;
        font-size: 18px;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #FF4C4C;
        color: #FFFFFF;
        transform: scale(1.05);
    }

    /* Progress bar custom animation */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #FF4C4C, #FFB347);
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% {opacity: 0.8;}
        50% {opacity: 1;}
        100% {opacity: 0.8;}
    }

    /* Fade-in for metrics */
    .stMetric {
        animation: fadeMetric 1s ease-in;
    }
    @keyframes fadeMetric {
        from {opacity: 0; transform: translateY(5px);}
        to {opacity: 1; transform: translateY(0);}
    }

    /* Fade-in for header */
    .fadeInHeader {
        animation: fadeIn 2s;
    }
    @keyframes fadeIn {
        from {opacity:0;}
        to {opacity:1;}
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# HEADER
# =========================
st.markdown(
    "<div class='fadeInHeader' style='text-align:center;'><h1>📊 Portfolio Optimizer with Risk Forecasting</h1></div>",
    unsafe_allow_html=True
)

# =========================
# INPUTS
# =========================
with st.sidebar:
    st.header("⚙️ Input Parameters")
    tickers_input = st.text_input("Enter tickers (comma-separated):", "AAPL")
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    
    start_date = st.date_input("Start date", value=datetime(2024, 1, 1))
    end_date = st.date_input("End date", value=datetime(2024, 12, 31))
    
    uploaded_file = st.file_uploader("Upload CSV of historical data", type=["csv"])

# =========================
# FETCH DATA
# =========================
if uploaded_file:
    data = pd.read_csv(uploaded_file, parse_dates=[0], index_col=0)
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    data = data.dropna(subset=['Close'])
    returns = data["Close"].pct_change().dropna().to_frame()
else:
    close_prices = pd.DataFrame()
    for ticker in tickers:
        df = fetch_data(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        close_prices[ticker] = df["Close"]
    returns = close_prices.pct_change().dropna()

# =========================
# OPTIMIZATION BUTTON
# =========================
if st.button("Optimize Portfolio"):
    # show progress bar
    progress_bar = st.progress(0)
    for percent in range(0, 101, 10):
        time.sleep(0.05)
        progress_bar.progress(percent)
    
    # OPTIMIZE
    weights, expected_return, risk = optimize_portfolio(returns)
    
    # =========================
    # LAYOUT: TABS
    # =========================
    tab1, tab2, tab3 = st.tabs(["Portfolio Weights", "Volatility", "Forecast"])
    
    with tab1:
        st.subheader("📌 Optimal Portfolio Weights")
        df_weights = pd.DataFrame({
            "Ticker": tickers,
            "Weight": weights.round(4)
        })
        gb = GridOptionsBuilder.from_dataframe(df_weights)
        gb.configure_default_column(editable=True)
        AgGrid(df_weights, gridOptions=gb.build(), height=250)
        
        st.markdown(f"**Expected Portfolio Return:** {expected_return:.4f} (~{expected_return*100:.2f}% per day)")
        st.markdown(f"**Portfolio Risk (Variance):** {risk:.4f} (~{risk**0.5*100:.2f}% daily std dev)")
    
    with tab2:
        st.subheader("📈 Rolling Portfolio Volatility")
        vol_series = compute_rolling_volatility(returns.mean(axis=1))
        fig = px.line(vol_series, title="Portfolio Rolling Volatility", labels={"index": "Date", "value": "Volatility"})
        fig.update_traces(mode="lines+markers", line=dict(color='firebrick', width=3))
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📉 Forecasted Portfolio Risk")
        risk_model = train_risk_model(vol_series)
        forecast = predict_volatility(risk_model, vol_series[-5:])
        st.metric("Next-Period Forecasted Volatility", f"{forecast:.4f}")
        st.write("""
        **Interpretation:**
        - Expected Return: ~0.13% per day gain
        - Risk: Daily return fluctuates with ~1.41% standard deviation
        """)
