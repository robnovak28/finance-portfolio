import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data_fetcher import (get_sp500_tickers, get_fundamentals, 
                          download_price_data, calculate_momentum)
from factor_model import compute_factor_scores, build_portfolio
from backtest import simple_backtest

st.set_page_config(page_title='Factor Stock Screener', layout='wide')
st.title('S&P 500 Factor-Based Stock Screener')

# Sidebar controls
st.sidebar.header('Factor Weights')
value_w = st.sidebar.slider('Value Weight', 0.0, 1.0, 0.30, 0.01)
mom_w = st.sidebar.slider('Momentum Weight', 0.0, 1.0, 0.40, 0.01)
qual_w = st.sidebar.slider('Quality Weight', 0.0, 1.0, 0.30, 0.01)
n_stocks = st.sidebar.slider('Portfolio Size', 5, 50, 20)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    tickers = get_sp500_tickers()
    fundamentals = pd.read_csv('fundamentals.csv')
    price_data = download_price_data(tickers)
    momentum = calculate_momentum(price_data, tickers)
    return fundamentals, momentum

with st.spinner('Loading data... (first run takes 15-30 min)'):
    fundamentals, momentum = load_data()

# Score and rank
scored = compute_factor_scores(fundamentals, momentum)
portfolio = build_portfolio(scored, n_stocks)
 
# Display portfolio table
st.subheader(f'Top {n_stocks} Ranked Stocks')
st.dataframe(portfolio, use_container_width=True)
 
# Sector breakdown chart
sector_counts = portfolio['sector'].value_counts()
fig_sector = px.pie(values=sector_counts.values, 
                    names=sector_counts.index,
                    title='Portfolio Sector Allocation')
st.plotly_chart(fig_sector, use_container_width=True)
 
# Factor score distributions
fig_factors = go.Figure()
for col in ['value_score', 'momentum_score', 'quality_score']:
    fig_factors.add_trace(go.Histogram(
        x=scored[col], name=col.replace('_', ' ').title(), 
        opacity=0.7))
fig_factors.update_layout(title='Factor Score Distributions',
                          barmode='overlay')
st.plotly_chart(fig_factors, use_container_width=True)
