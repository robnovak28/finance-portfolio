import pandas as pd
import numpy as np
import yfinance as yf

def simple_backtest(portfolio_tickers, start_date, end_date):
    # Download portfolio stock prices
    port_data = yf.download(portfolio_tickers, start=start_date, 
                            end=end_date, auto_adjust=True)
    port_returns = port_data['Close'].pct_change().dropna()

    # Equal-weight daily portfolio return
    portfolio_daily = port_returns.mean(axis=1)
    portfolio_cumulative = (1 + portfolio_daily).cumprod() - 1
    
    # Benchmark: SPY
    spy = yf.download('SPY', start=start_date, end=end_date, auto_adjust=True)
    spy_returns = spy['Close'].pct_change().dropna()
    spy_cumulative = (1 + spy_returns).cumprod() - 1

    # Calculate key metrics
    ann_factor = 252
    port_ann_return = portfolio_daily.mean() * ann_factor
    port_ann_vol = portfolio_daily.std() * np.sqrt(ann_factor)
    sharpe = port_ann_return / port_ann_vol
    alpha = port_ann_return - (spy_returns.mean() * ann_factor)

    metrics = {
        'Portfolio Annual Return': f'{port_ann_return:.2%}',
        'SPY Annual Return': f'{spy_returns.mean() * ann_factor:.2%}',
        'Alpha': f'{alpha:.2%}',
        'Portfolio Volatility': f'{port_ann_vol:.2%}',
        'Sharpe Ratio': f'{sharpe:.2f}',
    }
    
    return metrics, portfolio_cumulative, spy_cumulative
