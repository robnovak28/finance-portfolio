import pandas as pd
import requests
from io import StringIO

def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    sp500_table = tables[0]
    tickers = sp500_table['Symbol'].tolist()
    # Fix tickers with dots (BRK.B -> BRK-B for yfinance)
    tickers = [t.replace('.', '-') for t in tickers]
    return tickers

# Check to see if working properly
if __name__ == '__main__':
    tickers = get_sp500_tickers()
    print(f'Found {len(tickers)} tickers')
    print(tickers[:10])  # Print first 10

import yfinance as yf
from datetime import datetime, timedelta

def download_price_data(tickers):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=400)  # ~13 months

# Download all at once (much faster than one at a time)
    data = yf.download(
        tickers,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        group_by='ticker',
        auto_adjust=True,
        threads=True
    )
    return data

import time

def get_fundamentals(tickers):
    fundamentals = []
    
    for i, ticker in enumerate(tickers):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            fundamentals.append({
                'ticker': ticker,
                'pe_ratio': info.get('trailingPE'),
                'ev_ebitda': info.get('enterpriseToEbitda'),
                'roe': info.get('returnOnEquity'),
                'gross_margin': info.get('grossMargins'),
                'market_cap': info.get('marketCap'),
                'sector': info.get('sector'),
                'name': info.get('shortName'),
            })
            if i % 50 == 0:
                print(f'Processed {i}/{len(tickers)} tickers...')
            time.sleep(0.2)  # Be respectful to the API
        except Exception as e:
            print(f'Error on {ticker}: {e}')
            continue
    return pd.DataFrame(fundamentals)

def calculate_momentum(price_data, tickers):
    momentum_dict = {}
    for ticker in tickers:
        try:
            prices = price_data[ticker]['Close'].dropna()
            if len(prices) < 252:  # Need ~1 year of trading days
                continue
            # 12-1 month momentum: return from 12 months ago to 1 month ago
            ret_12m = prices.iloc[-22] / prices.iloc[-252] - 1
            momentum_dict[ticker] = ret_12m
        except:
            continue
    return momentum_dict
