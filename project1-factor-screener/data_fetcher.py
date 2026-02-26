"""
data_fetcher.py
---------------
Market data utilities for the S&P 500 factor screener.

Functions
---------
get_sp500_tickers()         -- scrape S&P 500 tickers from Wikipedia
download_price_data()       -- bulk yfinance download (~13 months, 500 tickers)
calculate_momentum()        -- 12-1 month momentum dict  (legacy)
calculate_price_metrics()   -- DataFrame of momentum + 12M vol per ticker (new)
download_portfolio_prices() -- fast download for a small list of tickers
get_fundamentals()          -- yfinance fundamental scrape (one-time, slow)
"""

import os
import pandas as pd
import numpy as np
import requests
import time
import yfinance as yf
from io import StringIO
from datetime import datetime, timedelta


# ── S&P 500 ticker list ───────────────────────────────────────────────────────

def get_sp500_tickers(csv_fallback: str = None) -> list:
    """
    Scrape current S&P 500 constituents from Wikipedia.
    Falls back to tickers from fundamentals.csv if the request fails.
    """
    try:
        url     = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp    = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        tables  = pd.read_html(StringIO(resp.text))
        tickers = tables[0]["Symbol"].tolist()
        return [t.replace(".", "-") for t in tickers]
    except Exception:
        # Fall back to tickers already present in the local fundamentals CSV
        fallback = csv_fallback or os.path.join(os.path.dirname(__file__), "fundamentals.csv")
        if os.path.exists(fallback):
            df = pd.read_csv(fallback)
            if "ticker" in df.columns:
                return df["ticker"].dropna().tolist()
        raise


# ── Bulk price download (for momentum/vol scoring of full universe) ───────────

def download_price_data(tickers: list, max_retries: int = 3) -> object:
    """
    Download ~13 months of adjusted close prices for all S&P 500 tickers.
    Retries up to max_retries times with exponential backoff on connection errors.
    """
    end_date   = datetime.today()
    start_date = end_date - timedelta(days=400)
    last_err   = None
    for attempt in range(max_retries):
        try:
            data = yf.download(
                tickers,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                group_by="ticker",
                auto_adjust=True,
                threads=True,
                progress=False,
            )
            return data
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)   # 1 s, 2 s between retries
    raise last_err


# ── Price metrics ─────────────────────────────────────────────────────────────

def calculate_momentum(price_data, tickers: list) -> dict:
    """
    Legacy interface: returns dict {ticker: 12-1 month momentum return}.
    """
    mom_dict = {}
    for ticker in tickers:
        try:
            prices = price_data[ticker]["Close"].dropna()
            if len(prices) < 252:
                continue
            mom_dict[ticker] = float(prices.iloc[-22] / prices.iloc[-252] - 1)
        except Exception:
            continue
    return mom_dict


def calculate_price_metrics(price_data, tickers: list) -> pd.DataFrame:
    """
    Compute per-ticker momentum (12-1 month) and 12M realized volatility
    from the bulk price data download.

    Returns
    -------
    DataFrame indexed by ticker with columns:
        momentum  (12-1M price return, decimal)
        vol_12m   (trailing 12M annualized daily vol, decimal)
    """
    rows = []
    for ticker in tickers:
        try:
            prices = price_data[ticker]["Close"].dropna()
            if len(prices) < 252:
                continue
            momentum = float(prices.iloc[-22] / prices.iloc[-252] - 1)
            daily_r  = prices.pct_change().dropna()
            vol_12m  = float(daily_r.iloc[-252:].std() * np.sqrt(252))
            rows.append({"ticker": ticker, "momentum": momentum, "vol_12m": vol_12m})
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["ticker", "momentum", "vol_12m"]).set_index("ticker")
    return pd.DataFrame(rows).set_index("ticker")


# ── Portfolio-specific price download (backtest, optimizer, FF attribution) ───

def download_portfolio_prices(tickers: list, years: int = 3,
                               max_retries: int = 3) -> pd.DataFrame:
    """
    Download adjusted close prices for a small list of portfolio tickers.
    Retries up to max_retries times with exponential backoff on connection errors.
    """
    if not tickers:
        return pd.DataFrame()

    end_date   = datetime.today()
    start_date = end_date - timedelta(days=365 * years + 10)
    last_err   = None

    for attempt in range(max_retries):
        try:
            raw = yf.download(
                tickers,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                return pd.DataFrame()
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Close"]
            else:
                prices = raw[["Close"]].rename(columns={"Close": tickers[0]})
            return prices.dropna(how="all")
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    raise last_err


# ── One-time fundamentals scrape ──────────────────────────────────────────────

def get_fundamentals(tickers: list) -> pd.DataFrame:
    """
    Fetch fundamentals for each ticker via yfinance.
    Rate-limited to ~0.2s/ticker; results should be saved to fundamentals.csv.
    """
    rows = []
    for i, ticker in enumerate(tickers):
        try:
            info = yf.Ticker(ticker).info
            rows.append({
                "ticker":       ticker,
                "pe_ratio":     info.get("trailingPE"),
                "ev_ebitda":    info.get("enterpriseToEbitda"),
                "roe":          info.get("returnOnEquity"),
                "gross_margin": info.get("grossMargins"),
                "market_cap":   info.get("marketCap"),
                "sector":       info.get("sector"),
                "name":         info.get("shortName"),
            })
            if i % 50 == 0:
                print(f"Fetched {i}/{len(tickers)} tickers...")
            time.sleep(0.2)
        except Exception as e:
            print(f"Error on {ticker}: {e}")
    return pd.DataFrame(rows)


# ── Script entry point ────────────────────────────────────────────────────────

def refresh_fundamentals_csv(output_path: str = "fundamentals.csv") -> pd.DataFrame:
    """
    Re-fetch fundamentals for all S&P 500 tickers and save to CSV.
    Run periodically (e.g., weekly) to keep data fresh.
    Takes ~15-30 minutes due to yfinance rate limiting.
    """
    print("Fetching S&P 500 tickers...")
    tickers = get_sp500_tickers()
    print(f"Fetching fundamentals for {len(tickers)} tickers (this takes ~15-30 min)...")
    df = get_fundamentals(tickers)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Refresh S&P 500 fundamentals data")
    parser.add_argument("--update", action="store_true", help="Re-fetch and save fundamentals.csv")
    args = parser.parse_args()

    if args.update:
        refresh_fundamentals_csv("fundamentals.csv")
    else:
        tickers = get_sp500_tickers()
        print(f"Found {len(tickers)} tickers: {tickers[:10]}")
        print("Run with --update to re-fetch fundamentals.csv")
