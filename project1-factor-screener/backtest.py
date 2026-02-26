"""
backtest.py
-----------
Portfolio backtesting engine for the factor screener.

run_backtest(): Downloads historical prices for the portfolio tickers,
computes daily equal-weight portfolio returns, and returns a comprehensive
result dict including rolling risk metrics and monthly return table.

simple_backtest(): Legacy interface (kept for compatibility).
"""

import numpy as np
import pandas as pd
import yfinance as yf

from analysis.risk_metrics import (
    compute_risk_metrics,
    rolling_risk_metrics,
    monthly_returns_table,
)


def run_backtest(
    portfolio_tickers: list,
    start_date: str,
    end_date: str,
    benchmark: str = "SPY",
    rf_rate: float = 0.05,
) -> dict:
    """
    Run a full backtest for an equal-weight portfolio.

    Parameters
    ----------
    portfolio_tickers : list of ticker strings
    start_date        : "YYYY-MM-DD"
    end_date          : "YYYY-MM-DD"
    benchmark         : benchmark ticker (default: SPY)
    rf_rate           : annual risk-free rate

    Returns
    -------
    dict with keys:
        port_daily    : daily portfolio returns (Series)
        bench_daily   : daily benchmark returns (Series)
        port_cum      : cumulative portfolio returns (Series)
        bench_cum     : cumulative benchmark returns (Series)
        metrics       : dict of scalar risk/return metrics
        rolling_df    : DataFrame of rolling Sharpe, alpha, beta (252-day)
        drawdown      : drawdown Series
        monthly_table : DataFrame (years x months)
        tickers_used  : list of tickers actually used
        start, end    : actual date range strings
        n_trading_days: int
    """
    all_tickers = list(portfolio_tickers) + [benchmark]

    raw = yf.download(
        all_tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        return _empty_result()

    # Extract Close prices into a flat DataFrame
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": all_tickers[0]})

    prices = prices.dropna(how="all")

    if benchmark not in prices.columns:
        return _empty_result()

    # Benchmark daily returns
    bench_daily = prices[benchmark].pct_change().dropna()

    # Portfolio: equal-weight, drop tickers with too many gaps
    port_cols   = [t for t in portfolio_tickers if t in prices.columns]
    port_prices = prices[port_cols].dropna(how="all")
    port_rets   = port_prices.pct_change()
    port_rets   = port_rets.dropna(axis=1, thresh=int(len(port_rets) * 0.8))
    tickers_used = list(port_rets.columns)
    port_daily   = port_rets.mean(axis=1).dropna()

    # Align to common dates
    idx         = port_daily.index.intersection(bench_daily.index)
    port_daily  = port_daily.loc[idx]
    bench_daily = bench_daily.loc[idx]

    if len(port_daily) < 30:
        return _empty_result()

    # Cumulative returns
    port_cum  = (1 + port_daily).cumprod() - 1
    bench_cum = (1 + bench_daily).cumprod() - 1

    # Full risk metrics
    risk     = compute_risk_metrics(port_daily, bench_daily, rf_rate=rf_rate)
    drawdown = risk.pop("drawdown_series")
    risk.pop("cumulative_returns")

    # Rolling 252-day metrics
    rolling_df = pd.DataFrame()
    if len(port_daily) >= 252:
        rolling_df = rolling_risk_metrics(port_daily, bench_daily, window=252)

    # Monthly return pivot
    monthly = monthly_returns_table(port_daily)

    return {
        "port_daily":     port_daily,
        "bench_daily":    bench_daily,
        "port_cum":       port_cum,
        "bench_cum":      bench_cum,
        "metrics":        risk,
        "rolling_df":     rolling_df,
        "drawdown":       drawdown,
        "monthly_table":  monthly,
        "tickers_used":   tickers_used,
        "start":          str(idx[0].date()),
        "end":            str(idx[-1].date()),
        "n_trading_days": len(port_daily),
    }


def _empty_result() -> dict:
    return {
        "port_daily":     pd.Series(dtype=float),
        "bench_daily":    pd.Series(dtype=float),
        "port_cum":       pd.Series(dtype=float),
        "bench_cum":      pd.Series(dtype=float),
        "metrics":        {},
        "rolling_df":     pd.DataFrame(),
        "drawdown":       pd.Series(dtype=float),
        "monthly_table":  pd.DataFrame(),
        "tickers_used":   [],
        "start":          None,
        "end":            None,
        "n_trading_days": 0,
    }


# ── Legacy interface ──────────────────────────────────────────────────────────

def simple_backtest(portfolio_tickers, start_date, end_date):
    """Legacy function kept for backward compatibility."""
    result = run_backtest(portfolio_tickers, start_date, end_date)
    m = result["metrics"]
    metrics = {
        "Portfolio Annual Return": f"{m.get('Annualized Return', 0):.2%}",
        "SPY Annual Return":       f"{m.get('Benchmark Return', 0):.2%}",
        "Alpha":                   f"{m.get('Alpha (Ann.)', 0):.2%}",
        "Portfolio Volatility":    f"{m.get('Annualized Volatility', 0):.2%}",
        "Sharpe Ratio":            f"{m.get('Sharpe Ratio', 0):.2f}",
    }
    return metrics, result["port_cum"], result["bench_cum"]
