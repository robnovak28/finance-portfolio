"""
risk_metrics.py
---------------
Comprehensive risk and return analytics for a portfolio.

Metrics computed:
  - Annualized Return, Volatility, Sharpe Ratio
  - Sortino Ratio (downside deviation)
  - Calmar Ratio (return / max drawdown)
  - Max Drawdown + Duration
  - Value at Risk (VaR) 95% / 99% -- historical simulation
  - Conditional VaR / Expected Shortfall (CVaR) 95% / 99%
  - Alpha, Beta vs benchmark (OLS)
  - Information Ratio (alpha / tracking error)
  - Tracking Error
  - Win Rate, Skewness, Excess Kurtosis
  - Rolling Sharpe, Alpha, Beta (252-day window)
  - Monthly returns pivot table
"""

import numpy as np
import pandas as pd


ANN = 252


def compute_risk_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    rf_rate: float = 0.05,
) -> dict:
    """
    Full suite of risk/return metrics.

    Parameters
    ----------
    portfolio_returns : daily returns (not cumulative)
    benchmark_returns : daily benchmark returns (aligned index)
    rf_rate           : annual risk-free rate (decimal)

    Returns
    -------
    dict -- all scalar metrics plus 'drawdown_series' and 'cumulative_returns'
    """
    rf_daily = (1 + rf_rate) ** (1 / ANN) - 1

    # ── Annualized return & volatility ──────────────────────────────────────
    ann_return = (1 + portfolio_returns.mean()) ** ANN - 1
    ann_vol    = portfolio_returns.std() * np.sqrt(ANN)
    sharpe     = (ann_return - rf_rate) / ann_vol if ann_vol > 0 else np.nan

    # ── Sortino (downside std) ───────────────────────────────────────────────
    down = portfolio_returns[portfolio_returns < rf_daily]
    down_vol = down.std() * np.sqrt(ANN) if len(down) > 1 else np.nan
    sortino  = (ann_return - rf_rate) / down_vol if (down_vol and down_vol > 0) else np.nan

    # ── Max Drawdown & Calmar ────────────────────────────────────────────────
    cum       = (1 + portfolio_returns).cumprod()
    roll_max  = cum.cummax()
    drawdown  = (cum - roll_max) / roll_max
    max_dd    = drawdown.min()
    calmar    = ann_return / abs(max_dd) if max_dd < 0 else np.nan

    # Max drawdown duration (consecutive trading days underwater)
    underwater = (drawdown < 0)
    max_dur, cur = 0, 0
    for u in underwater:
        cur = cur + 1 if u else 0
        max_dur = max(max_dur, cur)

    # ── VaR / CVaR ───────────────────────────────────────────────────────────
    var_95  = float(np.percentile(portfolio_returns, 5))
    cvar_95 = float(portfolio_returns[portfolio_returns <= var_95].mean())
    var_99  = float(np.percentile(portfolio_returns, 1))
    cvar_99 = float(portfolio_returns[portfolio_returns <= var_99].mean())

    # ── Alpha, Beta, IR, TE ──────────────────────────────────────────────────
    aligned = pd.DataFrame({"p": portfolio_returns, "b": benchmark_returns}).dropna()
    if len(aligned) > 30:
        cov_mat  = np.cov(aligned["p"], aligned["b"])
        beta     = cov_mat[0, 1] / cov_mat[1, 1] if cov_mat[1, 1] > 0 else np.nan
        alpha_d  = aligned["p"].mean() - beta * aligned["b"].mean()
        alpha_ann = (1 + alpha_d) ** ANN - 1
        te        = (aligned["p"] - aligned["b"]).std() * np.sqrt(ANN)
        ir        = alpha_ann / te if te > 0 else np.nan
        bench_ann = (1 + aligned["b"].mean()) ** ANN - 1
        r2        = np.corrcoef(aligned["p"], aligned["b"])[0, 1] ** 2
    else:
        beta = alpha_ann = alpha_d = te = ir = bench_ann = r2 = np.nan

    win_rate = float((portfolio_returns > 0).mean())
    skew     = float(portfolio_returns.skew())
    kurt     = float(portfolio_returns.kurtosis())

    return {
        # Return
        "Annualized Return":        ann_return,
        "Benchmark Return":         bench_ann,
        "Alpha (Ann.)":             alpha_ann,
        "Beta":                     beta,
        "R-Squared":                r2,
        # Risk
        "Annualized Volatility":    ann_vol,
        "Sharpe Ratio":             sharpe,
        "Sortino Ratio":            sortino,
        "Calmar Ratio":             calmar,
        "Max Drawdown":             max_dd,
        "Max DD Duration (days)":   max_dur,
        # Credit metrics
        "Information Ratio":        ir,
        "Tracking Error":           te,
        # Tail risk
        "VaR 95% (daily)":          var_95,
        "CVaR 95% (daily)":         cvar_95,
        "VaR 99% (daily)":          var_99,
        "CVaR 99% (daily)":         cvar_99,
        # Distribution
        "Win Rate":                  win_rate,
        "Skewness":                  skew,
        "Excess Kurtosis":           kurt,
        # Series
        "drawdown_series":           drawdown,
        "cumulative_returns":        cum - 1,
    }


def rolling_risk_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = ANN,
) -> pd.DataFrame:
    """
    Compute rolling Sharpe, alpha, beta, return, volatility.

    Returns DataFrame indexed by date.
    """
    aligned = pd.DataFrame({"p": portfolio_returns, "b": benchmark_returns}).dropna()
    rows = []

    for i in range(window, len(aligned) + 1):
        w_p = aligned["p"].iloc[i - window : i]
        w_b = aligned["b"].iloc[i - window : i]

        ann_ret = (1 + w_p.mean()) ** ANN - 1
        ann_vol = w_p.std() * np.sqrt(ANN)
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan

        cov_mat = np.cov(w_p, w_b)
        beta    = cov_mat[0, 1] / cov_mat[1, 1] if cov_mat[1, 1] > 0 else np.nan
        alpha_d = w_p.mean() - (beta or 0) * w_b.mean()
        alpha   = (1 + alpha_d) ** ANN - 1

        rows.append({
            "date":           aligned.index[i - 1],
            "rolling_sharpe": sharpe,
            "rolling_alpha":  alpha,
            "rolling_beta":   beta,
            "rolling_return": ann_ret,
            "rolling_vol":    ann_vol,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")


def monthly_returns_table(daily_returns: pd.Series) -> pd.DataFrame:
    """
    Convert daily returns to a years x months pivot table.

    Returns DataFrame with month columns and an 'Annual' column.
    """
    monthly = (1 + daily_returns).resample("ME").prod() - 1
    tbl = monthly.to_frame("ret")
    tbl["year"]  = tbl.index.year
    tbl["month"] = tbl.index.month
    pivot = tbl.pivot(index="year", columns="month", values="ret")
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = month_names[: len(pivot.columns)]
    pivot["Annual"] = (1 + pivot.fillna(0)).prod(axis=1) - 1
    return pivot
