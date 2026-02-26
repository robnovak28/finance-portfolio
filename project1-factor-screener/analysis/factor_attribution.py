"""
factor_attribution.py
---------------------
Fama-French / Carhart 4-factor model regression for portfolio return attribution.

Uses the Carhart (1997) 4-factor model:
  r_p - RF = alpha + beta_mkt*(Mkt-RF) + beta_smb*SMB + beta_hml*HML + beta_mom*MOM + e

Factor definitions (Ken French Data Library):
  Mkt-RF  : Excess market return (CRSP value-weighted market return minus T-bill)
  SMB     : Small-Minus-Big (size factor)
  HML     : High-Minus-Low (value factor based on book-to-market)
  MOM     : Momentum (prior 12-1 month return)

Data source: pandas_datareader, Kenneth French Data Library.
Falls back to a CAPM-only regression if FF data is unavailable.
"""

import numpy as np
import pandas as pd
import warnings

try:
    import pandas_datareader.data as web
    HAS_PDR = True
except ImportError:
    HAS_PDR = False


def fetch_ff_factors(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch daily Fama-French 3-Factor + Momentum from Ken French Data Library.

    Returns
    -------
    DataFrame with columns: Mkt-RF, SMB, HML, MOM, RF  (all as decimals, daily)
    None if unavailable.
    """
    if not HAS_PDR:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ff3 = web.DataReader(
                "F-F_Research_Data_Factors_daily", "famafrench",
                start=start_date, end=end_date,
            )[0]
            mom = web.DataReader(
                "F-F_Momentum_Factor_daily", "famafrench",
                start=start_date, end=end_date,
            )[0]

        ff3.index = pd.to_datetime(ff3.index)
        mom.index = pd.to_datetime(mom.index)

        # Convert percent to decimal
        ff3 = ff3 / 100
        mom = mom / 100

        factors = ff3.join(mom[["Mom"]], how="inner")
        factors.columns = ["Mkt-RF", "SMB", "HML", "RF", "MOM"]
        return factors

    except Exception:
        return None


def _ols(X: np.ndarray, y: np.ndarray):
    """
    OLS regression via normal equations. Returns (betas, se, t_stats, p_values, r2, adj_r2).
    X must already include a constant column.
    """
    from scipy import stats

    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    betas   = XtX_inv @ X.T @ y
    y_hat   = X @ betas
    resid   = y - y_hat

    df_resid = n - k
    s2   = (resid @ resid) / df_resid
    se   = np.sqrt(np.diag(XtX_inv) * s2)
    t_st = betas / se
    pval = 2 * (1 - stats.t.cdf(np.abs(t_st), df=df_resid))

    ss_res = resid @ resid
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2     = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - 1) / df_resid

    return betas, se, t_st, pval, r2, adj_r2


def run_ff4_regression(
    portfolio_returns: pd.Series,
    factors_df: pd.DataFrame,
) -> dict:
    """
    Run Carhart 4-factor OLS regression.

    Parameters
    ----------
    portfolio_returns : daily returns Series
    factors_df        : DataFrame with columns Mkt-RF, SMB, HML, MOM, RF

    Returns
    -------
    dict with alpha, betas, t-stats, p-values, R², factor contributions
    """
    ANN = 252
    aligned = pd.DataFrame({"port": portfolio_returns}).join(factors_df, how="inner").dropna()

    if len(aligned) < 60:
        return None

    excess_ret   = aligned["port"] - aligned["RF"]
    factor_cols  = ["Mkt-RF", "SMB", "HML", "MOM"]
    X_factors    = aligned[factor_cols].values
    y            = excess_ret.values
    n            = len(y)

    X_aug = np.column_stack([np.ones(n), X_factors])
    betas, se, t_st, pval, r2, adj_r2 = _ols(X_aug, y)

    alpha_daily = betas[0]
    alpha_ann   = (1 + alpha_daily) ** ANN - 1
    alpha_se    = se[0]
    alpha_t     = t_st[0]
    alpha_p     = pval[0]

    factor_betas  = {f: betas[i + 1] for i, f in enumerate(factor_cols)}
    factor_tstats = {f: t_st[i + 1]  for i, f in enumerate(factor_cols)}
    factor_pvals  = {f: pval[i + 1]  for i, f in enumerate(factor_cols)}

    # Annualized factor contributions: beta_i * mean_factor * 252
    factor_contribs = {
        f: factor_betas[f] * aligned[f].mean() * ANN
        for f in factor_cols
    }

    # Total return decomposition
    total_ann_ret    = (1 + aligned["port"].mean()) ** ANN - 1
    rf_ann           = (1 + aligned["RF"].mean()) ** ANN - 1
    market_contrib   = factor_contribs["Mkt-RF"]
    size_contrib     = factor_contribs["SMB"]
    value_contrib    = factor_contribs["HML"]
    momentum_contrib = factor_contribs["MOM"]
    unexplained      = total_ann_ret - rf_ann - market_contrib - size_contrib - value_contrib - momentum_contrib - alpha_ann

    return {
        "alpha_daily":    alpha_daily,
        "alpha_ann":      alpha_ann,
        "alpha_se":       alpha_se,
        "alpha_tstat":    alpha_t,
        "alpha_pvalue":   alpha_p,
        "betas":          factor_betas,
        "t_stats":        factor_tstats,
        "p_values":       factor_pvals,
        "r_squared":      r2,
        "adj_r_squared":  adj_r2,
        "factor_contributions": factor_contribs,
        "decomposition": {
            "Total Return (Ann.)":     total_ann_ret,
            "Risk-Free Rate":          rf_ann,
            "Market (Mkt-RF)":         market_contrib,
            "Size (SMB)":              size_contrib,
            "Value (HML)":             value_contrib,
            "Momentum (MOM)":          momentum_contrib,
            "Jensen's Alpha":          alpha_ann,
        },
        "n_obs":         n,
        "factor_names":  factor_cols,
    }


def run_capm_fallback(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> dict:
    """
    Simple CAPM regression (alpha + beta * market) as fallback.
    Used when FF factor data is unavailable.
    """
    ANN = 252
    aligned = pd.DataFrame({"p": portfolio_returns, "b": benchmark_returns}).dropna()
    if len(aligned) < 30:
        return None

    y = aligned["p"].values
    X = np.column_stack([np.ones(len(y)), aligned["b"].values])
    betas, se, t_st, pval, r2, adj_r2 = _ols(X, y)

    alpha_ann = (1 + betas[0]) ** ANN - 1
    beta_mkt  = betas[1]

    return {
        "alpha_daily":   betas[0],
        "alpha_ann":     alpha_ann,
        "alpha_se":      se[0],
        "alpha_tstat":   t_st[0],
        "alpha_pvalue":  pval[0],
        "betas":         {"Mkt-RF": beta_mkt},
        "t_stats":       {"Mkt-RF": t_st[1]},
        "p_values":      {"Mkt-RF": pval[1]},
        "r_squared":     r2,
        "adj_r_squared": adj_r2,
        "factor_names":  ["Mkt-RF"],
        "n_obs":         len(aligned),
        "factor_contributions": {
            "Mkt-RF": beta_mkt * aligned["b"].mean() * ANN,
        },
    }
