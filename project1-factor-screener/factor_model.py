"""
factor_model.py
---------------
Multi-factor scoring and portfolio construction for the S&P 500 screener.

Four factors (all scored 0–1, where 1 = best):
  Value     : percentile rank of cheapness (P/E, EV/EBITDA) -- low multiples = better
  Momentum  : percentile rank of 12-1M price return -- high momentum = better
  Quality   : percentile rank of ROE and gross margin -- high = better
  Low Vol   : percentile rank of 12M realized vol -- low vol = better

Composite score: user-weighted average of the four factor scores.
Portfolio: top-N stocks by composite score (equal-weighted by default).
"""

import pandas as pd
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pct_rank_high(series: pd.Series) -> pd.Series:
    """Percentile rank where HIGH value = score closer to 1."""
    return series.rank(ascending=True, pct=True, na_option="keep").fillna(0.5)


def _pct_rank_low(series: pd.Series) -> pd.Series:
    """Percentile rank where LOW value = score closer to 1."""
    return series.rank(ascending=False, pct=True, na_option="keep").fillna(0.5)


# ── Factor Scoring ────────────────────────────────────────────────────────────

def compute_factor_scores(
    fundamentals_df: pd.DataFrame,
    price_metrics,           # DataFrame indexed by ticker OR legacy dict
    weights: dict | None = None,
) -> pd.DataFrame:
    """
    Score every stock in fundamentals_df on four factors.

    Parameters
    ----------
    fundamentals_df : DataFrame with columns ticker, pe_ratio, ev_ebitda,
                      roe, gross_margin, market_cap, sector, name
    price_metrics   : DataFrame indexed by ticker with columns momentum, vol_12m
                      (or legacy dict {ticker: momentum_value})
    weights         : dict with keys value, momentum, quality, lowvol (summing to ~1)
                      Defaults to {value: 0.25, momentum: 0.35, quality: 0.25, lowvol: 0.15}

    Returns
    -------
    DataFrame with factor scores + raw metrics + composite + final_rank
    """
    if weights is None:
        weights = {"value": 0.25, "momentum": 0.35, "quality": 0.25, "lowvol": 0.15}

    # Normalize weights
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {k: v / total_w for k, v in weights.items()}

    df = fundamentals_df.copy()

    # ── Merge price metrics ──────────────────────────────────────────────────
    if isinstance(price_metrics, dict):
        # Legacy interface: dict of momentum values
        df["momentum"] = df["ticker"].map(price_metrics)
        df["vol_12m"]  = np.nan
    else:
        # New interface: DataFrame indexed by ticker
        df = df.merge(
            price_metrics[["momentum", "vol_12m"]].reset_index(),
            on="ticker", how="left",
        )

    # Drop rows missing all key signals
    df = df.dropna(subset=["pe_ratio", "momentum"], how="all")

    # ── VALUE factor (low P/E and low EV/EBITDA = good) ─────────────────────
    valid_pe       = df["pe_ratio"].where(df["pe_ratio"] > 0)
    valid_ev       = df["ev_ebitda"].where(df["ev_ebitda"] > 0)
    pe_score       = _pct_rank_low(valid_pe)
    ev_score       = _pct_rank_low(valid_ev)
    df["value_score"] = (
        pe_score.fillna(0.5) * 0.5 + ev_score.fillna(0.5) * 0.5
    )

    # ── MOMENTUM factor (high 12-1M return = good) ───────────────────────────
    df["momentum_score"] = _pct_rank_high(df["momentum"])

    # ── QUALITY factor (high ROE and gross margin = good) ────────────────────
    roe_score    = _pct_rank_high(df["roe"])
    margin_score = _pct_rank_high(df["gross_margin"])
    df["quality_score"] = (
        roe_score.fillna(0.5) * 0.5 + margin_score.fillna(0.5) * 0.5
    )

    # ── LOW VOLATILITY factor (low 12M vol = good) ───────────────────────────
    if df["vol_12m"].notna().sum() > 10:
        df["lowvol_score"] = _pct_rank_low(df["vol_12m"])
    else:
        df["lowvol_score"] = 0.5  # neutral if unavailable

    # ── COMPOSITE score (weighted average) ───────────────────────────────────
    df["composite_score"] = (
        df["value_score"]    * weights["value"]    +
        df["momentum_score"] * weights["momentum"] +
        df["quality_score"]  * weights["quality"]  +
        df["lowvol_score"]   * weights["lowvol"]
    )

    # Rank: 1 = best composite score
    df["final_rank"] = df["composite_score"].rank(ascending=False).astype(int)
    df = df.sort_values("final_rank")

    return df


# ── Portfolio Construction ────────────────────────────────────────────────────

def build_portfolio(scored_df: pd.DataFrame, n_stocks: int = 20) -> pd.DataFrame:
    """
    Select top-N stocks by composite score, equal-weighted.

    Returns DataFrame with display-ready columns.
    """
    portfolio = scored_df.head(n_stocks).copy()
    portfolio["weight"] = 1.0 / n_stocks

    # Convenience columns for display
    portfolio["market_cap_b"]     = portfolio["market_cap"] / 1e9
    portfolio["roe_pct"]          = portfolio["roe"] * 100
    portfolio["gross_margin_pct"] = portfolio["gross_margin"] * 100
    portfolio["momentum_pct"]     = portfolio["momentum"] * 100
    portfolio["vol_12m_pct"]      = portfolio["vol_12m"] * 100 if "vol_12m" in portfolio.columns else np.nan

    cols = [
        "final_rank", "ticker", "name", "sector",
        "market_cap_b", "pe_ratio", "ev_ebitda",
        "roe_pct", "gross_margin_pct", "momentum_pct", "vol_12m_pct",
        "value_score", "momentum_score", "quality_score", "lowvol_score",
        "composite_score", "weight",
    ]
    return portfolio[[c for c in cols if c in portfolio.columns]]
