"""
formatting.py
-------------
Display helpers for project1-factor-screener.
"""

import numpy as np
import pandas as pd


def fmt_pct(val, decimals=1):
    if pd.isna(val) or val is None:
        return "—"
    return f"{val:.{decimals}%}"


def fmt_x(val, decimals=1):
    if pd.isna(val) or val is None:
        return "—"
    return f"{val:.{decimals}f}x"


def fmt_score(val, decimals=2):
    if pd.isna(val) or val is None:
        return "—"
    return f"{val:.{decimals}f}"


def fmt_billions(val, decimals=1):
    if pd.isna(val) or val is None:
        return "—"
    return f"${val / 1e9:.{decimals}f}B"


def fmt_millions(val, decimals=0):
    if pd.isna(val) or val is None:
        return "—"
    return f"${val / 1e6:,.{decimals}f}M"


def score_color(val):
    """CSS background for a 0–1 factor score (1 = best)."""
    if not isinstance(val, (int, float)) or np.isnan(val):
        return "background-color: #333; color: #aaa"
    if val >= 0.80:
        return "background-color: #16a085; color: white"
    if val >= 0.60:
        return "background-color: #2ecc71; color: black"
    if val >= 0.40:
        return "background-color: #f1c40f; color: black"
    if val >= 0.20:
        return "background-color: #e74c3c; color: white"
    return "background-color: #c0392b; color: white"


# Column config for Streamlit dataframe display
def build_portfolio_column_config():
    """Return st.column_config dict for the holdings table."""
    import streamlit as st
    return {
        "final_rank":      st.column_config.NumberColumn("Rank", format="%d", width="small"),
        "ticker":          st.column_config.TextColumn("Ticker", width="small"),
        "name":            st.column_config.TextColumn("Company", width="medium"),
        "sector":          st.column_config.TextColumn("Sector", width="medium"),
        "market_cap_b":    st.column_config.NumberColumn("Mkt Cap ($B)", format="$%.1f", width="small"),
        "pe_ratio":        st.column_config.NumberColumn("P/E", format="%.1fx", width="small"),
        "ev_ebitda":       st.column_config.NumberColumn("EV/EBITDA", format="%.1fx", width="small"),
        "roe_pct":         st.column_config.NumberColumn("ROE (%)", format="%.1f%%", width="small"),
        "gross_margin_pct":st.column_config.NumberColumn("Gross Margin (%)", format="%.1f%%", width="small"),
        "momentum_pct":    st.column_config.NumberColumn("Mom 12-1M (%)", format="%.1f%%", width="small"),
        "vol_12m_pct":     st.column_config.NumberColumn("Vol 12M (%)", format="%.1f%%", width="small"),
        "value_score":     st.column_config.ProgressColumn("Value", min_value=0, max_value=1, format="%.2f"),
        "momentum_score":  st.column_config.ProgressColumn("Momentum", min_value=0, max_value=1, format="%.2f"),
        "quality_score":   st.column_config.ProgressColumn("Quality", min_value=0, max_value=1, format="%.2f"),
        "lowvol_score":    st.column_config.ProgressColumn("Low Vol", min_value=0, max_value=1, format="%.2f"),
        "composite_score": st.column_config.ProgressColumn("Composite", min_value=0, max_value=1, format="%.2f"),
    }


def format_metrics_table(metrics_dict):
    """Format a dict of metrics into a two-column display DataFrame."""
    rows = []
    for k, v in metrics_dict.items():
        if isinstance(v, float):
            if "Return" in k or "Alpha" in k or "Drawdown" in k or "VaR" in k or "CVaR" in k or "%" in k or "Rate" in k or "Volatility" in k or "Error" in k:
                display = fmt_pct(v)
            elif "Ratio" in k or "Beta" in k or "R2" in k or "R-sq" in k:
                display = f"{v:.3f}"
            elif "Duration" in k:
                display = f"{int(v)} days"
            else:
                display = f"{v:.4f}"
        else:
            display = str(v)
        rows.append({"Metric": k, "Value": display})
    return pd.DataFrame(rows)
