"""
formatting.py
-------------
Number formatting helpers for Streamlit tables and charts.
"""

import numpy as np
import pandas as pd


def fmt_millions(val, decimals: int = 1) -> str:
    if pd.isna(val) or val is None:
        return "—"
    return f"${val:,.{decimals}f}M"


def fmt_pct(val, decimals: int = 1) -> str:
    if pd.isna(val) or val is None:
        return "—"
    return f"{val:.{decimals}%}"


def fmt_multiple(val, decimals: int = 2) -> str:
    if pd.isna(val) or val is None:
        return "—"
    return f"{val:.{decimals}f}x"


def fmt_irr(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.1%}"


def fmt_moic(val) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.2f}x"


# Row-level IS formatter: dollar rows vs percent rows
PCT_ROWS = {"Gross Margin", "EBITDA Margin"}

def format_is_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format IS DataFrame for display: $M for dollar rows, % for margin rows."""
    out = df.copy().astype(object)
    for row_label in df.index:
        for col in df.columns:
            v = df.loc[row_label, col]
            if row_label in PCT_ROWS:
                out.loc[row_label, col] = fmt_pct(v)
            else:
                out.loc[row_label, col] = fmt_millions(v)
    return out


def format_bs_df(df: pd.DataFrame, skip_rows: set = None) -> pd.DataFrame:
    """Format BS DataFrame for display."""
    skip_rows = skip_rows or {"BS Check (Assets − L+E)"}
    out = df.copy().astype(object)
    for row_label in df.index:
        for col in df.columns:
            v = df.loc[row_label, col]
            if row_label in skip_rows:
                out.loc[row_label, col] = f"${v:,.1f}M" if not pd.isna(v) else "—"
            else:
                out.loc[row_label, col] = fmt_millions(v)
    return out


def format_cfs_df(df: pd.DataFrame) -> pd.DataFrame:
    """Format CFS DataFrame for display."""
    return format_bs_df(df, skip_rows=set())


def style_sensitivity_table(df: pd.DataFrame, is_irr: bool = True):
    """
    Apply gradient coloring to a sensitivity DataFrame.
    Works on raw numeric DataFrames.
    Returns a pandas Styler object.
    """
    def irr_color(val):
        if not isinstance(val, (int, float)) or np.isnan(val):
            return "background-color: #444; color: #aaa"
        if val < 0.10:
            return "background-color: #c0392b; color: white"
        if val < 0.15:
            return "background-color: #e74c3c; color: white"
        if val < 0.18:
            return "background-color: #e67e22; color: white"
        if val < 0.22:
            return "background-color: #f1c40f; color: black"
        if val < 0.27:
            return "background-color: #2ecc71; color: black"
        return "background-color: #16a085; color: white"

    def moic_color(val):
        if not isinstance(val, (int, float)) or np.isnan(val):
            return "background-color: #444; color: #aaa"
        if val < 1.5:
            return "background-color: #c0392b; color: white"
        if val < 2.0:
            return "background-color: #e74c3c; color: white"
        if val < 2.5:
            return "background-color: #f1c40f; color: black"
        if val < 3.0:
            return "background-color: #2ecc71; color: black"
        return "background-color: #16a085; color: white"

    color_fn = irr_color if is_irr else moic_color
    fmt_fn   = (lambda v: fmt_pct(v, 1)) if is_irr else (lambda v: fmt_multiple(v, 2))

    # applymap was renamed to map in pandas 2.1+; support both
    _map = getattr(pd.DataFrame, "map", None) or getattr(pd.DataFrame, "applymap")

    display_df = _map(df, lambda v: fmt_fn(v) if isinstance(v, (int, float)) else v)
    return display_df.style.apply(
        lambda col: [color_fn(df.loc[idx, col.name]) for idx in df.index],
        axis=0
    )
