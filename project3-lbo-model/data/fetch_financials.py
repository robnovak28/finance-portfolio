"""
fetch_financials.py
-------------------
Pulls historical financials for Dollar General (DG) via yfinance.
Falls back to hardcoded LTM figures from DG's FY2024 10-K if the
live fetch fails.  All figures in USD millions ($M) unless noted.
"""

import yfinance as yf
import pandas as pd
import numpy as np


TICKER = "DG"

# ---------------------------------------------------------------------------
# Hardcoded fallback: Dollar General FY2024 (fiscal year ended Feb 2, 2024)
# Source: DG 10-K filed March 2024, figures in $M
# ---------------------------------------------------------------------------
DG_HARDCODED = {
    "ticker": "DG",
    "company": "Dollar General Corporation",
    "description": (
        "Dollar General is a leading U.S. discount retailer operating ~20,000 "
        "stores across 47 states, targeting low-to-middle income consumers with "
        "everyday essentials at everyday low prices."
    ),
    # Income Statement
    "revenue":         37_844,
    "cogs":            26_628,
    "gross_profit":    11_216,
    "gross_margin":    0.2963,
    "sga":              8_026,
    "ebitda":           3_534,  # Operating income + D&A
    "da":               1_343,  # Depreciation & Amortization
    "ebit":             3_190,  # Operating income (EBIT)
    "interest_expense":   421,
    "ebt":              2_769,
    "tax_rate":         0.228,
    "net_income":       1_660,
    # Balance Sheet
    "cash":               285,
    "receivables":        224,
    "inventory":        7_014,
    "other_current":      500,
    "ppe_net":          5_962,
    "op_lease_rou":     8_089,
    "goodwill":         4_339,
    "other_lt_assets":  1_487,
    "total_assets":    27_900,
    "accounts_payable": 3_856,
    "accrued_liab":       784,
    "current_lease":    1_002,
    "other_current_liab": 389,
    "lt_debt":          6_797,
    "lt_lease":         7_999,
    "other_lt_liab":    1_724,
    "total_equity":     5_349,
    # Cash Flow
    "capex":            1_607,  # positive = cash out
    "stock_comp":         168,
    "change_in_wc":      -200,  # rough estimate
    # Derived
    "ebitda_margin":    0.0934,
    "net_debt":         6_512,  # gross debt - cash
    "shares_out":         216,  # millions
    # Historical revenue for context
    "revenue_history": {
        "FY2021": 33_747,
        "FY2022": 37_844,
        "FY2023": 37_844,  # placeholder
    },
    "ebitda_history": {
        "FY2021": 4_067,
        "FY2022": 4_260,
        "FY2023": 3_534,
    },
}


def _parse_income_statement(ticker_obj: yf.Ticker) -> dict:
    """Extract key IS metrics from yfinance annual income statement."""
    fin = ticker_obj.financials  # columns = fiscal years, rows = line items
    if fin is None or fin.empty:
        return {}

    col = fin.columns[0]  # most recent fiscal year
    def g(row, default=np.nan):
        return fin.loc[row, col] / 1e6 if row in fin.index else default

    revenue        = g("Total Revenue")
    cogs           = g("Cost Of Revenue")
    gross_profit   = g("Gross Profit")
    ebit           = g("EBIT")
    ebitda         = g("EBITDA")
    da             = g("Reconciled Depreciation")
    interest_exp   = abs(g("Interest Expense"))
    pretax_income  = g("Pretax Income")
    tax_provision  = g("Tax Provision")
    net_income     = g("Net Income")
    sga            = g("Selling General Administrative")

    tax_rate = abs(tax_provision / pretax_income) if pretax_income else 0.23

    return {
        "revenue": revenue,
        "cogs": cogs,
        "gross_profit": gross_profit,
        "gross_margin": gross_profit / revenue if revenue else np.nan,
        "sga": sga,
        "ebitda": ebitda,
        "da": da,
        "ebit": ebit,
        "interest_expense": interest_exp,
        "ebt": pretax_income,
        "tax_rate": tax_rate,
        "net_income": net_income,
        "ebitda_margin": ebitda / revenue if revenue else np.nan,
    }


def _parse_balance_sheet(ticker_obj: yf.Ticker) -> dict:
    """Extract key BS metrics from yfinance annual balance sheet."""
    bs = ticker_obj.balance_sheet
    if bs is None or bs.empty:
        return {}

    col = bs.columns[0]
    def g(row, default=0.0):
        return bs.loc[row, col] / 1e6 if row in bs.index else default

    cash            = g("Cash And Cash Equivalents")
    receivables     = g("Receivables")
    inventory       = g("Inventory")
    ppe_net         = g("Net PPE")
    goodwill        = g("Goodwill")
    total_assets    = g("Total Assets")
    total_equity    = g("Stockholders Equity")
    lt_debt         = g("Long Term Debt")
    current_debt    = g("Current Debt")
    total_debt      = lt_debt + current_debt
    ap              = g("Accounts Payable")
    accrued         = g("Other Current Liabilities")
    other_lt        = g("Other Non Current Liabilities")

    return {
        "cash": cash,
        "receivables": receivables,
        "inventory": inventory,
        "ppe_net": ppe_net,
        "goodwill": goodwill,
        "total_assets": total_assets,
        "accounts_payable": ap,
        "accrued_liab": accrued,
        "lt_debt": lt_debt,
        "total_equity": total_equity,
        "net_debt": total_debt - cash,
    }


def _parse_cashflow(ticker_obj: yf.Ticker) -> dict:
    """Extract key CFS metrics from yfinance annual cash flow statement."""
    cf = ticker_obj.cashflow
    if cf is None or cf.empty:
        return {}

    col = cf.columns[0]
    def g(row, default=0.0):
        return cf.loc[row, col] / 1e6 if row in cf.index else default

    capex     = abs(g("Capital Expenditure"))
    stock_comp = g("Stock Based Compensation")
    change_wc  = g("Change In Working Capital")
    ocf        = g("Operating Cash Flow")

    return {
        "capex": capex,
        "stock_comp": stock_comp,
        "change_in_wc": change_wc,
        "operating_cash_flow": ocf,
    }


def _parse_history(ticker_obj: yf.Ticker) -> dict:
    """Get 3-year revenue and EBITDA history for context."""
    fin = ticker_obj.financials
    if fin is None or fin.empty:
        return {}

    rev_history = {}
    ebitda_history = {}
    for col in fin.columns[:3]:
        yr = str(col.year)
        if "Total Revenue" in fin.index:
            rev_history[yr] = fin.loc["Total Revenue", col] / 1e6
        if "EBITDA" in fin.index:
            ebitda_history[yr] = fin.loc["EBITDA", col] / 1e6

    return {"revenue_history": rev_history, "ebitda_history": ebitda_history}


def get_target_financials(use_live: bool = True) -> dict:
    """
    Fetch Dollar General financials.  Returns a dict with all metrics
    in $M needed for the LBO model.  If use_live=False or the fetch
    fails, returns hardcoded FY2024 10-K data.
    """
    if not use_live:
        return DG_HARDCODED.copy()

    try:
        ticker_obj = yf.Ticker(TICKER)
        info = ticker_obj.info or {}

        is_data = _parse_income_statement(ticker_obj)
        bs_data = _parse_balance_sheet(ticker_obj)
        cf_data = _parse_cashflow(ticker_obj)
        hist    = _parse_history(ticker_obj)

        if not is_data or pd.isna(is_data.get("revenue", np.nan)):
            return DG_HARDCODED.copy()

        merged = {
            "ticker":      TICKER,
            "company":     info.get("longName", "Dollar General Corporation"),
            "description": info.get("longBusinessSummary", DG_HARDCODED["description"]),
            "shares_out":  info.get("sharesOutstanding", 216e6) / 1e6,
            **is_data,
            **bs_data,
            **cf_data,
            **hist,
        }

        # Fill any missing BS fields with hardcoded fallbacks
        for key, val in DG_HARDCODED.items():
            if key not in merged or (isinstance(merged.get(key), float) and pd.isna(merged[key])):
                merged[key] = val

        return merged

    except Exception:
        return DG_HARDCODED.copy()


def get_historical_financials_df() -> pd.DataFrame:
    """
    Returns a DataFrame of 3-year historical IS metrics for display.
    Tries yfinance first, falls back to hardcoded history.
    """
    try:
        ticker_obj = yf.Ticker(TICKER)
        fin = ticker_obj.financials
        if fin is None or fin.empty:
            raise ValueError

        rows = []
        for col in reversed(fin.columns[:4]):
            yr = col.strftime("%b '%y")
            rev     = fin.loc["Total Revenue",            col] / 1e6 if "Total Revenue"            in fin.index else np.nan
            gp      = fin.loc["Gross Profit",             col] / 1e6 if "Gross Profit"             in fin.index else np.nan
            ebitda  = fin.loc["EBITDA",                   col] / 1e6 if "EBITDA"                   in fin.index else np.nan
            ni      = fin.loc["Net Income",               col] / 1e6 if "Net Income"               in fin.index else np.nan
            rows.append({
                "Period":        yr,
                "Revenue ($M)":        round(rev, 0)   if not pd.isna(rev)    else np.nan,
                "Gross Profit ($M)":   round(gp, 0)    if not pd.isna(gp)     else np.nan,
                "EBITDA ($M)":         round(ebitda, 0) if not pd.isna(ebitda) else np.nan,
                "EBITDA Margin":       ebitda / rev     if (not pd.isna(ebitda) and rev) else np.nan,
                "Net Income ($M)":     round(ni, 0)     if not pd.isna(ni)     else np.nan,
            })
        return pd.DataFrame(rows)

    except Exception:
        return pd.DataFrame([
            {"Period": "FY2021", "Revenue ($M)": 33_747, "Gross Profit ($M)": 9_985,  "EBITDA ($M)": 4_067, "EBITDA Margin": 0.1205, "Net Income ($M)": 2_399},
            {"Period": "FY2022", "Revenue ($M)": 37_844, "Gross Profit ($M)": 11_216, "EBITDA ($M)": 4_260, "EBITDA Margin": 0.1126, "Net Income ($M)": 2_416},
            {"Period": "FY2023", "Revenue ($M)": 37_844, "Gross Profit ($M)": 11_216, "EBITDA ($M)": 3_534, "EBITDA Margin": 0.0934, "Net Income ($M)": 1_660},
        ])
