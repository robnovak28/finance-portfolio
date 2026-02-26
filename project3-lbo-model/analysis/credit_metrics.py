"""
credit_metrics.py
-----------------
Extended credit analysis on the projected LBO capital structure.

Computed metrics (by year):
  - Gross / Net Leverage (Total Debt / EBITDA  and  Net Debt / EBITDA)
  - Interest Coverage Ratio (EBITDA / Cash Interest)
  - Fixed Charge Coverage Ratio (EBITDA / (Cash Interest + Req Amort))
  - Debt Service Coverage Ratio (OCF / Cash Interest)
  - Total Debt by Tranche (stacked waterfall)
  - Cumulative Debt Paydown vs. Entry
  - Implied Credit Rating Proxy (simplistic mapping)

Also generates a "debt waterfall" DataFrame suitable for a stacked bar chart.
"""

import numpy as np
import pandas as pd


# Simplified leverage → implied credit rating mapping (investment-grade proxy)
LEVERAGE_RATING_MAP = [
    (2.0,  "BBB+/Baa1"),
    (3.0,  "BBB/Baa2"),
    (3.5,  "BBB-/Baa3"),
    (4.5,  "BB+/Ba1"),
    (5.5,  "BB/Ba2"),
    (6.5,  "BB-/Ba3"),
    (7.5,  "B+/B1"),
    (9.0,  "B/B2"),
    (99.0, "B-/B3 or below"),
]


def _implied_rating(leverage: float) -> str:
    for threshold, rating in LEVERAGE_RATING_MAP:
        if leverage <= threshold:
            return rating
    return "CCC"


def build_credit_dashboard(
    model_result: dict,
    assumptions,   # DealAssumptions
) -> dict:
    """
    Build extended credit metrics from a model result.

    Parameters
    ----------
    model_result : return value of lbo_engine.run_model()
    assumptions  : DealAssumptions used for that run

    Returns
    -------
    {
      "credit_df"   : extended year-by-year metrics DataFrame,
      "waterfall_df": stacked debt waterfall DataFrame,
      "covenant_df" : covenant headroom DataFrame,
    }
    """
    is_df       = model_result["is_df"]
    cfs_df      = model_result["cfs_df"]
    debt_sched  = model_result["debt_schedule"]
    bs_df       = model_result["bs_df"]
    n           = assumptions.hold_years

    rows = []
    for yr in range(1, n + 1):
        col      = f"Year {yr}"
        sched_yr = debt_sched["schedule"][yr]

        ebitda      = is_df.loc["EBITDA",            col]
        ebit        = is_df.loc["EBIT",              col]
        rev         = is_df.loc["Revenue",           col]
        ni          = is_df.loc["Net Income",        col]
        int_exp     = is_df.loc["Interest Expense",  col]
        cash_int    = sched_yr["cash_interest"]
        req_amort   = sched_yr["required_amort"]
        end_debt    = sched_yr["ending_debt"]
        cash        = bs_df.loc["Cash & Equivalents", col]
        net_debt    = end_debt - cash
        ocf         = cfs_df.loc["Operating CF",     col]
        fcf         = cfs_df.loc["Free Cash Flow",   col]
        capex       = abs(cfs_df.loc["(-) CapEx",    col])

        gross_lev   = end_debt / ebitda if ebitda > 0 else np.nan
        net_lev     = net_debt / ebitda if ebitda > 0 else np.nan
        int_cov     = ebitda / cash_int if cash_int > 0 else np.nan
        fccr        = ebitda / (cash_int + req_amort) if (cash_int + req_amort) > 0 else np.nan
        dscr        = ocf    / cash_int if cash_int > 0 else np.nan

        rows.append({
            "Year":                    yr,
            "Revenue ($M)":            round(rev, 1),
            "EBITDA ($M)":             round(ebitda, 1),
            "EBITDA Margin":           ebitda / rev if rev > 0 else 0,
            "Total Debt ($M)":         round(end_debt, 1),
            "Net Debt ($M)":           round(net_debt, 1),
            "Gross Leverage (x)":      round(gross_lev, 2),
            "Net Leverage (x)":        round(net_lev, 2),
            "Cash Interest ($M)":      round(cash_int, 1),
            "Interest Coverage (x)":   round(int_cov, 2),
            "Fixed Charge Coverage (x)": round(fccr, 2),
            "DSCR (x)":                round(dscr, 2),
            "Operating CF ($M)":       round(ocf, 1),
            "Free Cash Flow ($M)":     round(fcf, 1),
            "FCF / EBITDA":            fcf / ebitda if ebitda > 0 else 0,
            "CapEx / Revenue":         capex / rev if rev > 0 else 0,
            "Implied Rating":          _implied_rating(gross_lev),
        })

    credit_df = pd.DataFrame(rows)

    # ---- Debt Waterfall (stacked bar data) ----
    waterfall_rows = []
    for yr in range(0, n + 1):  # Year 0 = opening
        row = {"Year": f"Year {yr}" if yr > 0 else "Entry"}
        if yr == 0:
            for t in assumptions.debt_tranches:
                row[t.name] = t.principal
        else:
            tranche_bals = debt_sched["schedule"][yr]["balances_by_tranche"]
            for name, bal in tranche_bals.items():
                row[name] = round(bal, 1)
        waterfall_rows.append(row)

    waterfall_df = pd.DataFrame(waterfall_rows).set_index("Year")

    # ---- Covenant Headroom ----
    # Typical LBO covenants: Max Gross Leverage, Min Interest Coverage
    # Generated dynamically for any hold period (slider range 3–8 years)
    # Leverage steps down 0.5x/yr from 6.5x, floored at 3.0x
    # Coverage steps up 0.25x every two years from 2.0x, capped at 3.0x
    max_lev_covenant = [max(3.0, 6.5 - 0.5 * (yr - 1)) for yr in range(1, n + 1)]
    min_cov_covenant = [min(3.0, 2.0 + 0.25 * ((yr - 1) // 2)) for yr in range(1, n + 1)]

    cov_rows = []
    for yr in range(1, n + 1):
        row  = credit_df[credit_df["Year"] == yr].iloc[0]
        act_lev = row["Gross Leverage (x)"]
        act_cov = row["Interest Coverage (x)"]
        lev_covenant = max_lev_covenant[yr - 1]
        cov_covenant = min_cov_covenant[yr - 1]
        cov_rows.append({
            "Year":                     yr,
            "Gross Leverage":           round(act_lev, 2),
            "Max Leverage Covenant":    lev_covenant,
            "Leverage Headroom (x)":    round(lev_covenant - act_lev, 2),
            "In Compliance (Leverage)": "YES" if act_lev <= lev_covenant else "NO",
            "Interest Coverage":        round(act_cov, 2),
            "Min Coverage Covenant":    cov_covenant,
            "Coverage Headroom (x)":    round(act_cov - cov_covenant, 2),
            "In Compliance (Coverage)": "YES" if act_cov >= cov_covenant else "NO",
        })
    covenant_df = pd.DataFrame(cov_rows)

    return {
        "credit_df":   credit_df,
        "waterfall_df":waterfall_df,
        "covenant_df": covenant_df,
    }
