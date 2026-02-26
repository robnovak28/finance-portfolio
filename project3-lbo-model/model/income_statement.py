"""
income_statement.py
-------------------
Projects the Income Statement for each year of the hold period.

Revenue drivers → Gross Profit → EBITDA → EBIT → EBT → Net Income

Notes:
  - Sponsor management fee is modeled as an above-EBITDA operating cost
    so it reduces EBITDA visible to lenders.  The model tracks both
    "reported EBITDA" (after mgmt fee) and "adjusted EBITDA" (before).
  - Interest expense is passed in from the debt schedule (circular dependency
    broken by iterating: project IS → project debt schedule → re-project IS).
  - All values in $M.
"""

import pandas as pd
from model.assumptions import DealAssumptions


def build_income_statement(
    assumptions: DealAssumptions,
    interest_expense_by_year: list[float],  # from debt schedule
) -> pd.DataFrame:
    """
    Returns a wide DataFrame with one column per projected year.
    Index = line item labels.
    """
    n = assumptions.hold_years
    cols = [f"Year {i}" for i in range(1, n + 1)]
    data = {col: {} for col in cols}

    revenue = assumptions.entry_revenue
    for i, col in enumerate(cols):
        d = data[col]

        # --- Revenue ---
        revenue = revenue * (1 + assumptions.revenue_growth[i])
        d["Revenue"] = revenue

        # --- Gross Profit ---
        # Gross margin assumed constant at entry level (slight step-up in bull)
        # Here we back-calculate from EBITDA margin + SG&A structure
        # Using DG's ~30% gross margin as base, held roughly constant
        gm = 0.304 + i * 0.002          # slight improvement (supply chain)
        d["Gross Profit"] = revenue * gm
        d["COGS"]         = revenue - d["Gross Profit"]
        d["Gross Margin"] = gm

        # --- EBITDA ---
        d["EBITDA"]        = revenue * assumptions.ebitda_margin[i]
        d["EBITDA Margin"] = assumptions.ebitda_margin[i]

        # --- Management Fee (above-the-line for lender reporting) ---
        d["Mgmt Fee"]      = assumptions.mgmt_fee_annual
        d["Adj EBITDA"]    = d["EBITDA"] + d["Mgmt Fee"]  # Lender-adjusted

        # --- D&A ---
        d["D&A"]  = revenue * assumptions.da_pct_revenue[i]
        d["EBIT"] = d["EBITDA"] - d["D&A"]

        # --- Interest Expense ---
        d["Interest Expense"] = interest_expense_by_year[i]

        # --- EBT and Taxes ---
        d["EBT"]       = d["EBIT"] - d["Interest Expense"]
        d["Tax"]       = max(0.0, d["EBT"]) * assumptions.tax_rate
        d["Net Income"]= d["EBT"] - d["Tax"]

    df = pd.DataFrame(data)
    # Round to 1 decimal
    float_rows = [r for r in df.index if r not in ["Gross Margin", "EBITDA Margin"]]
    df.loc[float_rows] = df.loc[float_rows].round(1)
    return df


def is_summary_df(is_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a presentation-ready IS showing only key line items
    with formatting helpers (percent rows flagged separately).
    """
    key_rows = [
        "Revenue",
        "Gross Profit",
        "Gross Margin",
        "EBITDA",
        "EBITDA Margin",
        "Mgmt Fee",
        "Adj EBITDA",
        "D&A",
        "EBIT",
        "Interest Expense",
        "EBT",
        "Tax",
        "Net Income",
    ]
    available = [r for r in key_rows if r in is_df.index]
    return is_df.loc[available].copy()
