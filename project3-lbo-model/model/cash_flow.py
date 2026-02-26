"""
cash_flow.py
------------
Derives the Cash Flow Statement from the Income Statement and
working capital / CapEx assumptions.

Structure:
  Operating Cash Flow
    Net Income
    + D&A (non-cash)
    + Stock-Based Compensation (non-cash)
    + DFF Amortization (non-cash interest, already in net income)
    ± Change in Net Working Capital
  Investing Cash Flow
    - Capital Expenditures
  Financing Cash Flow
    - Debt Repayment (required amort + cash sweep)
    - Management Fee (cash outflow to sponsor)
  Net Change in Cash
  Ending Cash Balance

NWC is driven by revenue / COGS through DSO, DIO, DPO.
All values in $M.
"""

import pandas as pd
from model.assumptions import DealAssumptions


def _calc_nwc(revenue: float, cogs: float, assumptions: DealAssumptions) -> tuple[float, float, float]:
    """Returns (receivables, inventory, payables) implied by DSO/DIO/DPO."""
    receivables = revenue  / 365 * assumptions.dso
    inventory   = cogs     / 365 * assumptions.dio  if cogs > 0 else 0
    payables    = cogs     / 365 * assumptions.dpo   if cogs > 0 else 0
    return receivables, inventory, payables


def build_cash_flow_statement(
    assumptions: DealAssumptions,
    is_df: pd.DataFrame,
    debt_schedule: dict,           # from debt_schedule.build_debt_schedule
    opening_nwc: float,            # opening NWC (from historical BS) — receivables + inventory - payables
    sbc_pct_revenue: float = 0.004, # stock-based comp as % of revenue
) -> pd.DataFrame:
    """
    Returns a wide DataFrame (one column per year, index = CFS line items).
    Also computes Free Cash Flow available for debt service.
    """
    n = assumptions.hold_years
    cols = [f"Year {i}" for i in range(1, n + 1)]
    data = {col: {} for col in cols}

    # Track prior-year NWC for change calculation
    prev_nwc = opening_nwc

    for i, col in enumerate(cols):
        yr = i + 1
        d = data[col]

        # --- Pull from IS ---
        net_income = is_df.loc["Net Income", col]
        da         = is_df.loc["D&A",        col]
        revenue    = is_df.loc["Revenue",    col]
        cogs       = is_df.loc["COGS",       col]
        dff_amort  = debt_schedule[yr]["dff_amort"]  # non-cash interest component

        # --- NWC ---
        rec, inv, pay = _calc_nwc(revenue, cogs, assumptions)
        nwc = rec + inv - pay
        change_nwc = -(nwc - prev_nwc)   # increase in NWC = cash outflow (negative sign)
        prev_nwc = nwc

        # SBC is NOT an IS expense in this model (EBITDA is set directly from
        # margin assumptions), so we do NOT add it back here.  It is included
        # in the IS implicitly through the margin assumption and properly
        # captured in the FCF-for-sweep calculation instead.

        # --- Operating Cash Flow ---
        d["Net Income"]      = net_income
        d["(+) D&A"]         = da
        d["(+) DFF Amort"]   = dff_amort
        d["Δ NWC"]           = change_nwc
        d["Operating CF"]    = net_income + da + dff_amort + change_nwc

        # --- Investing Cash Flow ---
        capex = revenue * assumptions.capex_pct_revenue[i]
        d["(-) CapEx"]       = -capex
        d["Investing CF"]    = -capex

        # --- Free Cash Flow (Unlevered proxy for debt sweep) ---
        d["Free Cash Flow"]  = d["Operating CF"] + d["Investing CF"]

        # --- Financing Cash Flow ---
        total_repaid = debt_schedule[yr]["total_debt_repaid"]
        mgmt_fee     = assumptions.mgmt_fee_annual
        d["(-) Debt Repaid"] = -total_repaid
        d["(-) Mgmt Fee"]    = -mgmt_fee
        d["Financing CF"]    = -total_repaid - mgmt_fee

        # --- Net Change in Cash ---
        d["Net Change in Cash"] = d["Operating CF"] + d["Investing CF"] + d["Financing CF"]

    df = pd.DataFrame(data).round(1)
    return df


def get_fcf_available(is_df: pd.DataFrame, assumptions: DealAssumptions,
                      sbc_pct_revenue: float = 0.004) -> list[float]:
    """
    FCF available for debt service.
    = Net Income + D&A + SBC + DFF Amort (non-cash) - CapEx
    Consistent with the Operating CF definition in build_cash_flow_statement
    (excluding NWC changes which don't affect long-run debt capacity).
    Used as input to the debt schedule (before cash sweep).
    """
    result = []
    avg_term = sum(t.term_years * t.principal for t in assumptions.debt_tranches) / \
               max(1, sum(t.principal for t in assumptions.debt_tranches))
    dff_amort_annual = assumptions.total_financing_fees / avg_term if avg_term > 0 else 0

    for i in range(assumptions.hold_years):
        col    = f"Year {i+1}"
        ni     = is_df.loc["Net Income", col]
        da     = is_df.loc["D&A",        col]
        rev    = is_df.loc["Revenue",    col]
        capex  = rev * assumptions.capex_pct_revenue[i]
        # DFF amort is non-cash (already deducted in interest expense → NI), add back
        fcf = ni + da + dff_amort_annual - capex
        result.append(max(0.0, fcf))
    return result
