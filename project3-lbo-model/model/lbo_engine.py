"""
lbo_engine.py
-------------
Master orchestrator: runs the full LBO model for a given set of
DealAssumptions and returns all outputs in a single ModelResult.

Handles the iterative dependency between the Income Statement
and Debt Schedule (interest expense → IS → FCF → debt schedule →
interest expense).  Two iterations converge for any reasonable deal.

Computes:
  - Opening Balance Sheet
  - Projected IS / CFS / BS (Year 1..N)
  - Debt Schedule (with cash sweep)
  - Exit Analysis: Exit EV, exit equity, IRR, MOIC
  - Returns Attribution Bridge
  - Credit Metrics Time Series

All monetary values in $M.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from model.assumptions import DealAssumptions
from model.income_statement import build_income_statement
from model.cash_flow import build_cash_flow_statement, get_fcf_available
from model.balance_sheet import build_opening_balance_sheet, build_projected_balance_sheets
from model.debt_schedule import build_debt_schedule, get_interest_expense_by_year, get_ending_debt_by_year


# ---------------------------------------------------------------------------
# IRR / MOIC helpers
# ---------------------------------------------------------------------------

def _irr(cash_flows: list[float]) -> float:
    """Compute IRR given a list of cash flows (index 0 = t=0 outflow)."""
    def npv(r):
        return sum(cf / (1 + r) ** t for t, cf in enumerate(cash_flows))
    try:
        return brentq(npv, -0.999, 100.0, xtol=1e-8, maxiter=500)
    except Exception:
        return np.nan


def _moic(invested: float, proceeds: float) -> float:
    if invested <= 0:
        return np.nan
    return proceeds / invested


# ---------------------------------------------------------------------------
# Main model run
# ---------------------------------------------------------------------------

def run_model(assumptions: DealAssumptions) -> dict:
    """
    Run the full LBO model.

    Returns
    -------
    dict with keys:
      opening_bs, is_df, cfs_df, bs_df,
      debt_schedule, returns, credit_metrics, summary
    """
    n = assumptions.hold_years

    # ---- OPENING BALANCE SHEET ----
    opening_bs = build_opening_balance_sheet(assumptions)

    # ---- ITERATIVE IS / DEBT SCHEDULE SOLVE ----
    # Seed interest expense at 0 for iteration 0
    interest_expense = [0.0] * n
    debt_sched = None

    for iteration in range(3):   # 2-3 iterations is sufficient
        # Build IS with current interest estimate
        is_df = build_income_statement(assumptions, interest_expense)

        # Compute FCF available for debt service (pre-interest, uses EBT approx)
        ebitda_list = [is_df.loc["EBITDA", f"Year {i+1}"] for i in range(n)]
        fcf_list    = get_fcf_available(is_df, assumptions)

        # Build debt schedule
        debt_sched = build_debt_schedule(assumptions, fcf_list, ebitda_list)

        # Extract new interest expense estimates for next iteration
        new_interest = get_interest_expense_by_year(debt_sched["schedule"])

        # Check convergence
        max_diff = max(abs(new_interest[i] - interest_expense[i]) for i in range(n))
        interest_expense = new_interest
        if max_diff < 0.5:   # $0.5M tolerance → converged
            break

    # Final IS with converged interest
    is_df = build_income_statement(assumptions, interest_expense)

    # ---- CASH FLOW STATEMENT ----
    # Compute opening NWC from historical BS
    opening_nwc = (
        assumptions.opening_receivables
        + assumptions.opening_inventory
        - assumptions.opening_ap
    )
    cfs_df = build_cash_flow_statement(
        assumptions, is_df, debt_sched["schedule"], opening_nwc
    )

    # ---- PROJECTED BALANCE SHEETS ----
    bs_df = build_projected_balance_sheets(
        assumptions, opening_bs, is_df, cfs_df, debt_sched["schedule"]
    )

    # ---- EXIT ANALYSIS ----
    exit_yr     = assumptions.exit_year
    exit_col    = f"Year {exit_yr}"
    exit_ebitda = is_df.loc["EBITDA", exit_col]
    exit_ev     = exit_ebitda * assumptions.exit_ev_multiple
    exit_debt   = get_ending_debt_by_year(debt_sched["schedule"])[exit_yr - 1]
    exit_cash   = bs_df.loc["Cash & Equivalents", exit_col]
    exit_net_debt = exit_debt - exit_cash
    exit_equity   = exit_ev - exit_net_debt

    # Sponsor equity check (invested at t=0)
    equity_invested = assumptions.sponsor_equity

    # IRR: cash flows = [-invested at t=0,  0 for intermediate (no dividends), exit_equity at t=N]
    # Management fees flow back to LP as distributions — include cumulative mgmt fees as intermediate
    cash_flows = [-equity_invested]
    for yr in range(1, n + 1):
        if yr < exit_yr:
            cash_flows.append(0.0)
        else:
            cash_flows.append(exit_equity)

    irr  = _irr(cash_flows)
    moic = _moic(equity_invested, exit_equity)

    # ---- RETURNS ATTRIBUTION BRIDGE ----
    # Decompose returns into: EBITDA Growth, Multiple Expansion/Contraction, Deleveraging
    entry_ev        = assumptions.entry_ev
    entry_net_debt  = assumptions.total_new_debt - assumptions.min_cash_balance
    entry_eq        = entry_ev - entry_net_debt

    # Component 1: EBITDA Growth (hold entry multiple, apply to delta EBITDA)
    ebitda_growth_contribution = (exit_ebitda - assumptions.entry_ebitda) * assumptions.entry_ev_multiple

    # Component 2: Multiple Expansion (hold exit EBITDA, delta in multiple)
    multiple_expansion_contribution = exit_ebitda * (assumptions.exit_ev_multiple - assumptions.entry_ev_multiple)

    # Component 3: Deleveraging (reduction in net debt)
    deleveraging_contribution = entry_net_debt - exit_net_debt

    # Sanity check: sum should ≈ exit_equity - entry_eq
    total_value_creation = exit_equity - entry_eq

    bridge = {
        "Entry Equity Value ($M)":              round(entry_eq, 1),
        "EBITDA Growth ($M)":                   round(ebitda_growth_contribution, 1),
        "Multiple Expansion / (Contraction) ($M)": round(multiple_expansion_contribution, 1),
        "Deleveraging ($M)":                    round(deleveraging_contribution, 1),
        "Total Value Creation ($M)":            round(total_value_creation, 1),
        "Exit Equity Value ($M)":               round(exit_equity, 1),
    }

    returns = {
        "entry_ebitda":     round(assumptions.entry_ebitda, 1),
        "entry_ev":         round(entry_ev, 1),
        "entry_ev_multiple":assumptions.entry_ev_multiple,
        "entry_equity":     round(equity_invested, 1),
        "entry_net_debt":   round(entry_net_debt, 1),
        "exit_ebitda":      round(exit_ebitda, 1),
        "exit_ev":          round(exit_ev, 1),
        "exit_ev_multiple": assumptions.exit_ev_multiple,
        "exit_net_debt":    round(exit_net_debt, 1),
        "exit_equity":      round(exit_equity, 1),
        "irr":              irr,
        "moic":             moic,
        "hold_years":       n,
        "bridge":           bridge,
    }

    # ---- CREDIT METRICS TIME SERIES ----
    credit = []
    for yr in range(1, n + 1):
        col       = f"Year {yr}"
        ebitda    = is_df.loc["EBITDA", col]
        rev       = is_df.loc["Revenue", col]
        ebit      = is_df.loc["EBIT",   col]
        int_exp   = is_df.loc["Interest Expense", col]
        ni        = is_df.loc["Net Income", col]
        end_debt  = debt_sched["schedule"][yr]["ending_debt"]
        end_cash  = bs_df.loc["Cash & Equivalents", col]
        net_debt  = end_debt - end_cash
        capex     = abs(cfs_df.loc["(-) CapEx", col])
        ocf       = cfs_df.loc["Operating CF", col]

        credit.append({
            "Year":                   yr,
            "Revenue ($M)":           round(rev, 1),
            "EBITDA ($M)":            round(ebitda, 1),
            "EBITDA Margin":          ebitda / rev if rev else 0,
            "Gross Leverage (x)":     round(end_debt / ebitda, 2) if ebitda > 0 else np.nan,
            "Net Leverage (x)":       round(net_debt  / ebitda, 2) if ebitda > 0 else np.nan,
            "Interest Coverage (x)":  round(ebitda / int_exp, 2) if int_exp > 0 else np.nan,
            "DSCR (x)":               round(ocf / int_exp, 2) if int_exp > 0 else np.nan,
            "Net Income ($M)":        round(ni, 1),
            "CapEx ($M)":             round(capex, 1),
            "CapEx / Revenue":        capex / rev if rev else 0,
            "FCF Yield (on equity)":  round(cfs_df.loc["Free Cash Flow", col] / equity_invested, 4),
        })
    credit_df = pd.DataFrame(credit)

    # ---- SUMMARY TABLE ----
    summary = {
        "Entry EV ($M)":           round(entry_ev, 1),
        "Entry EV/EBITDA":         f"{assumptions.entry_ev_multiple:.1f}x",
        "Entry Gross Leverage":    f"{assumptions.entry_leverage:.1f}x",
        "Equity Invested ($M)":    round(equity_invested, 1),
        "Exit EV ($M)":            round(exit_ev, 1),
        "Exit EV/EBITDA":          f"{assumptions.exit_ev_multiple:.1f}x",
        "Exit EBITDA ($M)":        round(exit_ebitda, 1),
        "Exit Net Debt ($M)":      round(exit_net_debt, 1),
        "Exit Equity ($M)":        round(exit_equity, 1),
        "IRR":                     f"{irr:.1%}" if not np.isnan(irr) else "N/A",
        "MOIC":                    f"{moic:.2f}x" if not np.isnan(moic) else "N/A",
        "Hold Period":             f"{n} years",
    }

    return {
        "opening_bs":    opening_bs,
        "is_df":         is_df,
        "cfs_df":        cfs_df,
        "bs_df":         bs_df,
        "debt_schedule": debt_sched,
        "returns":       returns,
        "credit_df":     credit_df,
        "summary":       summary,
    }
