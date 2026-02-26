"""
sensitivity.py
--------------
Two-way sensitivity tables for LBO returns.

Table 1: Entry EV/EBITDA (rows) vs Exit EV/EBITDA (cols) → IRR and MOIC
Table 2: Revenue CAGR (rows) vs Exit EV/EBITDA (cols) → IRR
Table 3: Entry EV/EBITDA (rows) vs Entry Leverage (cols) → IRR

Results are color-coded:
  IRR < 15%          → red
  15% ≤ IRR < 20%    → yellow
  IRR ≥ 20%          → green  (displayed as raw floats for Streamlit styling)
"""

import numpy as np
import pandas as pd
from model.assumptions import DealAssumptions, base_case
from model.lbo_engine import run_model


def _run_point(base: DealAssumptions, **overrides) -> tuple[float, float]:
    """Run model with scalar overrides, return (irr, moic)."""
    a = DealAssumptions(
        entry_ebitda      = overrides.get("entry_ebitda",      base.entry_ebitda),
        entry_ev_multiple = overrides.get("entry_ev_multiple", base.entry_ev_multiple),
        entry_revenue     = overrides.get("entry_revenue",     base.entry_revenue),
        existing_net_debt = base.existing_net_debt,
        existing_cash     = base.existing_cash,
        existing_gross_debt = base.existing_gross_debt,
        revenue_growth    = overrides.get("revenue_growth",    base.revenue_growth),
        ebitda_margin     = overrides.get("ebitda_margin",     base.ebitda_margin),
        da_pct_revenue    = base.da_pct_revenue,
        capex_pct_revenue = base.capex_pct_revenue,
        tax_rate          = base.tax_rate,
        exit_ev_multiple  = overrides.get("exit_ev_multiple",  base.exit_ev_multiple),
        hold_years        = base.hold_years,
        debt_tranches     = overrides.get("debt_tranches",     base.debt_tranches),
        opening_receivables    = base.opening_receivables,
        opening_inventory      = base.opening_inventory,
        opening_other_current  = base.opening_other_current,
        opening_ppe            = base.opening_ppe,
        opening_op_lease_rou   = base.opening_op_lease_rou,
        opening_other_lt_assets= base.opening_other_lt_assets,
        opening_ap             = base.opening_ap,
        opening_accrued        = base.opening_accrued,
        opening_current_lease  = base.opening_current_lease,
        opening_other_current_liab = base.opening_other_current_liab,
        opening_lt_lease       = base.opening_lt_lease,
        opening_other_lt_liab  = base.opening_other_lt_liab,
    )
    try:
        r = run_model(a)["returns"]
        return r["irr"], r["moic"]
    except Exception:
        return np.nan, np.nan


def entry_vs_exit_multiple(
    base: DealAssumptions | None = None,
    entry_multiples: list[float] = [8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0],
    exit_multiples:  list[float] = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (irr_table, moic_table).
    Rows = entry EV/EBITDA, Columns = exit EV/EBITDA.
    """
    base = base or base_case()
    irr_data  = {}
    moic_data = {}

    for exit_m in exit_multiples:
        irr_col  = {}
        moic_col = {}
        for entry_m in entry_multiples:
            irr, moic = _run_point(base,
                                   entry_ev_multiple=entry_m,
                                   exit_ev_multiple=exit_m)
            irr_col[f"{entry_m:.1f}x"]  = irr
            moic_col[f"{entry_m:.1f}x"] = moic
        irr_data[f"Exit {exit_m:.1f}x"]  = irr_col
        moic_data[f"Exit {exit_m:.1f}x"] = moic_col

    irr_df  = pd.DataFrame(irr_data)
    moic_df = pd.DataFrame(moic_data)
    irr_df.index.name  = "Entry Multiple"
    moic_df.index.name = "Entry Multiple"
    return irr_df, moic_df


def rev_growth_vs_exit_multiple(
    base: DealAssumptions | None = None,
    rev_cagrs:      list[float] = [-0.02, 0.00, 0.02, 0.04, 0.06, 0.08],
    exit_multiples: list[float] = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0],
) -> pd.DataFrame:
    """
    IRR sensitivity: rows = revenue CAGR, cols = exit multiple.
    """
    base = base or base_case()
    n = base.hold_years
    irr_data = {}

    for exit_m in exit_multiples:
        col = {}
        for cagr in rev_cagrs:
            rev_growth = [cagr] * n
            irr, _ = _run_point(base,
                                 revenue_growth=rev_growth,
                                 exit_ev_multiple=exit_m)
            col[f"{cagr:+.0%}"] = irr
        irr_data[f"Exit {exit_m:.1f}x"] = col

    df = pd.DataFrame(irr_data)
    df.index.name = "Revenue CAGR"
    return df


def leverage_vs_entry_multiple(
    base: DealAssumptions | None = None,
    leverage_levels: list[float] = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0],
    entry_multiples: list[float] = [8.5, 9.5, 10.5, 11.5, 12.5],
) -> pd.DataFrame:
    """
    IRR sensitivity: rows = entry leverage (x EBITDA), cols = entry EV/EBITDA.
    Scales total debt proportionally across tranches.
    """
    base   = base or base_case()
    ebitda = base.entry_ebitda
    irr_data = {}

    for entry_m in entry_multiples:
        col = {}
        for lev in leverage_levels:
            # Scale all debt tranches proportionally to hit target leverage
            target_debt = lev * ebitda
            orig_debt   = sum(t.principal for t in base.debt_tranches)
            scale       = target_debt / orig_debt if orig_debt > 0 else 1.0

            import copy
            new_tranches = copy.deepcopy(base.debt_tranches)
            for t in new_tranches:
                t.principal = t.principal * scale

            irr, _ = _run_point(base,
                                  entry_ev_multiple=entry_m,
                                  debt_tranches=new_tranches)
            col[f"{lev:.1f}x"] = irr
        irr_data[f"Entry {entry_m:.1f}x"] = col

    df = pd.DataFrame(irr_data)
    df.index.name = "Leverage"
    return df


def irr_color(val: float) -> str:
    """Return a CSS color class string for a given IRR value."""
    if np.isnan(val):
        return "background-color: #555555; color: white"
    if val < 0.15:
        return "background-color: #e74c3c; color: white"
    if val < 0.20:
        return "background-color: #f39c12; color: black"
    if val < 0.25:
        return "background-color: #2ecc71; color: black"
    return "background-color: #1abc9c; color: black"
