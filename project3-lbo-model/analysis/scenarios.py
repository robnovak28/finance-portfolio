"""
scenarios.py
------------
Defines Bull / Base / Bear scenarios and runs the LBO model for each.
Returns a comparison DataFrame plus individual ModelResult dicts.
"""

import numpy as np
import pandas as pd
from model.assumptions import DealAssumptions, base_case, bull_case, bear_case
from model.lbo_engine import run_model


SCENARIO_NAMES = ["Bear", "Base", "Bull"]


def _make_scenario_assumptions(base: DealAssumptions | None = None) -> dict[str, DealAssumptions]:
    from copy import deepcopy
    if base is None:
        base = base_case()

    bull = deepcopy(base)
    bull.revenue_growth   = [r + 0.020 for r in base.revenue_growth]
    bull.ebitda_margin    = [m + 0.010 for m in base.ebitda_margin]
    bull.exit_ev_multiple = base.exit_ev_multiple + 1.5

    bear = deepcopy(base)
    bear.revenue_growth   = [r - 0.025 for r in base.revenue_growth]
    bear.ebitda_margin    = [m - 0.012 for m in base.ebitda_margin]
    bear.exit_ev_multiple = max(4.0, base.exit_ev_multiple - 1.5)

    return {
        "Bear": bear,
        "Base": deepcopy(base),
        "Bull": bull,
    }


def run_scenarios(base_assumptions: DealAssumptions | None = None) -> dict:
    """
    Run all three scenarios.

    Returns
    -------
    {
      "results"      : {scenario_name: model_result_dict},
      "assumptions"  : {scenario_name: DealAssumptions},
      "comparison_df": pd.DataFrame  (key metrics across scenarios),
      "returns_df"   : pd.DataFrame  (IRR / MOIC summary table),
    }
    """
    scenarios = _make_scenario_assumptions(base_assumptions)
    results   = {}
    for name, assum in scenarios.items():
        results[name] = run_model(assum)

    # Build comparison DataFrame
    rows = []
    for name in SCENARIO_NAMES:
        a  = scenarios[name]
        r  = results[name]["returns"]
        cr = results[name]["credit_df"]
        n  = a.hold_years
        yr5_col = cr[cr["Year"] == n].iloc[0] if not cr.empty else {}

        irr   = r["irr"]
        moic  = r["moic"]
        rows.append({
            "Scenario":                  name,
            "Rev CAGR (Yr1–Yr5)":       f"{_cagr(a.entry_revenue, r['exit_ebitda'] / a.ebitda_margin[-1], n):.1%}",
            "Exit EBITDA Margin":        f"{a.ebitda_margin[-1]:.1%}",
            "Exit EV/EBITDA":            f"{a.exit_ev_multiple:.1f}x",
            "Entry EV ($M)":             f"${r['entry_ev']:,.0f}",
            "Entry Leverage":            f"{a.entry_leverage:.1f}x",
            "Exit EV ($M)":              f"${r['exit_ev']:,.0f}",
            "Exit Net Debt ($M)":        f"${r['exit_net_debt']:,.0f}",
            "Exit Equity ($M)":          f"${r['exit_equity']:,.0f}",
            "IRR":                       f"{irr:.1%}" if not np.isnan(irr) else "N/A",
            "MOIC":                      f"{moic:.2f}x" if not np.isnan(moic) else "N/A",
            "Yr5 Net Leverage":          f"{yr5_col.get('Net Leverage (x)', np.nan):.2f}x" if isinstance(yr5_col, pd.Series) else "N/A",
        })

    comparison_df = pd.DataFrame(rows).set_index("Scenario")

    # Compact returns summary
    returns_df = pd.DataFrame({
        "Metric": ["IRR", "MOIC", "Entry EV ($M)", "Exit EV ($M)",
                   "Exit Equity ($M)", "Equity Invested ($M)"],
        **{
            name: [
                f"{results[name]['returns']['irr']:.1%}" if not np.isnan(results[name]['returns']['irr']) else "N/A",
                f"{results[name]['returns']['moic']:.2f}x",
                f"${results[name]['returns']['entry_ev']:,.0f}",
                f"${results[name]['returns']['exit_ev']:,.0f}",
                f"${results[name]['returns']['exit_equity']:,.0f}",
                f"${results[name]['returns']['entry_equity']:,.0f}",
            ]
            for name in SCENARIO_NAMES
        }
    }).set_index("Metric")

    return {
        "results":       results,
        "assumptions":   scenarios,
        "comparison_df": comparison_df,
        "returns_df":    returns_df,
    }


def _cagr(entry_rev: float, exit_rev: float, years: int) -> float:
    if entry_rev <= 0 or years <= 0:
        return 0.0
    return (exit_rev / entry_rev) ** (1 / years) - 1
