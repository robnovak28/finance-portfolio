"""
debt_schedule.py
----------------
Builds the full debt amortization schedule for all tranches.

Key mechanics:
  - Required amortization per tranche (fixed % of original principal)
  - Cash sweep: excess free cash flow applied to debt in seniority order
    (Revolver → TLA → TLB → Sr Notes → HY Notes)
  - Interest expense calculated on average beginning + ending balance
  - Deferred financing fees amortized over tranche life (adds to interest)
  - Running net leverage and interest coverage tracked

Returns a dict keyed by year (1..N) with full schedule detail,
plus a summary DataFrame.
"""

import numpy as np
import pandas as pd
from model.assumptions import DealAssumptions, DebtTranche


# Cash sweep priority (lower index = higher priority)
SWEEP_ORDER = ["Revolver", "Term Loan A", "Term Loan B",
               "Senior Secured Notes", "High Yield Notes"]


def _sweep_priority(tranche_name: str) -> int:
    for i, name in enumerate(SWEEP_ORDER):
        if name.lower() in tranche_name.lower():
            return i
    return 99


def build_debt_schedule(
    assumptions: DealAssumptions,
    fcf_available: list[float],   # Free cash flow available each year (after CapEx, taxes)
    ebitda_by_year: list[float],  # For leverage / coverage stats
) -> dict:
    """
    Parameters
    ----------
    assumptions    : DealAssumptions
    fcf_available  : list len=N of FCF available for debt service (before interest)
    ebitda_by_year : list len=N of projected EBITDA

    Returns
    -------
    {
      "schedule"     : dict[year] → dict with per-tranche balances + totals
      "summary_df"   : pd.DataFrame  (year-by-year summary)
      "tranche_dfs"  : dict[name] → pd.DataFrame  (per-tranche detail)
    }
    """
    n = assumptions.hold_years
    tranches = assumptions.debt_tranches
    sorted_tranches = sorted(tranches, key=lambda t: _sweep_priority(t.name))

    # Opening balances
    balances = {t.name: t.principal for t in tranches}
    # Deferred financing fee balances (amortized straight-line over tranche life)
    dff_balances = {t.name: t.financing_fee for t in tranches}

    schedule = {}
    tranche_records = {t.name: [] for t in tranches}

    for yr in range(1, n + 1):
        yr_data = {"year": yr}
        fcf = fcf_available[yr - 1]
        ebitda = ebitda_by_year[yr - 1]

        # Step 1: Compute required amortization (based on original principal)
        req_amort = {t.name: t.annual_amort for t in tranches if not t.is_revolver}
        for t in tranches:
            if t.is_revolver:
                req_amort[t.name] = 0.0

        # Step 2: Pay required amortization from FCF
        total_req_amort = sum(req_amort.values())
        fcf_after_req_amort = fcf - total_req_amort

        # Step 3: Cash sweep (apply excess FCF to debt, highest priority first)
        sweep_payments = {t.name: 0.0 for t in tranches}
        remaining_for_sweep = max(0.0, fcf_after_req_amort) * assumptions.cash_sweep_pct

        for t in sorted_tranches:
            if remaining_for_sweep <= 0:
                break
            # Can't sweep more than outstanding balance
            max_sweep = max(0.0, balances[t.name] - req_amort.get(t.name, 0.0))
            # Check call protection (notes may not be callable yet)
            if t.call_protection_yrs > 0 and yr <= t.call_protection_yrs:
                continue
            sweep = min(remaining_for_sweep, max_sweep)
            sweep_payments[t.name] = sweep
            remaining_for_sweep -= sweep

        # Step 4: Update balances
        ending_balances = {}
        for t in tranches:
            paid = req_amort.get(t.name, 0.0) + sweep_payments[t.name]
            ending_balances[t.name] = max(0.0, balances[t.name] - paid)

        # Step 5: Interest expense (on average balance, beginning + ending)
        # Also add deferred financing fee amortization (DFF amort)
        interest = {}
        dff_amort = {}
        for t in tranches:
            avg_balance = (balances[t.name] + ending_balances[t.name]) / 2
            interest[t.name] = avg_balance * t.all_in_rate
            # DFF amort: straight-line over tranche life
            dff_amort[t.name] = t.financing_fee / t.term_years if t.term_years > 0 else 0.0
            dff_balances[t.name] = max(0.0, dff_balances[t.name] - dff_amort[t.name])

        total_cash_interest = sum(interest.values())
        total_dff_amort     = sum(dff_amort.values())
        # GAAP interest = cash interest + DFF amort (non-cash)
        total_interest_expense = total_cash_interest + total_dff_amort

        total_ending_debt = sum(ending_balances.values())
        total_beg_debt    = sum(balances.values())
        total_amort       = sum(req_amort.values()) + sum(sweep_payments.values())

        # Credit metrics
        leverage = total_ending_debt / ebitda if ebitda > 0 else np.nan
        interest_coverage = ebitda / total_interest_expense if total_interest_expense > 0 else np.nan

        yr_data.update({
            "beginning_debt":      total_beg_debt,
            "required_amort":      total_req_amort,
            "cash_sweep":          sum(sweep_payments.values()),
            "total_debt_repaid":   total_amort,
            "ending_debt":         total_ending_debt,
            "cash_interest":       total_cash_interest,
            "dff_amort":           total_dff_amort,
            "total_interest_expense": total_interest_expense,
            "fcf_available":       fcf,
            "remaining_fcf":       fcf - total_amort,
            "gross_leverage":      leverage,
            "interest_coverage":   interest_coverage,
            "balances_by_tranche": {**ending_balances},
            "interest_by_tranche": {**interest},
            "sweep_by_tranche":    {**sweep_payments},
            "req_amort_by_tranche":{**req_amort},
        })

        # Per-tranche detail records
        for t in tranches:
            tranche_records[t.name].append({
                "Year":                     yr,
                "Beginning Balance ($M)":   round(balances[t.name], 1),
                "Required Amort ($M)":      round(req_amort.get(t.name, 0.0), 1),
                "Cash Sweep ($M)":          round(sweep_payments[t.name], 1),
                "Total Repaid ($M)":        round(req_amort.get(t.name, 0.0) + sweep_payments[t.name], 1),
                "Ending Balance ($M)":      round(ending_balances[t.name], 1),
                "Rate":                     f"{t.all_in_rate:.2%}",
                "Interest Expense ($M)":    round(interest[t.name], 1),
                "DFF Amort ($M)":           round(dff_amort[t.name], 1),
            })

        schedule[yr] = yr_data
        balances = ending_balances

    # Build summary DataFrame
    rows = []
    for yr, d in schedule.items():
        row = {
            "Year":                   yr,
            "Beg. Total Debt ($M)":   round(d["beginning_debt"], 1),
            "Req. Amortization ($M)": round(d["required_amort"], 1),
            "Cash Sweep ($M)":        round(d["cash_sweep"], 1),
            "Total Repaid ($M)":      round(d["total_debt_repaid"], 1),
            "End. Total Debt ($M)":   round(d["ending_debt"], 1),
            "Cash Interest ($M)":     round(d["cash_interest"], 1),
            "DFF Amort ($M)":         round(d["dff_amort"], 1),
            "Interest Expense ($M)":  round(d["total_interest_expense"], 1),
            "Gross Leverage (x)":     round(d["gross_leverage"], 2) if not np.isnan(d["gross_leverage"]) else np.nan,
            "Interest Coverage (x)":  round(d["interest_coverage"], 2) if not np.isnan(d["interest_coverage"]) else np.nan,
        }
        # Per-tranche ending balances
        for name, bal in d["balances_by_tranche"].items():
            row[f"{name} ($M)"] = round(bal, 1)
        rows.append(row)

    summary_df = pd.DataFrame(rows)

    tranche_dfs = {
        name: pd.DataFrame(records) for name, records in tranche_records.items()
    }

    return {
        "schedule":    schedule,
        "summary_df":  summary_df,
        "tranche_dfs": tranche_dfs,
    }


def get_interest_expense_by_year(schedule: dict) -> list[float]:
    """Extract total GAAP interest expense list from schedule dict."""
    return [schedule[yr]["total_interest_expense"] for yr in sorted(schedule.keys())]


def get_ending_debt_by_year(schedule: dict) -> list[float]:
    """Extract ending total debt by year."""
    return [schedule[yr]["ending_debt"] for yr in sorted(schedule.keys())]


def sources_uses_df(assumptions: DealAssumptions) -> pd.DataFrame:
    """Build Sources & Uses table for display."""
    tranches = assumptions.debt_tranches

    sources = []
    for t in tranches:
        label = f"{t.name} ({'SOFR+{:.0f}bps'.format(t.spread*1e4) if t.is_floating else '{:.2%} fixed'.format(t.fixed_rate)}, {t.term_years}yr)"
        sources.append({"Item": label, "Amount ($M)": t.principal, "% of Total": None})
    sources.append({"Item": "Sponsor Equity", "Amount ($M)": assumptions.sponsor_equity, "% of Total": None})

    total_sources = sum(s["Amount ($M)"] for s in sources)
    for s in sources:
        s["% of Total"] = f"{s['Amount ($M)'] / total_sources:.1%}"

    uses = [
        {"Item": "Purchase Equity (EV – Net Debt)", "Amount ($M)": round(assumptions.entry_equity_value, 1), "% of Total": None},
        {"Item": "Repay Existing Gross Debt",        "Amount ($M)": assumptions.existing_gross_debt,          "% of Total": None},
        {"Item": "M&A Advisory & Legal Fees",        "Amount ($M)": round(assumptions.advisory_fees, 1),      "% of Total": None},
        {"Item": "Financing Fees / OID",             "Amount ($M)": round(assumptions.total_financing_fees, 1),"% of Total": None},
        {"Item": "Cash to Balance Sheet",            "Amount ($M)": assumptions.min_cash_balance,              "% of Total": None},
    ]
    total_uses = sum(u["Amount ($M)"] for u in uses)
    for u in uses:
        u["% of Total"] = f"{u['Amount ($M)'] / total_uses:.1%}"

    sources_df = pd.DataFrame(sources)
    uses_df    = pd.DataFrame(uses)
    # Add totals row
    sources_df = pd.concat([
        sources_df,
        pd.DataFrame([{"Item": "Total Sources", "Amount ($M)": round(total_sources, 1), "% of Total": "100.0%"}])
    ], ignore_index=True)
    uses_df = pd.concat([
        uses_df,
        pd.DataFrame([{"Item": "Total Uses", "Amount ($M)": round(total_uses, 1), "% of Total": "100.0%"}])
    ], ignore_index=True)

    return sources_df, uses_df
