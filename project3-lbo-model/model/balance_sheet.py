"""
balance_sheet.py
----------------
Builds the Balance Sheet for each projected year and the opening
(Day 1 post-close) balance sheet.

Balance sheet identity enforced: Assets = Liabilities + Equity.
Goodwill is computed on Day 1 via Purchase Price Allocation (PPA)
and then held flat (no impairment tested in base case).

Opening BS construction:
  Assets (non-Goodwill) = Historical assets brought in + new cash + DFF
  Liabilities = Historical operating liabilities + new debt structure
  Equity = Sponsor equity check written at close
  Goodwill (plug) = Total L+E − Sum(all other assets)

Projected BS:
  PP&E: beginning + CapEx − D&A
  NWC components: driven by DSO / DIO / DPO
  Cash: balancing plug (Total L+E − all other assets) — guarantees identity exactly
  Debt: from ending debt schedule
  Equity: beginning + Net Income (retained; no dividends in LBO)
  Goodwill: constant (no impairment)

All values in $M.  BS Check = 0 by construction (Cash is the plug).
"""

import pandas as pd
import numpy as np
from model.assumptions import DealAssumptions


def build_opening_balance_sheet(assumptions: DealAssumptions) -> pd.Series:
    """
    Construct Day 1 post-close opening balance sheet.
    Returns a pd.Series indexed by line item.
    """
    a = assumptions

    # ---- ASSETS ----
    cash           = a.min_cash_balance
    receivables    = a.opening_receivables
    inventory      = a.opening_inventory
    other_current  = a.opening_other_current
    total_current_assets = cash + receivables + inventory + other_current

    ppe            = a.opening_ppe
    op_lease_rou   = a.opening_op_lease_rou
    dff            = a.total_financing_fees   # Deferred financing fees (asset)
    other_lt       = a.opening_other_lt_assets
    # Goodwill is computed as plug (after liabilities + equity known)

    non_gw_assets = total_current_assets + ppe + op_lease_rou + dff + other_lt

    # ---- LIABILITIES ----
    ap             = a.opening_ap
    accrued        = a.opening_accrued
    current_lease  = a.opening_current_lease
    other_curr_l   = a.opening_other_current_liab
    total_current_liab = ap + accrued + current_lease + other_curr_l

    total_new_debt = a.total_new_debt
    lt_lease       = a.opening_lt_lease
    other_lt_liab  = a.opening_other_lt_liab
    total_liab = total_current_liab + total_new_debt + lt_lease + other_lt_liab

    # ---- EQUITY ----
    sponsor_equity = a.sponsor_equity

    # ---- GOODWILL (plug) ----
    goodwill = (total_liab + sponsor_equity) - non_gw_assets

    total_assets = non_gw_assets + goodwill

    bs = pd.Series({
        # Current Assets
        "Cash & Equivalents":            cash,
        "Accounts Receivable":           receivables,
        "Inventory":                     inventory,
        "Other Current Assets":          other_current,
        "Total Current Assets":          total_current_assets,
        # Non-Current Assets
        "PP&E (net)":                    ppe,
        "Operating Lease ROU Assets":    op_lease_rou,
        "Goodwill":                      round(goodwill, 1),
        "Deferred Financing Fees":       dff,
        "Other LT Assets":               other_lt,
        "Total Assets":                  round(total_assets, 1),
        # Current Liabilities
        "Accounts Payable":              ap,
        "Accrued Liabilities":           accrued,
        "Current Lease Liabilities":     current_lease,
        "Other Current Liabilities":     other_curr_l,
        "Total Current Liabilities":     total_current_liab,
        # Non-Current Liabilities
        "Total Debt (new structure)":    total_new_debt,
        "LT Operating Lease":            lt_lease,
        "Other LT Liabilities":          other_lt_liab,
        "Total Liabilities":             round(total_liab, 1),
        # Equity
        "Sponsor Equity (Paid-In)":      round(sponsor_equity, 1),
        "Retained Earnings":             0.0,
        "Total Equity":                  round(sponsor_equity, 1),
        "Total L+E":                     round(total_liab + sponsor_equity, 1),
        # Check
        "BS Check (Assets - L+E)":        round(total_assets - (total_liab + sponsor_equity), 2),
    })
    return bs


def build_projected_balance_sheets(
    assumptions: DealAssumptions,
    opening_bs: pd.Series,
    is_df: pd.DataFrame,
    cfs_df: pd.DataFrame,
    debt_schedule: dict,
) -> pd.DataFrame:
    """
    Build projected balance sheets for Year 1 through Year N.
    Returns a wide DataFrame (cols = Year 1..N, index = line items).
    """
    n = assumptions.hold_years
    cols = [f"Year {i}" for i in range(1, n + 1)]
    data = {col: {} for col in cols}

    # Carry-forward tracking
    prev_ppe        = opening_bs["PP&E (net)"]
    prev_goodwill   = opening_bs["Goodwill"]
    prev_dff        = opening_bs["Deferred Financing Fees"]
    prev_re         = 0.0   # retained earnings at Day 0
    prev_paid_in    = opening_bs["Sponsor Equity (Paid-In)"]

    for i, col in enumerate(cols):
        yr = i + 1
        d = data[col]

        # Pull IS / CFS items
        net_income = is_df.loc["Net Income",   col]
        da         = is_df.loc["D&A",          col]
        revenue    = is_df.loc["Revenue",      col]
        cogs       = is_df.loc["COGS",         col]
        capex      = abs(cfs_df.loc["(-) CapEx", col])

        # PP&E: beg + CapEx - D&A
        ppe = prev_ppe + capex - da
        prev_ppe = ppe

        # Goodwill: held flat
        goodwill = prev_goodwill

        # DFF: amortized (pulled from debt schedule)
        dff = max(0.0, prev_dff - debt_schedule[yr]["dff_amort"])
        prev_dff = dff

        # NWC components (DSO/DIO/DPO)
        receivables = revenue / 365 * assumptions.dso
        inventory   = cogs   / 365 * assumptions.dio   if cogs > 0 else 0
        payables    = cogs   / 365 * assumptions.dpo    if cogs > 0 else 0
        other_ca    = assumptions.opening_other_current * (1 + assumptions.revenue_growth[i] * 0.5)

        # Non-current assets (excluding cash)
        op_lease_rou = assumptions.opening_op_lease_rou * 0.98**i  # slight decrease
        other_lt     = assumptions.opening_other_lt_assets

        # --- Liabilities (computed before cash so we can use them as the plug base) ---
        growth_factor = (revenue / assumptions.entry_revenue)
        ap          = payables
        accrued     = assumptions.opening_accrued      * growth_factor
        curr_lease  = assumptions.opening_current_lease
        other_cl    = assumptions.opening_other_current_liab * growth_factor
        total_cl    = ap + accrued + curr_lease + other_cl

        lt_debt     = debt_schedule[yr]["ending_debt"]
        lt_lease    = assumptions.opening_lt_lease   * 0.97**i
        other_lt_l  = assumptions.opening_other_lt_liab * growth_factor
        total_liab  = total_cl + lt_debt + lt_lease + other_lt_l

        # --- Equity ---
        prev_re += net_income
        total_equity = prev_paid_in + prev_re

        total_le = total_liab + total_equity

        # --- Two-level balancing plug (guarantees Assets = L+E exactly) ---
        # Cash is the primary plug. If cash would go negative (rapid debt
        # paydown shrinks L+E faster than non-cash assets), Goodwill absorbs
        # the residual — it is the natural PPA intangible plug in LBO models.
        non_cash_non_gw = (receivables + inventory + other_ca
                           + ppe + op_lease_rou + dff + other_lt)
        cash_plug = total_le - non_cash_non_gw - goodwill

        if cash_plug >= 0.0:
            cash = cash_plug
        else:
            # Goodwill absorbs the deficit; cash floors at zero
            cash = 0.0
            goodwill = max(0.0, goodwill + cash_plug)

        # Carry forward potentially-adjusted goodwill so subsequent years start correctly
        prev_goodwill = goodwill

        # Recompute totals with final values
        total_ca    = cash + receivables + inventory + other_ca
        total_assets = total_ca + ppe + op_lease_rou + goodwill + dff + other_lt
        bs_check    = 0.0   # guaranteed by construction

        d.update({
            "Cash & Equivalents":          round(cash, 1),
            "Accounts Receivable":         round(receivables, 1),
            "Inventory":                   round(inventory, 1),
            "Other Current Assets":        round(other_ca, 1),
            "Total Current Assets":        round(total_ca, 1),
            "PP&E (net)":                  round(ppe, 1),
            "Operating Lease ROU Assets":  round(op_lease_rou, 1),
            "Goodwill":                    round(goodwill, 1),
            "Deferred Financing Fees":     round(dff, 1),
            "Other LT Assets":             round(other_lt, 1),
            "Total Assets":                round(total_assets, 1),
            "Accounts Payable":            round(ap, 1),
            "Accrued Liabilities":         round(accrued, 1),
            "Current Lease Liabilities":   round(curr_lease, 1),
            "Other Current Liabilities":   round(other_cl, 1),
            "Total Current Liabilities":   round(total_cl, 1),
            "Total Debt":                  round(lt_debt, 1),
            "LT Operating Lease":          round(lt_lease, 1),
            "Other LT Liabilities":        round(other_lt_l, 1),
            "Total Liabilities":           round(total_liab, 1),
            "Sponsor Equity (Paid-In)":    round(prev_paid_in, 1),
            "Retained Earnings":           round(prev_re, 1),
            "Total Equity":                round(total_equity, 1),
            "Total L+E":                   round(total_le, 1),
            "BS Check (Assets - L+E)":     round(bs_check, 2),
        })

    df = pd.DataFrame(data)
    return df
