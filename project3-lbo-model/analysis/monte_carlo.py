"""
monte_carlo.py
--------------
Monte Carlo simulation on two key LBO return drivers:
  1. Revenue CAGR  (normally distributed around base case)
  2. Exit EV/EBITDA multiple (normally distributed around base case)

Optional co-variation:
  - EBITDA margin is lightly correlated with revenue growth
    (higher revenue → better operating leverage → higher margin)

10,000 simulations → distribution of IRR and MOIC.

Returns:
  - Raw DataFrame of simulation results
  - Percentile summary DataFrame
  - Probability of exceeding common hurdles (15%, 20%, 25% IRR; 2x, 3x MOIC)
"""

import numpy as np
import pandas as pd
from dataclasses import replace as dc_replace
from model.assumptions import DealAssumptions, base_case
from model.lbo_engine import run_model


def run_monte_carlo(
    n_sims: int = 10_000,
    seed: int   = 42,
    # Revenue CAGR distribution
    rev_cagr_mean: float | None = None,    # None → use base assumptions
    rev_cagr_std:  float = 0.020,          # +/- ~2% annual
    # Exit multiple distribution
    exit_mult_mean: float | None = None,
    exit_mult_std:  float = 1.50,          # +/- ~1.5 turns
    # EBITDA margin  (correlated with rev growth)
    margin_std: float = 0.010,             # +/- ~1% margin
    base_assumptions: DealAssumptions | None = None,
) -> dict:
    """
    Run Monte Carlo simulation.

    Parameters
    ----------
    n_sims          : number of simulation paths
    seed            : random seed for reproducibility
    rev_cagr_std    : annual std dev of simulated revenue CAGR
    exit_mult_std   : std dev of exit EV/EBITDA multiple
    margin_std      : std dev of EBITDA margin shock applied uniformly
    base_assumptions: base case to perturb (default: base_case())

    Returns
    -------
    {
      "raw_df"       : pd.DataFrame  (n_sims rows, columns = [irr, moic, ...])
      "percentile_df": pd.DataFrame  (percentile summary)
      "probability_df": pd.DataFrame (probability of exceeding hurdles)
    }
    """
    rng = np.random.default_rng(seed)
    base = base_assumptions or base_case()
    n    = base.hold_years

    if rev_cagr_mean is None:
        rev_cagr_mean = float(np.mean(base.revenue_growth))
    if exit_mult_mean is None:
        exit_mult_mean = base.exit_ev_multiple

    # Correlated sampling: revenue growth and EBITDA margin have +0.6 correlation
    corr_matrix = np.array([[1.0, 0.6, 0.0],
                             [0.6, 1.0, 0.0],
                             [0.0, 0.0, 1.0]])
    L = np.linalg.cholesky(corr_matrix)

    # Draw standard normals: shape (n_sims, 3) → [rev_shock, margin_shock, mult_shock]
    z = rng.standard_normal((n_sims, 3))
    correlated = z @ L.T

    rev_shocks    = correlated[:, 0] * rev_cagr_std
    margin_shocks = correlated[:, 1] * margin_std
    mult_shocks   = correlated[:, 2] * exit_mult_std   # exit multiple uncorrelated

    simulated_rev_cagr  = rev_cagr_mean  + rev_shocks
    simulated_exit_mult = exit_mult_mean + mult_shocks
    simulated_margin_adj = margin_shocks   # uniform margin shift applied to all years

    records = []
    for i in range(n_sims):
        cagr     = float(simulated_rev_cagr[i])
        ex_mult  = float(max(4.0, simulated_exit_mult[i]))  # floor at 4x
        m_adj    = float(simulated_margin_adj[i])

        # Build perturbed assumptions
        new_rev_growth = [max(-0.20, cagr + rng.normal(0, 0.005)) for _ in range(n)]
        new_margin     = [max(0.01, base.ebitda_margin[j] + m_adj) for j in range(n)]

        a = DealAssumptions(
            entry_ebitda     = base.entry_ebitda,
            entry_ev_multiple= base.entry_ev_multiple,
            existing_net_debt= base.existing_net_debt,
            existing_cash    = base.existing_cash,
            existing_gross_debt = base.existing_gross_debt,
            entry_revenue    = base.entry_revenue,
            revenue_growth   = new_rev_growth,
            ebitda_margin    = new_margin,
            da_pct_revenue   = base.da_pct_revenue,
            capex_pct_revenue= base.capex_pct_revenue,
            tax_rate         = base.tax_rate,
            exit_ev_multiple = ex_mult,
            hold_years       = n,
            exit_year        = n,           # must match hold_years for any hold period
            debt_tranches    = base.debt_tranches,
            # BS / NWC
            opening_receivables   = base.opening_receivables,
            opening_inventory     = base.opening_inventory,
            opening_other_current = base.opening_other_current,
            opening_ppe           = base.opening_ppe,
            opening_op_lease_rou  = base.opening_op_lease_rou,
            opening_other_lt_assets = base.opening_other_lt_assets,
            opening_ap            = base.opening_ap,
            opening_accrued       = base.opening_accrued,
            opening_current_lease = base.opening_current_lease,
            opening_other_current_liab = base.opening_other_current_liab,
            opening_lt_lease      = base.opening_lt_lease,
            opening_other_lt_liab = base.opening_other_lt_liab,
        )

        try:
            result = run_model(a)
            r = result["returns"]
            irr  = r["irr"]
            moic = r["moic"]
            if np.isnan(irr) or irr < -0.99 or irr > 5.0:
                continue
            records.append({
                "IRR":              irr,
                "MOIC":             moic,
                "Exit EV/EBITDA":   ex_mult,
                "Rev CAGR":         float(np.mean(new_rev_growth)),
                "Exit EBITDA Margin": new_margin[-1],
                "Exit Equity ($M)": r["exit_equity"],
            })
        except Exception:
            continue

    raw_df = pd.DataFrame(records)
    if raw_df.empty:
        raise RuntimeError(
            f"All {n_sims} simulation paths failed — likely a model configuration error "
            f"(e.g. hold_years={n}, exit_year mismatch, or extreme assumptions)."
        )

    # ---- Percentile Summary ----
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    perc_rows = []
    for p in percentiles:
        perc_rows.append({
            "Percentile":       f"{p}th",
            "IRR":              f"{np.percentile(raw_df['IRR'], p):.1%}",
            "MOIC":             f"{np.percentile(raw_df['MOIC'], p):.2f}x",
            "Exit EV/EBITDA":   f"{np.percentile(raw_df['Exit EV/EBITDA'], p):.1f}x",
            "Exit EBITDA Margin": f"{np.percentile(raw_df['Exit EBITDA Margin'], p):.1%}",
        })
    percentile_df = pd.DataFrame(perc_rows)

    # ---- Probability Table ----
    prob_rows = [
        {"Hurdle":   "IRR > 15%",  "Probability": f"{(raw_df['IRR'] > 0.15).mean():.1%}"},
        {"Hurdle":   "IRR > 20%",  "Probability": f"{(raw_df['IRR'] > 0.20).mean():.1%}"},
        {"Hurdle":   "IRR > 25%",  "Probability": f"{(raw_df['IRR'] > 0.25).mean():.1%}"},
        {"Hurdle":   "IRR > 30%",  "Probability": f"{(raw_df['IRR'] > 0.30).mean():.1%}"},
        {"Hurdle":   "MOIC > 2.0x","Probability": f"{(raw_df['MOIC'] > 2.0).mean():.1%}"},
        {"Hurdle":   "MOIC > 2.5x","Probability": f"{(raw_df['MOIC'] > 2.5).mean():.1%}"},
        {"Hurdle":   "MOIC > 3.0x","Probability": f"{(raw_df['MOIC'] > 3.0).mean():.1%}"},
        {"Hurdle":   "IRR < 0%",   "Probability": f"{(raw_df['IRR'] < 0.0).mean():.1%}"},
    ]
    probability_df = pd.DataFrame(prob_rows)

    return {
        "raw_df":        raw_df,
        "percentile_df": percentile_df,
        "probability_df":probability_df,
        "n_valid_sims":  len(raw_df),
    }
