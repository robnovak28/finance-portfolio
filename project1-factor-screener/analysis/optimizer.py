"""
optimizer.py
------------
Mean-variance portfolio optimization (Markowitz, 1952).

Computes:
  1. Random portfolio scatter  (Monte Carlo, 3 000 portfolios)
  2. Efficient Frontier        (quadratic programming via scipy)
  3. Minimum Variance Portfolio
  4. Maximum Sharpe Portfolio  (tangency portfolio)
  5. Equal-Weight Benchmark

Constraints:
  - Weights sum to 1
  - Long-only (no short positions)
  - Maximum weight per stock: 15%  (avoids single-stock concentration)

All returns are annualized.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def compute_efficient_frontier(
    prices_df: pd.DataFrame,
    rf_rate: float = 0.05,
    n_points: int = 60,
    n_random: int = 3000,
    max_weight: float = 0.15,
) -> dict | None:
    """
    Parameters
    ----------
    prices_df  : daily closing prices with tickers as columns
    rf_rate    : annual risk-free rate
    n_points   : number of points on the efficient frontier
    n_random   : number of random portfolios for background scatter
    max_weight : maximum weight per stock (0–1)

    Returns
    -------
    dict with keys:
        frontier_df  : DataFrame (vol, return, sharpe)
        min_var      : dict (weights, vol, return, sharpe)
        max_sharpe   : dict (weights, vol, return, sharpe)
        equal_weight : dict (weights, vol, return, sharpe)
        random_df    : DataFrame (vol, return, sharpe)
        tickers      : list of ticker strings
    None if insufficient data.
    """
    ANN = 252

    # Build clean returns matrix
    rets = prices_df.pct_change().dropna(how="any")
    rets = rets.dropna(axis=1)
    tickers = list(rets.columns)
    n = len(tickers)

    if n < 2:
        return None

    mu      = rets.mean().values * ANN
    cov_ann = rets.cov().values * ANN

    def port_stats(w: np.ndarray):
        ret    = float(w @ mu)
        vol    = float(np.sqrt(w @ cov_ann @ w))
        sharpe = (ret - rf_rate) / vol if vol > 0 else 0.0
        return ret, vol, sharpe

    bounds      = [(0.0, max_weight)] * n
    w0          = np.ones(n) / n
    sum_constr  = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    # ── Minimum Variance ──────────────────────────────────────────────────────
    res_mv = minimize(
        lambda w: np.sqrt(w @ cov_ann @ w),
        w0, method="SLSQP", bounds=bounds, constraints=[sum_constr],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    mv_w = res_mv.x if res_mv.success else w0
    mv_ret, mv_vol, mv_sharpe = port_stats(mv_w)

    # ── Maximum Sharpe ────────────────────────────────────────────────────────
    def neg_sharpe(w):
        ret, vol, _ = port_stats(w)
        return -(ret - rf_rate) / vol if vol > 0 else 0.0

    res_ms = minimize(
        neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=[sum_constr],
        options={"maxiter": 500, "ftol": 1e-10},
    )
    ms_w = res_ms.x if res_ms.success else w0
    ms_ret, ms_vol, ms_sharpe = port_stats(ms_w)

    # ── Efficient Frontier ────────────────────────────────────────────────────
    target_rets = np.linspace(mv_ret, min(mu.max() * 0.90, ms_ret * 1.5), n_points)
    frontier = []
    for target in target_rets:
        cons = [
            sum_constr,
            {"type": "eq", "fun": lambda w, t=target: float(w @ mu) - t},
        ]
        res = minimize(
            lambda w: np.sqrt(w @ cov_ann @ w),
            w0, method="SLSQP", bounds=bounds, constraints=cons,
            options={"maxiter": 300, "ftol": 1e-9},
        )
        if res.success:
            r, v, s = port_stats(res.x)
            frontier.append({"return": r, "vol": v, "sharpe": s})
    frontier_df = pd.DataFrame(frontier) if frontier else pd.DataFrame()

    # ── Random Portfolios ─────────────────────────────────────────────────────
    rng  = np.random.default_rng(42)
    rand_rows = []
    for _ in range(n_random):
        w = rng.dirichlet(np.ones(n))
        # Apply max weight constraint (simple clip + renorm)
        w = np.clip(w, 0, max_weight)
        if w.sum() > 0:
            w /= w.sum()
        r, v, s = port_stats(w)
        rand_rows.append({"return": r, "vol": v, "sharpe": s})
    random_df = pd.DataFrame(rand_rows)

    # ── Equal Weight ──────────────────────────────────────────────────────────
    ew_ret, ew_vol, ew_sharpe = port_stats(w0)

    return {
        "frontier_df":  frontier_df,
        "min_var":      {"return": mv_ret, "vol": mv_vol, "sharpe": mv_sharpe,
                         "weights": pd.Series(mv_w, index=tickers)},
        "max_sharpe":   {"return": ms_ret, "vol": ms_vol, "sharpe": ms_sharpe,
                         "weights": pd.Series(ms_w, index=tickers)},
        "equal_weight": {"return": ew_ret, "vol": ew_vol, "sharpe": ew_sharpe,
                         "weights": pd.Series(w0, index=tickers)},
        "random_df":    random_df,
        "tickers":      tickers,
        "n_assets":     n,
    }


def weights_comparison_df(ef_result: dict) -> pd.DataFrame:
    """
    Build a comparison DataFrame of EW vs Min-Var vs Max-Sharpe weights.
    Only includes stocks with non-trivial weight in at least one portfolio.
    """
    tickers = ef_result["tickers"]
    rows = []
    for t in tickers:
        ew  = float(ef_result["equal_weight"]["weights"].get(t, 0))
        mv  = float(ef_result["min_var"]["weights"].get(t, 0))
        ms  = float(ef_result["max_sharpe"]["weights"].get(t, 0))
        if max(ew, mv, ms) > 0.001:
            rows.append({"ticker": t, "ew": ew, "min_var": mv, "max_sharpe": ms})
    df = pd.DataFrame(rows).sort_values("max_sharpe", ascending=False)
    return df
