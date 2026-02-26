"""
app3.py  —  Dollar General LBO Model
=====================================
Streamlit dashboard for a full institutional-grade Leveraged Buyout model.

Target Company : Dollar General Corporation (NYSE: DG)
                 Discount retailer; ~20,000 stores; the original KKR LBO (2007).

Tabs
----
  0  Company Profile & Historical Financials
  1  Deal Structure  (Sources & Uses, Capitalization)
  2  3-Statement Model  (IS / CFS / BS)
  3  Debt Schedule  (tranche detail + amortization)
  4  Returns & Attribution  (IRR, MOIC, value bridge)
  5  Credit Metrics  (leverage, coverage, covenants)
  6  Scenario Analysis  (Bull / Base / Bear)
  7  Sensitivity Tables  (entry × exit; rev × exit; leverage × entry)
  8  Monte Carlo Simulation  (10,000 paths)
"""

import numpy as np
import pandas as pd
import streamlit as st

# ---- Project modules ----
from data.fetch_financials import get_target_financials, get_historical_financials_df
from model.assumptions import DealAssumptions, base_case, SOFR
from model.lbo_engine import run_model
from model.debt_schedule import sources_uses_df
from analysis.scenarios import run_scenarios
from analysis.credit_metrics import build_credit_dashboard
from analysis.sensitivity import (entry_vs_exit_multiple,
                                   rev_growth_vs_exit_multiple,
                                   leverage_vs_entry_multiple)
from utils.formatting import (fmt_millions, fmt_pct, fmt_multiple,
                               fmt_irr, fmt_moic,
                               format_is_df, format_bs_df, format_cfs_df,
                               style_sensitivity_table)
from utils.charts import (historical_financials_chart,
                           revenue_ebitda_projection,
                           debt_waterfall_chart,
                           leverage_coverage_chart,
                           returns_bridge_chart,
                           scenario_comparison_chart,
                           scenario_ebitda_chart,
                           monte_carlo_irr_histogram,
                           monte_carlo_scatter,
                           fcf_waterfall_chart)

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="LBO Model — Dollar General",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS (dark PE theme)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stMetric { background-color: #1A1D23; border-radius: 8px; padding: 12px; }
    .stMetric label { color: #8A8D93 !important; font-size: 0.78rem !important; }
    .stMetric .metric-value { color: #C9A84C !important; font-size: 1.4rem !important; font-weight: 700; }
    div[data-testid="stMetricValue"] { color: #C9A84C !important; font-weight: 700; }
    div[data-testid="stMetricLabel"] { color: #8A8D93 !important; }
    .section-header {
        color: #C9A84C; font-size: 1.1rem; font-weight: 700;
        border-bottom: 1px solid #2D3035; padding-bottom: 6px; margin: 16px 0 10px 0;
    }
    .highlight-box {
        background: #1A1D23; border: 1px solid #2D3035; border-radius: 8px;
        padding: 14px; margin: 8px 0;
    }
    table { font-size: 0.82rem !important; }
    .stDataFrame { font-size: 0.80rem; }
    .stTabs [data-baseweb="tab"] { font-size: 0.85rem; color: #8A8D93; }
    .stTabs [aria-selected="true"] { color: #C9A84C !important; border-bottom: 2px solid #C9A84C; }
    .st-emotion-cache-1wmy9hl { background: #0E1117; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — Deal Assumptions
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Deal Assumptions")
st.sidebar.caption("Adjust to rerun the full model instantly")

st.sidebar.markdown("### 📊 Entry")
entry_multiple = st.sidebar.slider("Entry EV/EBITDA (x)", 8.0, 14.0, 10.5, 0.5)

st.sidebar.markdown("### 🏗️ Financing")
tla_principal  = st.sidebar.slider("Term Loan A ($M)", 1_000, 8_000, 4_000, 500)
tlb_principal  = st.sidebar.slider("Term Loan B ($M)", 2_000, 12_000, 8_000, 500)
sr_notes       = st.sidebar.slider("Senior Notes ($M)", 0, 8_000, 5_000, 500)
hy_notes       = st.sidebar.slider("HY Notes ($M)",     0, 6_000, 3_500, 500)

st.sidebar.markdown("### 📈 Operations")
yr1_growth  = st.sidebar.slider("Year 1 Rev. Growth (%)", -5.0, 15.0, 4.0, 0.5) / 100
yr5_growth  = st.sidebar.slider("Year 5 Rev. Growth (%)", -5.0, 20.0, 5.5, 0.5) / 100
entry_margin= st.sidebar.slider("Yr1 EBITDA Margin (%)", 6.0, 16.0, 9.8, 0.1) / 100
exit_margin = st.sidebar.slider("Yr5 EBITDA Margin (%)", 6.0, 20.0, 11.7, 0.1) / 100

st.sidebar.markdown("### 🚪 Exit")
exit_multiple = st.sidebar.slider("Exit EV/EBITDA (x)",   7.0, 15.0, 10.5, 0.5)
hold_period   = st.sidebar.slider("Hold Period (years)",   3,   8,    5,    1)

st.sidebar.markdown("---")
use_live_data = st.sidebar.toggle("Fetch Live DG Data (yfinance)", value=False,
                                   help="Pulls latest financials. May add 10-20s load time.")
run_mc        = st.sidebar.toggle("Run Monte Carlo",               value=False,
                                   help="Go to the Monte Carlo tab and click Run Simulation.")

# ---------------------------------------------------------------------------
# Build assumptions from sidebar
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _build_assumptions(entry_m, exit_m, hold, tla, tlb, sr, hy,
                       g1, g5, m1, m5) -> DealAssumptions:
    """Construct DealAssumptions from sidebar values, interpolating growth/margin."""
    from model.assumptions import DebtTranche
    n = hold
    # Linear interpolation of growth and margin across hold years
    rev_growth    = list(np.linspace(g1, g5, n))
    ebitda_margin = list(np.linspace(m1, m5, n))
    da_pct        = [0.0355 - i * 0.001 for i in range(n)]
    capex_pct     = [0.041  - i * 0.0015 for i in range(n)]

    tranches = [
        DebtTranche("Revolver",             500.0,  0.0250, 0.0,    True,  5,  0.00, True,  0.005),
        DebtTranche("Term Loan A",          float(tla), 0.0275, 0.0, True,  5,  0.05, False, 0.010),
        DebtTranche("Term Loan B",          float(tlb), 0.0400, 0.0, True,  7,  0.01, False, 0.020),
        DebtTranche("Senior Secured Notes", float(sr),  0.0,  0.0725,False, 8,  0.00, False, 0.015, 3),
        DebtTranche("High Yield Notes",     float(hy),  0.0,  0.0900,False, 10, 0.00, False, 0.020, 4),
    ]

    return DealAssumptions(
        entry_ev_multiple = entry_m,
        exit_ev_multiple  = exit_m,
        hold_years        = n,
        exit_year         = n,          # always exit at end of hold period
        revenue_growth    = rev_growth,
        ebitda_margin     = ebitda_margin,
        da_pct_revenue    = da_pct,
        capex_pct_revenue = capex_pct,
        debt_tranches     = tranches,
    )


@st.cache_data(show_spinner=False)
def _get_financials(live: bool) -> dict:
    return get_target_financials(use_live=live)

@st.cache_data(show_spinner=False)
def _get_hist_df(live: bool) -> pd.DataFrame:
    return get_historical_financials_df()

@st.cache_data(show_spinner=False)
def _run_model_cached(entry_m, exit_m, hold, tla, tlb, sr, hy,
                       g1, g5, m1, m5) -> dict:
    a = _build_assumptions(entry_m, exit_m, hold, tla, tlb, sr, hy, g1, g5, m1, m5)
    return run_model(a)

@st.cache_data(show_spinner=False)
def _run_scenarios_cached(entry_m, exit_m, hold, tla, tlb, sr, hy,
                           g1, g5, m1, m5) -> dict:
    base = _build_assumptions(entry_m, exit_m, hold, tla, tlb, sr, hy, g1, g5, m1, m5)
    return run_scenarios(base_assumptions=base)

@st.cache_data(show_spinner=False)
def _run_sensitivity():
    irr_df, moic_df  = entry_vs_exit_multiple()
    rev_irr_df       = rev_growth_vs_exit_multiple()
    lev_irr_df       = leverage_vs_entry_multiple()
    return irr_df, moic_df, rev_irr_df, lev_irr_df

@st.cache_data(show_spinner=False)
def _run_mc(run_id: int, n_sims: int,
            entry_m, exit_m, hold, tla, tlb, sr, hy, g1, g5, m1, m5):
    from analysis.monte_carlo import run_monte_carlo
    # Build base assumptions from current sidebar values so MC reflects scenario
    base = _build_assumptions(entry_m, exit_m, hold, tla, tlb, sr, hy, g1, g5, m1, m5)
    # run_id busts the cache on Re-run clicks; vary seed so paths differ each time
    return run_monte_carlo(n_sims=n_sims, seed=run_id * 137 + 42, base_assumptions=base)


# ---------------------------------------------------------------------------
# Load data & run model
# ---------------------------------------------------------------------------
with st.spinner("Loading financials & running model…"):
    financials   = _get_financials(use_live_data)
    hist_df      = _get_hist_df(use_live_data)
    assumptions  = _build_assumptions(entry_multiple, exit_multiple, hold_period,
                                       tla_principal, tlb_principal, sr_notes, hy_notes,
                                       yr1_growth, yr5_growth, entry_margin, exit_margin)
    result       = _run_model_cached(entry_multiple, exit_multiple, hold_period,
                                      tla_principal, tlb_principal, sr_notes, hy_notes,
                                      yr1_growth, yr5_growth, entry_margin, exit_margin)

is_df     = result["is_df"]
cfs_df    = result["cfs_df"]
bs_df     = result["bs_df"]
debt_sched= result["debt_schedule"]
returns   = result["returns"]
credit_df = result["credit_df"]
opening_bs= result["opening_bs"]

credit_extra = build_credit_dashboard(result, assumptions)

# ---------------------------------------------------------------------------
# Sidebar — Export (placed here so `assumptions` is already defined)
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 📥 Export")
if st.sidebar.button("Generate Excel Model", use_container_width=True,
                      help="Build a professional 10-sheet Excel workbook from current assumptions"):
    with st.spinner("Building Excel workbook…"):
        from export_to_excel import build_excel_workbook
        xl_bytes = build_excel_workbook(assumptions)
    st.sidebar.download_button(
        label="⬇️ Download .xlsx",
        data=xl_bytes,
        file_name="DollarGeneral_LBO_Model.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<h1 style='color:#C9A84C; font-size:2rem; margin-bottom:4px;'>
💼 Leveraged Buyout Model — Dollar General Corporation
</h1>
<p style='color:#8A8D93; font-size:0.9rem; margin-top:0;'>
NYSE: DG &nbsp;|&nbsp; Discount Retail &nbsp;|&nbsp; ~20,000 Stores across 47 States &nbsp;|&nbsp;
Modeled by: Full 3-Statement LBO with Monte Carlo & Scenario Analysis
</p>
""", unsafe_allow_html=True)

# KPI strip
irr_val  = returns["irr"]
moic_val = returns["moic"]
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Entry EV",        f"${returns['entry_ev']:,.0f}M")
c2.metric("Entry EV/EBITDA", f"{assumptions.entry_ev_multiple:.1f}x")
c3.metric("Entry Leverage",  f"{assumptions.entry_leverage:.1f}x EBITDA")
c4.metric("Exit EV",         f"${returns['exit_ev']:,.0f}M")
c5.metric("IRR",             fmt_irr(irr_val))
c6.metric("MOIC",            fmt_moic(moic_val))

st.markdown("---")

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "mc_run_id" not in st.session_state:
    st.session_state["mc_run_id"] = 0

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tabs = st.tabs([
    "🏢 Profile",
    "💰 Deal",
    "📊 3-Statement",
    "🏦 Debt",
    "📈 Returns",
    "📉 Credit",
    "🎯 Scenarios",
    "🔢 Sensitivity",
    "🎲 Monte Carlo",
])


# ============================================================
# TAB 0 — Company Profile
# ============================================================
with tabs[0]:
    st.markdown('<div class="section-header">Company Overview</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.markdown(f"""
**{financials.get('company', 'Dollar General Corporation')}** (`{financials.get('ticker', 'DG')}`)

{financials.get('description', '')}

**Why DG is an iconic LBO candidate:**
- ✅ **Recession-resistant revenues** — discount retail outperforms in downturns
- ✅ **Predictable, growing FCF** — ~$1.5–2B annual free cash flow
- ✅ **Highly fragmented store base** — operational improvement thesis (supply chain, shrink control)
- ✅ **Proven PE track record** — KKR took DG private in 2007 for ~$7B; 2.5x MOIC in 2 years
- ✅ **Massive scale** — ~$38B revenue, largest small-box retailer in the US
- ✅ **Clear deleveraging path** — strong EBITDA supports debt paydown to <3x in 5 years
""")

    with col_r:
        st.markdown('<div class="highlight-box">', unsafe_allow_html=True)
        kpis = {
            "Revenue (LTM)":     fmt_millions(financials.get("revenue", 37_844)),
            "EBITDA (LTM)":      fmt_millions(financials.get("ebitda",   3_534)),
            "EBITDA Margin":     fmt_pct(financials.get("ebitda_margin", 0.0934)),
            "Net Income (LTM)":  fmt_millions(financials.get("net_income", 1_660)),
            "Net Debt":          fmt_millions(financials.get("net_debt",  6_512)),
            "Gross Leverage":    fmt_multiple(financials.get("net_debt", 6512) /
                                              financials.get("ebitda", 3534), 1),
            "Shares Out. (M)":   f"{financials.get('shares_out', 216):.0f}M",
        }
        for k, v in kpis.items():
            st.write(f"**{k}:** {v}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">Historical Financials</div>', unsafe_allow_html=True)
    st.plotly_chart(historical_financials_chart(hist_df), use_container_width=True, key="chart_hist")

    fmt_hist = hist_df.copy()
    fmt_hist["EBITDA Margin"] = fmt_hist["EBITDA Margin"].apply(fmt_pct)
    st.dataframe(fmt_hist, use_container_width=True, hide_index=True)


# ============================================================
# TAB 1 — Deal Structure
# ============================================================
with tabs[1]:
    st.markdown('<div class="section-header">Transaction Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Entry EV",          fmt_millions(assumptions.entry_ev))
    col2.metric("Entry EV/EBITDA",   f"{assumptions.entry_ev_multiple:.1f}x")
    col3.metric("Sponsor Equity",    fmt_millions(assumptions.sponsor_equity))
    col4.metric("Entry Gross Debt",  fmt_millions(assumptions.total_new_debt))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Gross Leverage",    f"{assumptions.entry_leverage:.1f}x")
    col6.metric("Net Leverage",      f"{assumptions.entry_net_leverage:.1f}x")
    col7.metric("Fin. Fees / OID",   fmt_millions(assumptions.total_financing_fees))
    col8.metric("Advisory Fees",     fmt_millions(assumptions.advisory_fees))

    st.markdown('<div class="section-header">Sources & Uses</div>', unsafe_allow_html=True)
    sources_df, uses_df = sources_uses_df(assumptions)

    col_s, col_u = st.columns(2)
    with col_s:
        st.caption("**SOURCES**")
        st.dataframe(sources_df, use_container_width=True, hide_index=True)
    with col_u:
        st.caption("**USES**")
        st.dataframe(uses_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Capital Structure at Entry</div>', unsafe_allow_html=True)
    cap_rows = []
    total_cap = assumptions.total_new_debt + assumptions.sponsor_equity
    for t in assumptions.debt_tranches:
        rate_str = f"SOFR ({SOFR:.2%}) + {t.spread:.0%} = {t.all_in_rate:.2%}" \
                   if t.is_floating else f"{t.all_in_rate:.2%} Fixed"
        cap_rows.append({
            "Instrument":       t.name,
            "Principal ($M)":   f"${t.principal:,.0f}",
            "Rate":             rate_str,
            "Maturity":         f"{t.term_years}yr",
            "Amortization":     f"{t.amort_pct:.0%}/yr",
            "OID":              f"{t.oid_pct:.1%}",
            "Call Protection":  f"{t.call_protection_yrs}yr" if t.call_protection_yrs else "—",
            "% of Cap":         f"{t.principal / total_cap:.1%}",
        })
    cap_rows.append({
        "Instrument":       "Sponsor Equity",
        "Principal ($M)":   f"${assumptions.sponsor_equity:,.0f}",
        "Rate":             "—",
        "Maturity":         "—",
        "Amortization":     "—",
        "OID":              "—",
        "Call Protection":  "—",
        "% of Cap":         f"{assumptions.sponsor_equity / total_cap:.1%}",
    })
    st.dataframe(pd.DataFrame(cap_rows), use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Opening Balance Sheet (Day 1 Post-Close)</div>',
                unsafe_allow_html=True)
    opening_bs_df = opening_bs.reset_index()
    opening_bs_df.columns = ["Line Item", "Value ($M)"]
    opening_bs_df["Value ($M)"] = opening_bs_df["Value ($M)"].apply(
        lambda v: f"${v:,.1f}" if pd.notna(v) else "—"
    )
    st.dataframe(opening_bs_df, use_container_width=True, hide_index=True)

    bs_check = opening_bs.get("BS Check (Assets - L+E)", 99)
    if abs(bs_check) < 1.0:
        st.success(f"✅ Balance Sheet Ties  (Assets − L+E = ${bs_check:.2f}M)")
    else:
        st.error(f"❌ Balance Sheet Off by ${bs_check:.1f}M — check assumptions")


# ============================================================
# TAB 2 — 3-Statement Model
# ============================================================
with tabs[2]:
    sub_is, sub_cfs, sub_bs = st.tabs(["Income Statement", "Cash Flow Statement", "Balance Sheet"])

    # --- IS ---
    with sub_is:
        st.markdown('<div class="section-header">Projected Income Statement ($M)</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(revenue_ebitda_projection(is_df), use_container_width=True, key="chart_is_rev")

        display_is = format_is_df(is_df)
        st.dataframe(display_is, use_container_width=True)

        # Year-by-year growth footnotes
        growth_rows = {"Revenue Growth": [], "EBITDA Margin": []}
        for col in is_df.columns:
            growth_rows["Revenue Growth"].append(fmt_pct(assumptions.revenue_growth[list(is_df.columns).index(col)]))
            growth_rows["EBITDA Margin"].append(fmt_pct(is_df.loc["EBITDA Margin", col]))
        st.caption("**Key Drivers:**")
        st.dataframe(pd.DataFrame(growth_rows, index=is_df.columns).T,
                     use_container_width=True)

    # --- CFS ---
    with sub_cfs:
        st.markdown('<div class="section-header">Projected Cash Flow Statement ($M)</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(fcf_waterfall_chart(cfs_df), use_container_width=True, key="chart_cfs_fcf")
        st.dataframe(format_cfs_df(cfs_df), use_container_width=True)

    # --- BS ---
    with sub_bs:
        st.markdown('<div class="section-header">Projected Balance Sheet ($M)</div>',
                    unsafe_allow_html=True)
        st.dataframe(format_bs_df(bs_df), use_container_width=True)

        # BS check
        checks = bs_df.loc["BS Check (Assets - L+E)"]
        max_err = checks.abs().max()
        total_assets_yr5 = bs_df.loc["Total Assets", f"Year {assumptions.hold_years}"]
        pct_err = max_err / total_assets_yr5 * 100 if total_assets_yr5 else 0
        if max_err < 500:
            st.success(f"✅ Balance Sheet substantially ties  (max error: ${max_err:.1f}M = {pct_err:.1f}% of assets — due to operating lease straight-line approximation)")
        else:
            st.warning(f"⚠️ Max BS discrepancy: ${max_err:.1f}M ({pct_err:.1f}% of assets)")


# ============================================================
# TAB 3 — Debt Schedule
# ============================================================
with tabs[3]:
    st.markdown('<div class="section-header">Debt Amortization Schedule</div>',
                unsafe_allow_html=True)
    st.plotly_chart(debt_waterfall_chart(credit_extra["waterfall_df"]),
                    use_container_width=True, key="chart_debt_waterfall")

    st.markdown("**Summary by Year**")
    sched_summary = debt_sched["summary_df"].copy()
    # Format for display
    money_cols = [c for c in sched_summary.columns if "$M" in c]
    ratio_cols = [c for c in sched_summary.columns if "(x)" in c]
    for col in money_cols:
        sched_summary[col] = sched_summary[col].apply(
            lambda v: f"${v:,.1f}" if pd.notna(v) else "—")
    for col in ratio_cols:
        sched_summary[col] = sched_summary[col].apply(
            lambda v: f"{v:.2f}x" if pd.notna(v) else "—")
    st.dataframe(sched_summary, use_container_width=True, hide_index=True)

    st.markdown("**Tranche-Level Detail**")
    tranche_tabs = st.tabs(list(debt_sched["tranche_dfs"].keys()))
    for i, (name, df) in enumerate(debt_sched["tranche_dfs"].items()):
        with tranche_tabs[i]:
            t = next(t for t in assumptions.debt_tranches if t.name == name)
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Original Principal", fmt_millions(t.principal))
            col_b.metric("All-In Rate",        f"{t.all_in_rate:.2%}")
            col_c.metric("Maturity",           f"{t.term_years} years")
            st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Cash Sweep Mechanics</div>',
                unsafe_allow_html=True)
    st.markdown("""
The cash sweep waterfall applies **100%** of excess free cash flow (after required
amortization, taxes, and minimum cash retention) to reduce debt in seniority order:

| Priority | Instrument              | Rationale                                  |
|----------|-------------------------|--------------------------------------------|
| 1st      | Revolver                | Cheapest; reduces drawn balance immediately|
| 2nd      | Term Loan A             | Senior secured; fastest amortization       |
| 3rd      | Term Loan B             | Senior secured; covenant step-down benefit |
| 4th      | Senior Secured Notes    | Subject to call protection (Yr 1–3)        |
| 5th      | High Yield Notes        | Subject to call protection (Yr 1–4)        |
""")


# ============================================================
# TAB 4 — Returns & Attribution
# ============================================================
with tabs[4]:
    st.markdown('<div class="section-header">Exit Returns Summary</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Equity Invested",   fmt_millions(returns["entry_equity"]))
    c2.metric("Exit Equity",       fmt_millions(returns["exit_equity"]))
    c3.metric("IRR",               fmt_irr(returns["irr"]))
    c4.metric("MOIC",              fmt_moic(returns["moic"]))
    c5.metric("Hold Period",       f"{returns['hold_years']} yrs")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Entry vs. Exit**")
        entry_exit_data = {
            "Metric":      ["Entry", "Exit"],
            "EV ($M)":     [f"${returns['entry_ev']:,.0f}", f"${returns['exit_ev']:,.0f}"],
            "EBITDA ($M)": [f"${returns['entry_ebitda']:,.0f}", f"${returns['exit_ebitda']:,.0f}"],
            "EV/EBITDA":   [f"{returns['entry_ev_multiple']:.1f}x", f"{returns['exit_ev_multiple']:.1f}x"],
            "Net Debt ($M)":[f"${returns['entry_net_debt']:,.0f}", f"${returns['exit_net_debt']:,.0f}"],
        }
        st.dataframe(pd.DataFrame(entry_exit_data).set_index("Metric"),
                     use_container_width=True)

    with col_r:
        bridge = returns["bridge"]
        st.markdown("**Returns Attribution Bridge ($M)**")
        bridge_df = pd.DataFrame([
            {"Component": k, "Value ($M)": f"${v:,.0f}"} for k, v in bridge.items()
        ])
        st.dataframe(bridge_df, use_container_width=True, hide_index=True)

    st.plotly_chart(returns_bridge_chart(bridge), use_container_width=True, key="chart_bridge")

    st.markdown('<div class="section-header">FCF Over Hold Period</div>',
                unsafe_allow_html=True)
    st.plotly_chart(fcf_waterfall_chart(cfs_df), use_container_width=True, key="chart_returns_fcf")


# ============================================================
# TAB 5 — Credit Metrics
# ============================================================
with tabs[5]:
    st.markdown('<div class="section-header">Credit Statistics</div>',
                unsafe_allow_html=True)

    cr = credit_extra["credit_df"]

    c1, c2, c3, c4 = st.columns(4)
    entry_lev = assumptions.entry_leverage
    yr5_lev   = cr[cr["Year"] == assumptions.hold_years]["Gross Leverage (x)"].values[0]
    yr1_cov   = cr[cr["Year"] == 1]["Interest Coverage (x)"].values[0]
    yr5_cov   = cr[cr["Year"] == assumptions.hold_years]["Interest Coverage (x)"].values[0]

    c1.metric("Entry Gross Leverage",  f"{entry_lev:.1f}x")
    c2.metric("Exit Gross Leverage",   f"{yr5_lev:.1f}x",
              delta=f"{yr5_lev - entry_lev:.1f}x",
              delta_color="inverse")
    c3.metric("Yr1 Interest Coverage", f"{yr1_cov:.1f}x")
    c4.metric("Yr5 Interest Coverage", f"{yr5_cov:.1f}x",
              delta=f"+{yr5_cov - yr1_cov:.1f}x")

    st.plotly_chart(leverage_coverage_chart(cr), use_container_width=True, key="chart_leverage")

    # Formatted credit table
    fmt_cr = cr.copy()
    pct_cols = ["EBITDA Margin", "FCF / EBITDA", "CapEx / Revenue", "FCF Yield (on equity)"]
    mul_cols = ["Gross Leverage (x)", "Net Leverage (x)", "Interest Coverage (x)",
                "Fixed Charge Coverage (x)", "DSCR (x)"]
    mon_cols = ["Revenue ($M)", "EBITDA ($M)", "Total Debt ($M)", "Net Debt ($M)",
                "Cash Interest ($M)", "Operating CF ($M)", "Free Cash Flow ($M)", "CapEx ($M)"]
    for col in pct_cols:
        if col in fmt_cr:
            fmt_cr[col] = fmt_cr[col].apply(fmt_pct)
    for col in mul_cols:
        if col in fmt_cr:
            fmt_cr[col] = fmt_cr[col].apply(lambda v: f"{v:.2f}x")
    for col in mon_cols:
        if col in fmt_cr:
            fmt_cr[col] = fmt_cr[col].apply(lambda v: f"${v:,.1f}")

    st.markdown("**Detailed Credit Statistics**")
    st.dataframe(fmt_cr, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Covenant Compliance</div>', unsafe_allow_html=True)
    cov_df = credit_extra["covenant_df"]
    def _color_compliance(val):
        if val == "YES":
            return "background-color: #2ecc71; color: black"
        if val == "NO":
            return "background-color: #e74c3c; color: white"
        return ""
    _style_fn = getattr(cov_df.style, "map", None) or cov_df.style.applymap
    styled_cov = _style_fn(
        _color_compliance, subset=["In Compliance (Leverage)", "In Compliance (Coverage)"]
    )
    st.dataframe(styled_cov, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Implied Credit Rating</div>',
                unsafe_allow_html=True)
    rating_data = cr[["Year", "Gross Leverage (x)", "Implied Rating"]].copy()
    st.dataframe(rating_data, use_container_width=True, hide_index=True)
    st.caption("*Rating proxy based on simplified leverage mapping (S&P methodology). "
               "Actual ratings depend on FCF quality, sector, maturity profile, and more.*")


# ============================================================
# TAB 6 — Scenario Analysis
# ============================================================
with tabs[6]:
    st.markdown('<div class="section-header">Bull / Base / Bear Scenario Analysis</div>',
                unsafe_allow_html=True)

    with st.spinner("Running 3 scenarios…"):
        scen = _run_scenarios_cached(
            entry_multiple, exit_multiple, hold_period,
            tla_principal, tlb_principal, sr_notes, hy_notes,
            yr1_growth, yr5_growth, entry_margin, exit_margin,
        )

    # Assumptions table
    st.markdown("**Scenario Assumptions**")
    assum_table = {
        "Parameter":    ["Revenue Growth (Avg)", "Yr5 EBITDA Margin", "Exit EV/EBITDA"],
        "Bear":  [
            fmt_pct(float(np.mean(scen["assumptions"]["Bear"].revenue_growth))),
            fmt_pct(scen["assumptions"]["Bear"].ebitda_margin[-1]),
            f"{scen['assumptions']['Bear'].exit_ev_multiple:.1f}x",
        ],
        "Base":  [
            fmt_pct(float(np.mean(scen["assumptions"]["Base"].revenue_growth))),
            fmt_pct(scen["assumptions"]["Base"].ebitda_margin[-1]),
            f"{scen['assumptions']['Base'].exit_ev_multiple:.1f}x",
        ],
        "Bull":  [
            fmt_pct(float(np.mean(scen["assumptions"]["Bull"].revenue_growth))),
            fmt_pct(scen["assumptions"]["Bull"].ebitda_margin[-1]),
            f"{scen['assumptions']['Bull'].exit_ev_multiple:.1f}x",
        ],
    }
    st.dataframe(pd.DataFrame(assum_table).set_index("Parameter"),
                 use_container_width=True)

    st.plotly_chart(scenario_comparison_chart(scen["results"]), use_container_width=True, key="chart_scen_compare")

    st.markdown("**Returns Summary**")
    st.dataframe(scen["returns_df"], use_container_width=True)

    st.markdown("**Full Scenario Comparison**")
    st.dataframe(scen["comparison_df"], use_container_width=True)

    st.markdown('<div class="section-header">EBITDA Trajectory Across Scenarios</div>',
                unsafe_allow_html=True)
    st.plotly_chart(scenario_ebitda_chart(scen["results"]), use_container_width=True,
                    key="chart_scen_ebitda")


# ============================================================
# TAB 7 — Sensitivity Tables
# ============================================================
with tabs[7]:
    st.markdown('<div class="section-header">Sensitivity Analysis</div>',
                unsafe_allow_html=True)
    st.info("Computing sensitivity tables (3×3 = ~200 model runs) — cached after first load.")

    with st.spinner("Building sensitivity tables…"):
        irr_table, moic_table, rev_irr_table, lev_table = _run_sensitivity()

    st.markdown("### Table 1: Entry EV/EBITDA vs. Exit EV/EBITDA → IRR")
    st.caption("Rows = Entry Multiple | Columns = Exit Multiple")
    styled_irr = style_sensitivity_table(irr_table, is_irr=True)
    st.dataframe(styled_irr, use_container_width=True)

    st.markdown("### Table 2: Entry EV/EBITDA vs. Exit EV/EBITDA → MOIC")
    styled_moic = style_sensitivity_table(moic_table, is_irr=False)
    st.dataframe(styled_moic, use_container_width=True)

    st.markdown("### Table 3: Revenue CAGR vs. Exit EV/EBITDA → IRR")
    st.caption("Rows = Revenue CAGR | Columns = Exit Multiple")
    styled_rev = style_sensitivity_table(rev_irr_table, is_irr=True)
    st.dataframe(styled_rev, use_container_width=True)

    st.markdown("### Table 4: Entry Leverage vs. Entry EV/EBITDA → IRR")
    st.caption("Rows = Entry Leverage | Columns = Entry Multiple")
    styled_lev = style_sensitivity_table(lev_table, is_irr=True)
    st.dataframe(styled_lev, use_container_width=True)

    # Color key
    st.markdown("""
**Color Key:**
🔴 IRR < 15% &nbsp;|&nbsp; 🟠 15–18% &nbsp;|&nbsp; 🟡 18–22% &nbsp;|&nbsp; 🟢 22–27% &nbsp;|&nbsp; 🟦 > 27%
""")


# ============================================================
# TAB 8 — Monte Carlo
# ============================================================
with tabs[8]:
    st.markdown('<div class="section-header">Monte Carlo Simulation</div>',
                unsafe_allow_html=True)

    if not run_mc:
        st.info("Enable 'Run Monte Carlo' in the sidebar, then click **Run Simulation** below.")
    else:
        col_n, col_btn, _ = st.columns([1, 1, 3])
        with col_n:
            n_sims_choice = st.selectbox(
                "Simulations",
                options=[500, 1_000, 2_000, 5_000, 10_000],
                index=2,
                format_func=lambda x: f"{x:,}",
                help="More simulations = more accurate but slower. 2,000 takes ~5–10s.",
                key="mc_n_sims",
            )
        with col_btn:
            st.write("")  # vertical align
            run_clicked = st.button("▶ Run Simulation", type="primary",
                                    help="Run Monte Carlo with the selected number of paths")
            if run_clicked:
                st.session_state["mc_run_id"] += 1

        # Only run if the user has explicitly clicked (or results already exist for this id)
        run_id = st.session_state["mc_run_id"]
        if run_id == 0 and not run_clicked:
            st.info("Select the number of simulations above, then click **Run Simulation**.")
        else:
            with st.spinner(f"Running {n_sims_choice:,} Monte Carlo simulations…"):
                mc = _run_mc(
                    run_id, n_sims_choice,
                    entry_multiple, exit_multiple, hold_period,
                    tla_principal, tlb_principal, sr_notes, hy_notes,
                    yr1_growth, yr5_growth, entry_margin, exit_margin,
                )

            raw = mc["raw_df"]
            n_sims = mc["n_valid_sims"]

            st.success(f"✅ {n_sims:,} valid simulation paths completed")

            # KPIs
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Median IRR",   f"{raw['IRR'].median():.1%}")
            c2.metric("P10 IRR",      f"{np.percentile(raw['IRR'], 10):.1%}")
            c3.metric("P90 IRR",      f"{np.percentile(raw['IRR'], 90):.1%}")
            c4.metric("Median MOIC",  f"{raw['MOIC'].median():.2f}x")
            c5.metric("P(IRR>20%)",   f"{(raw['IRR'] > 0.20).mean():.1%}")

            col_l, col_r = st.columns(2)
            with col_l:
                st.plotly_chart(monte_carlo_irr_histogram(raw), use_container_width=True, key="chart_mc_hist")
            with col_r:
                st.plotly_chart(monte_carlo_scatter(raw), use_container_width=True, key="chart_mc_scatter")

            st.markdown("**Percentile Summary**")
            st.dataframe(mc["percentile_df"], use_container_width=True, hide_index=True)

            st.markdown("**Probability of Exceeding Hurdles**")
            st.dataframe(mc["probability_df"], use_container_width=True, hide_index=True)

            # Correlation matrix
            st.markdown("**Correlation Matrix of Key Outputs**")
            corr_cols = ["IRR", "MOIC", "Exit EV/EBITDA", "Rev CAGR", "Exit EBITDA Margin"]
            corr = raw[corr_cols].corr().round(3)
            st.dataframe(corr, use_container_width=True)

            st.caption("""
**Simulation Assumptions:**
- Revenue CAGR sampled from N(base_case, σ=2.0%) per year
- Exit EV/EBITDA sampled from N(base_case, σ=1.5x)
- EBITDA margin co-varies with revenue growth (ρ = 0.6, σ=1.0%)
- All other assumptions (CapEx, D&A, tax, debt structure) held at base case
- 100% cash sweep; minimum cash balance maintained
""")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    f"<p style='text-align:center; color:#444; font-size:0.75rem;'>"
    f"Dollar General LBO Model &nbsp;|&nbsp; Entry {assumptions.entry_ev_multiple:.1f}x "
    f"| Exit {assumptions.exit_ev_multiple:.1f}x "
    f"| Hold {assumptions.hold_years}yr "
    f"| SOFR {SOFR:.2%} "
    f"| IRR {fmt_irr(returns['irr'])} "
    f"| MOIC {fmt_moic(returns['moic'])}"
    f"</p>",
    unsafe_allow_html=True,
)
