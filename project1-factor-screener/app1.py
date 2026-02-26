"""
app1.py  --  S&P 500 Factor-Based Stock Screener
=================================================
Professional multi-tab Streamlit dashboard demonstrating:

  Factor Investing  -- Value / Momentum / Quality / Low Volatility scoring
  Portfolio Construction -- percentile ranking, composite weighting, equal-weight
  Risk Analytics    -- VaR, CVaR, Sortino, Calmar, Max Drawdown, Rolling metrics
  Performance Attribution -- Carhart 4-Factor (FF3 + Momentum) regression
  Portfolio Optimization  -- Mean-variance efficient frontier, Max Sharpe, Min Var

Tabs
----
  Overview     | key metrics, top holdings
  Holdings     | full portfolio table with factor score progress bars
  Factors      | distributions, correlations, sector breakdown, scatter
  Performance  | cumulative returns, rolling Sharpe/Alpha/Beta
  Risk         | drawdown, VaR/CVaR, monthly returns heatmap
  Attribution  | Fama-French 4-factor regression
  Optimization | efficient frontier, optimal weights
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from data_fetcher import (
    get_sp500_tickers,
    download_price_data,
    calculate_price_metrics,
    download_portfolio_prices,
    refresh_fundamentals_csv,
)
from factor_model import compute_factor_scores, build_portfolio
from backtest import run_backtest
from utils.formatting import build_portfolio_column_config, format_metrics_table
from utils.charts import (
    cumulative_returns_chart,
    drawdown_chart,
    rolling_metrics_chart,
    return_distribution_chart,
    monthly_returns_heatmap,
    factor_distribution_chart,
    factor_correlation_heatmap,
    sector_donut_chart,
    factor_scatter_chart,
    efficient_frontier_chart,
    ff_attribution_chart,
    capm_scatter_chart,
    portfolio_weights_chart,
)
from analysis.factor_attribution import fetch_ff_factors, run_ff4_regression, run_capm_fallback
from analysis.optimizer import compute_efficient_frontier, weights_comparison_df


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG + THEME
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="S&P 500 Factor Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
/* No !important on * — Streamlit's higher-specificity icon-font class rules
   (.material-symbols-rounded etc.) win naturally so arrows render as glyphs */
* { font-family: 'Inter', system-ui, sans-serif; }

.stApp { background: #0D1117; }
.block-container { padding: 1.5rem 2rem !important; max-width: 1400px; }
.stSidebar { background: #161B22 !important; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: #161B22;
    border: 1px solid #2A3441;
    border-radius: 8px;
    padding: 1rem 1.25rem;
}
[data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 600; color: #E6EDF3; }
[data-testid="stMetricLabel"] { font-size: 0.72rem; color: #8B9AB8; letter-spacing: 0.06em; text-transform: uppercase; }
[data-testid="stMetricDelta"] { font-size: 0.82rem; }

/* Headers */
h1, h2, h3, h4 { color: #E6EDF3 !important; letter-spacing: -0.02em; }
h1 { font-size: 1.75rem !important; font-weight: 700 !important; }
h2 { font-size: 1.3rem !important; font-weight: 600 !important; }
h3 { font-size: 1.1rem !important; }

/* Section divider */
.section-header {
    background: linear-gradient(135deg, #1C2230 0%, #161B22 100%);
    border-left: 3px solid #C9A96E;
    padding: 0.6rem 1rem;
    border-radius: 4px;
    margin: 1.25rem 0 0.75rem 0;
    color: #E6EDF3;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.info-box {
    background: #161B22;
    border: 1px solid #2A3441;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
    color: #8B9AB8;
    line-height: 1.55;
}
.badge {
    display: inline-block;
    background: #1C2230;
    border: 1px solid #2A3441;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.78rem;
    color: #C9A96E;
    margin: 2px;
}
/* Tabs — compact to avoid overflow scroll arrows */
.stTabs [data-baseweb="tab-list"] { background: #161B22; border-radius: 8px; padding: 4px; gap: 2px; }
.stTabs [data-baseweb="tab"] { background: transparent; border-radius: 6px; color: #8B9AB8; font-size: 0.82rem; font-weight: 500; padding: 6px 12px; }
.stTabs [aria-selected="true"] { background: #0D1117; color: #C9A96E !important; }
/* Hide tab overflow scroll arrows */
button[data-baseweb="tab-scroll-button"] { display: none !important; }

/* Sidebar */
.stSidebar .stSlider > div { padding: 0; }
.stSidebar label { color: #8B9AB8 !important; font-size: 0.82rem; }

/* Style the header bar to match the dark theme (don't hide — contains sidebar toggle) */
header[data-testid="stHeader"] {
    background-color: #0D1117 !important;
    border-bottom: 1px solid #161B22 !important;
}
/* Keep the sidebar collapse/expand button visible and styled */
button[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
    color: #8B9AB8 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    st.markdown("**Factor Weights**")
    w_val  = st.slider("Value",         0.0, 1.0, 0.25, 0.05, key="w_val")
    w_mom  = st.slider("Momentum",      0.0, 1.0, 0.35, 0.05, key="w_mom")
    w_qual = st.slider("Quality",       0.0, 1.0, 0.25, 0.05, key="w_qual")
    w_lvol = st.slider("Low Volatility",0.0, 1.0, 0.15, 0.05, key="w_lvol")

    w_total = w_val + w_mom + w_qual + w_lvol
    if w_total > 0:
        weights = {
            "value":    w_val  / w_total,
            "momentum": w_mom  / w_total,
            "quality":  w_qual / w_total,
            "lowvol":   w_lvol / w_total,
        }
    else:
        weights = {"value": 0.25, "momentum": 0.35, "quality": 0.25, "lowvol": 0.15}

    st.caption(
        f"Normalized — V: {weights['value']:.0%}  "
        f"M: {weights['momentum']:.0%}  "
        f"Q: {weights['quality']:.0%}  "
        f"LV: {weights['lowvol']:.0%}"
    )

    st.divider()
    st.markdown("**Portfolio**")
    n_stocks = st.slider("Portfolio Size", 10, 50, 20, key="n_stocks")

    st.divider()
    st.markdown("**Backtest**")
    backtest_years = st.selectbox("Lookback", ["1Y", "2Y", "3Y"], index=1)
    rf_pct = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 5.0, 0.25)
    rf_rate = rf_pct / 100

    st.divider()

    # ── Data Freshness ────────────────────────────────────────────────────────
    csv_path = os.path.join(os.path.dirname(__file__), "fundamentals.csv")
    if os.path.exists(csv_path):
        mtime     = os.path.getmtime(csv_path)
        age_days  = (datetime.now().timestamp() - mtime) / 86400
        age_label = f"{int(age_days)}d ago" if age_days >= 1 else "today"
    else:
        age_label = "missing"

    with st.expander(f"🔄 Data  ·  fundamentals: {age_label}"):
        st.caption("Price data refreshes hourly. Fundamentals update on demand (~20 min).")
        if st.button("Refresh fundamentals.csv", type="primary", key="btn_refresh"):
            with st.spinner("Fetching — do not close this tab (~20 min)..."):
                try:
                    refresh_fundamentals_csv(csv_path)
                    st.cache_data.clear()
                    st.success("Done — reload the page to apply new scores.")
                except Exception as e:
                    st.error(f"Refresh failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_screener_data():
    """Load fundamentals CSV + download bulk price data for scoring."""
    fundamentals  = pd.read_csv(csv_path)
    tickers       = get_sp500_tickers(csv_fallback=csv_path)
    price_data    = download_price_data(tickers)
    price_metrics = calculate_price_metrics(price_data, tickers)
    return fundamentals, price_metrics


@st.cache_data(ttl=3600, show_spinner=False)
def load_portfolio_prices(tickers_tuple: tuple, years: int) -> pd.DataFrame:
    """Download portfolio-only historical prices for backtest / optimizer."""
    return download_portfolio_prices(list(tickers_tuple), years=years)


@st.cache_data(ttl=3600, show_spinner=False)
def run_backtest_cached(tickers_tuple: tuple, start: str, end: str, rf: float) -> dict:
    return run_backtest(list(tickers_tuple), start, end, rf_rate=rf)


# Load data
with st.spinner("Loading market data (first run ~15 min, subsequent runs instant)..."):
    try:
        fundamentals, price_metrics = load_screener_data()
    except Exception as e:
        st.error(
            f"**Data load failed** — Yahoo Finance connection error. "
            f"This is usually transient. **Refresh the page** to retry. "
            f"\n\nDetail: `{e}`"
        )
        st.stop()

# Score & build portfolio
scored    = compute_factor_scores(fundamentals, price_metrics, weights)
portfolio = build_portfolio(scored, n_stocks)

# Backtest date range
bt_years  = int(backtest_years.replace("Y", ""))
bt_end    = datetime.today()
bt_start  = bt_end - timedelta(days=365 * bt_years + 10)
bt_start_str = bt_start.strftime("%Y-%m-%d")
bt_end_str   = bt_end.strftime("%Y-%m-%d")

port_tickers = tuple(portfolio["ticker"].tolist())

with st.spinner("Running backtest..."):
    bt = run_backtest_cached(port_tickers, bt_start_str, bt_end_str, rf_rate)

has_backtest = bt["n_trading_days"] > 30


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("## 📊  S&P 500 Factor-Based Stock Screener")
st.markdown(
    f'<span class="badge">Universe: S&P 500</span>'
    f'<span class="badge">Portfolio: {n_stocks} stocks</span>'
    f'<span class="badge">Factors: Value · Momentum · Quality · Low Vol</span>'
    f'<span class="badge">As of {datetime.today().strftime("%b %d, %Y")}</span>',
    unsafe_allow_html=True,
)
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview",
    "📋 Holdings",
    "🔬 Factors",
    "📈 Backtest",
    "⚠️ Risk",
    "🧮 FF4 Model",
    "🎯 Optimizer",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

with tab1:
    m = bt.get("metrics", {})

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        ann_ret = m.get("Annualized Return", 0)
        bench_r = m.get("Benchmark Return", 0)
        st.metric("Ann. Return", f"{ann_ret:.1%}", f"{ann_ret - bench_r:+.1%} vs SPY")
    with col2:
        alpha = m.get("Alpha (Ann.)", 0)
        st.metric("Alpha (Ann.)", f"{alpha:.1%}")
    with col3:
        sharpe = m.get("Sharpe Ratio", 0)
        st.metric("Sharpe Ratio", f"{sharpe:.2f}", "vs 1.0 hurdle")
    with col4:
        sortino = m.get("Sortino Ratio", 0)
        st.metric("Sortino Ratio", f"{sortino:.2f}")
    with col5:
        max_dd = m.get("Max Drawdown", 0)
        st.metric("Max Drawdown", f"{max_dd:.1%}")

    st.markdown("")

    col_a, col_b = st.columns([2, 1])

    with col_a:
        st.markdown('<div class="section-header">Cumulative Performance</div>', unsafe_allow_html=True)
        if has_backtest:
            fig = cumulative_returns_chart(bt["port_cum"], bt["bench_cum"])
            st.plotly_chart(fig, use_container_width=True, key="chart_overview_perf")
        else:
            st.info("Backtest data not available. Check your internet connection.")

    with col_b:
        st.markdown('<div class="section-header">Sector Exposure</div>', unsafe_allow_html=True)
        fig_sec = sector_donut_chart(portfolio)
        st.plotly_chart(fig_sec, use_container_width=True, key="chart_overview_sector")

    # Top 10 holdings mini-table
    st.markdown('<div class="section-header">Top 10 Holdings</div>', unsafe_allow_html=True)
    top10 = portfolio.head(10)[["final_rank", "ticker", "name", "sector",
                                 "composite_score", "pe_ratio", "ev_ebitda",
                                 "roe_pct", "gross_margin_pct", "momentum_pct"]].copy()
    st.dataframe(
        top10,
        use_container_width=True,
        hide_index=True,
        column_config={
            "final_rank":       st.column_config.NumberColumn("Rank", format="%d", width="small"),
            "ticker":           st.column_config.TextColumn("Ticker", width="small"),
            "name":             st.column_config.TextColumn("Company", width="medium"),
            "sector":           st.column_config.TextColumn("Sector", width="medium"),
            "composite_score":  st.column_config.ProgressColumn("Composite", min_value=0, max_value=1, format="%.2f"),
            "pe_ratio":         st.column_config.NumberColumn("P/E", format="%.1fx"),
            "ev_ebitda":        st.column_config.NumberColumn("EV/EBITDA", format="%.1fx"),
            "roe_pct":          st.column_config.NumberColumn("ROE (%)", format="%.1f%%"),
            "gross_margin_pct": st.column_config.NumberColumn("Gross Margin (%)", format="%.1f%%"),
            "momentum_pct":     st.column_config.NumberColumn("Mom 12-1M (%)", format="%.1f%%"),
        },
    )

    # Summary stats
    st.markdown('<div class="section-header">Portfolio Statistics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks", n_stocks)
    c2.metric("Sectors", portfolio["sector"].nunique())
    c3.metric("Avg P/E", f"{portfolio['pe_ratio'].median():.1f}x")
    c4.metric("Avg Momentum", f"{portfolio['momentum_pct'].mean():.1f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: HOLDINGS
# ═══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-header">Portfolio Holdings — Factor Scores</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">'
        'Factor scores are percentile ranks within the S&P 500 universe (0 = worst, 1 = best). '
        'Composite score is the user-weighted average. '
        'Progress bars show relative factor strength at a glance.'
        '</div>',
        unsafe_allow_html=True,
    )

    display_cols = [c for c in [
        "final_rank", "ticker", "name", "sector",
        "market_cap_b", "pe_ratio", "ev_ebitda",
        "roe_pct", "gross_margin_pct", "momentum_pct", "vol_12m_pct",
        "value_score", "momentum_score", "quality_score", "lowvol_score",
        "composite_score",
    ] if c in portfolio.columns]

    st.dataframe(
        portfolio[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config=build_portfolio_column_config(),
        height=680,
    )

    st.divider()

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown('<div class="section-header">Factor Score Distribution in Portfolio</div>', unsafe_allow_html=True)
        factor_cols = [c for c in ["value_score", "momentum_score", "quality_score", "lowvol_score"] if c in portfolio.columns]
        melted = portfolio[["ticker"] + factor_cols].melt(id_vars="ticker", var_name="Factor", value_name="Score")
        melted["Factor"] = melted["Factor"].str.replace("_score", "").str.title()

        import plotly.express as px
        fig_box = px.box(
            melted, x="Factor", y="Score",
            color="Factor",
            color_discrete_map={
                "Value": "#3498DB", "Momentum": "#C9A96E",
                "Quality": "#2ECC71", "Lowvol": "#9B59B6",
            },
            template="plotly_dark",
        )
        fig_box.update_layout(
            paper_bgcolor="#161B22", plot_bgcolor="#1C2230",
            font_family="Inter", showlegend=False,
            margin=dict(l=40, r=20, t=30, b=40),
            height=300,
        )
        fig_box.update_yaxes(range=[0, 1], title_text="Score (0-1)")
        st.plotly_chart(fig_box, use_container_width=True, key="chart_holdings_box")

    with col_r:
        st.markdown('<div class="section-header">Factor Weight Contributions</div>', unsafe_allow_html=True)
        contrib_data = {
            "Factor":     ["Value", "Momentum", "Quality", "Low Vol"],
            "Weight":     [weights["value"], weights["momentum"], weights["quality"], weights["lowvol"]],
            "Avg Score":  [
                portfolio.get("value_score", pd.Series([0.5])).mean(),
                portfolio.get("momentum_score", pd.Series([0.5])).mean(),
                portfolio.get("quality_score", pd.Series([0.5])).mean(),
                portfolio.get("lowvol_score", pd.Series([0.5])).mean(),
            ],
        }
        contrib_df = pd.DataFrame(contrib_data)
        contrib_df["Contribution"] = contrib_df["Weight"] * contrib_df["Avg Score"]

        import plotly.graph_objects as go
        fig_contrib = go.Figure(go.Bar(
            x=contrib_df["Factor"],
            y=contrib_df["Contribution"],
            marker_color=["#3498DB", "#C9A96E", "#2ECC71", "#9B59B6"],
            text=[f"{v:.3f}" for v in contrib_df["Contribution"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Contribution: %{y:.3f}<extra></extra>",
        ))
        fig_contrib.update_layout(
            paper_bgcolor="#161B22", plot_bgcolor="#1C2230",
            font=dict(family="Inter", color="#E6EDF3"),
            showlegend=False, height=300,
            margin=dict(l=40, r=20, t=30, b=40),
            yaxis=dict(gridcolor="#2A3441", title_text="Weighted Score Contribution"),
            xaxis=dict(gridcolor="#2A3441"),
        )
        st.plotly_chart(fig_contrib, use_container_width=True, key="chart_holdings_contrib")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: FACTOR ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tab3:
    # Full-width violin strip chart (needs horizontal space to breathe)
    st.markdown('<div class="section-header">Factor Score Distributions: Universe vs Portfolio</div>', unsafe_allow_html=True)
    fig_dist = factor_distribution_chart(scored, portfolio)
    st.plotly_chart(fig_dist, use_container_width=True, key="chart_factor_dist")

    # Sector donut — narrower, centred
    col_sec, _ = st.columns([1, 2])
    with col_sec:
        st.markdown('<div class="section-header">Sector Allocation</div>', unsafe_allow_html=True)
        fig_sector = sector_donut_chart(portfolio)
        st.plotly_chart(fig_sector, use_container_width=True, key="chart_factor_sector")

    st.markdown('<div class="section-header">Factor Correlation Matrix (Spearman)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">'
        'Spearman rank correlations between factor scores and raw metrics across the S&P 500 universe. '
        'Low correlation between factors indicates effective diversification of signal sources — '
        'ideally, Value and Momentum should be weakly correlated (they tend to capture different risk premia).'
        '</div>',
        unsafe_allow_html=True,
    )
    fig_corr = factor_correlation_heatmap(scored)
    st.plotly_chart(fig_corr, use_container_width=True, key="chart_factor_corr")

    st.markdown('<div class="section-header">Value vs Momentum Scatter</div>', unsafe_allow_html=True)
    fig_scatter = factor_scatter_chart(scored, "value_score", "momentum_score", portfolio)
    st.plotly_chart(fig_scatter, use_container_width=True, key="chart_factor_scatter")

    # Factor IC summary table
    st.markdown('<div class="section-header">Factor Summary Statistics</div>', unsafe_allow_html=True)
    score_cols = [c for c in ["value_score", "momentum_score", "quality_score", "lowvol_score"] if c in scored.columns]
    summary = scored[score_cols].describe().T
    summary.index = [c.replace("_score", "").title() for c in summary.index]
    summary = summary[["mean", "std", "25%", "50%", "75%"]]
    summary.columns = ["Mean", "Std Dev", "25th %ile", "Median", "75th %ile"]
    st.dataframe(
        summary.round(3),
        use_container_width=True,
        column_config={c: st.column_config.NumberColumn(c, format="%.3f") for c in summary.columns},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

with tab4:
    if not has_backtest:
        st.warning("Backtest data unavailable. Check internet connection.")
    else:
        m = bt["metrics"]

        # KPI row
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Ann. Return",  f"{m.get('Annualized Return', 0):.1%}")
        c2.metric("SPY Return",   f"{m.get('Benchmark Return', 0):.1%}")
        c3.metric("Alpha",        f"{m.get('Alpha (Ann.)', 0):.1%}")
        c4.metric("Beta",         f"{m.get('Beta', 1):.2f}")
        c5.metric("Sharpe",       f"{m.get('Sharpe Ratio', 0):.2f}")
        c6.metric("Info. Ratio",  f"{m.get('Information Ratio', 0):.2f}")

        st.markdown('<div class="section-header">Cumulative Returns</div>', unsafe_allow_html=True)
        fig_perf = cumulative_returns_chart(bt["port_cum"], bt["bench_cum"])
        st.plotly_chart(fig_perf, use_container_width=True, key="chart_perf_cum")

        if not bt["rolling_df"].empty:
            st.markdown('<div class="section-header">Rolling 252-Day Risk Metrics</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="info-box">'
                'Rolling metrics computed over a 252 trading-day (1-year) window. '
                'Rolling alpha shows whether outperformance is persistent or concentrated in specific regimes.'
                '</div>',
                unsafe_allow_html=True,
            )
            fig_roll = rolling_metrics_chart(bt["rolling_df"])
            st.plotly_chart(fig_roll, use_container_width=True, key="chart_perf_rolling")

        # Performance table
        st.markdown('<div class="section-header">Detailed Performance Metrics</div>', unsafe_allow_html=True)
        display_metrics = {
            k: v for k, v in m.items()
            if not isinstance(v, pd.Series) and k not in ("drawdown_series", "cumulative_returns")
        }
        metrics_df = format_metrics_table(display_metrics)
        col_m1, col_m2 = st.columns(2)
        half = len(metrics_df) // 2
        with col_m1:
            st.dataframe(metrics_df.iloc[:half], use_container_width=True, hide_index=True)
        with col_m2:
            st.dataframe(metrics_df.iloc[half:], use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: RISK
# ═══════════════════════════════════════════════════════════════════════════════

with tab5:
    if not has_backtest:
        st.warning("Backtest data unavailable.")
    else:
        m = bt["metrics"]

        # VaR / CVaR cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("VaR 95% (daily)",  f"{m.get('VaR 95% (daily)', 0):.2%}")
        c2.metric("CVaR 95% (daily)", f"{m.get('CVaR 95% (daily)', 0):.2%}")
        c3.metric("Max Drawdown",     f"{m.get('Max Drawdown', 0):.1%}")
        c4.metric("Max DD Duration",  f"{int(m.get('Max DD Duration (days)', 0))} days")

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown('<div class="section-header">Portfolio Drawdown</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="info-box">'
                'Drawdown = (current value - running peak) / running peak. '
                'Measures peak-to-trough loss at each point in time.'
                '</div>',
                unsafe_allow_html=True,
            )
            fig_dd = drawdown_chart(bt["drawdown"])
            st.plotly_chart(fig_dd, use_container_width=True, key="chart_risk_drawdown")

        with col_r:
            st.markdown('<div class="section-header">Daily Return Distribution</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="info-box">'
                'VaR = the daily loss not exceeded with 95% / 99% probability. '
                'CVaR (Expected Shortfall) = expected loss given that VaR threshold is breached.'
                '</div>',
                unsafe_allow_html=True,
            )
            fig_dist_r = return_distribution_chart(
                bt["port_daily"],
                m.get("VaR 95% (daily)", -0.02),
                m.get("CVaR 95% (daily)", -0.03),
                m.get("VaR 99% (daily)", -0.03),
                m.get("CVaR 99% (daily)", -0.04),
            )
            st.plotly_chart(fig_dist_r, use_container_width=True, key="chart_risk_dist")

        if not bt["monthly_table"].empty:
            st.markdown('<div class="section-header">Calendar Returns</div>', unsafe_allow_html=True)
            # Include annual column
            fig_cal = monthly_returns_heatmap(bt["monthly_table"])
            st.plotly_chart(fig_cal, use_container_width=True, key="chart_risk_calendar")

        # Risk metrics table
        st.markdown('<div class="section-header">Tail Risk Summary</div>', unsafe_allow_html=True)
        tail_keys = [
            "VaR 95% (daily)", "CVaR 95% (daily)", "VaR 99% (daily)", "CVaR 99% (daily)",
            "Max Drawdown", "Max DD Duration (days)",
            "Sortino Ratio", "Calmar Ratio",
            "Skewness", "Excess Kurtosis", "Win Rate",
        ]
        tail_data = {k: m[k] for k in tail_keys if k in m}
        st.dataframe(
            format_metrics_table(tail_data),
            use_container_width=True,
            hide_index=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: FACTOR ATTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

with tab6:
    st.markdown('<div class="section-header">Carhart 4-Factor Attribution Model</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">'
        '<b>Carhart (1997) 4-Factor Model:</b><br>'
        'r<sub>p</sub> − RF = α + β<sub>Mkt</sub>·(Mkt−RF) + β<sub>SMB</sub>·SMB + β<sub>HML</sub>·HML + β<sub>MOM</sub>·MOM + ε<br><br>'
        '<b>Mkt−RF</b>: Excess market return (CAPM market factor)<br>'
        '<b>SMB</b>: Small-Minus-Big — size premium (long small-cap, short large-cap)<br>'
        '<b>HML</b>: High-Minus-Low — value premium (long high B/M, short low B/M)<br>'
        '<b>MOM</b>: Prior-year momentum (long past winners, short past losers)<br>'
        '<b>α</b>: Jensen\'s alpha — return unexplained by the four factors<br>'
        '<br>Significance: * p&lt;0.10, ** p&lt;0.05, *** p&lt;0.01'
        '</div>',
        unsafe_allow_html=True,
    )

    if not has_backtest:
        st.warning("Backtest data needed for attribution. Check internet connection.")
    else:
        attribution = None
        ff_source   = "Fama-French 4-Factor"

        with st.spinner("Fetching Fama-French factors from Kenneth French Data Library..."):
            factors_df = fetch_ff_factors(bt["start"], bt["end"])

        if factors_df is not None:
            attribution = run_ff4_regression(bt["port_daily"], factors_df)

        if attribution is None:
            st.info("FF factor data unavailable — falling back to CAPM regression.")
            ff_source   = "CAPM (1-Factor)"
            attribution = run_capm_fallback(bt["port_daily"], bt["bench_daily"])

        if attribution is not None:
            # KPI cards
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Alpha (Ann.)", f"{attribution['alpha_ann']:.1%}",
                      f"t = {attribution['alpha_tstat']:.2f}")
            c2.metric("Market Beta", f"{attribution['betas'].get('Mkt-RF', 0):.3f}")
            c3.metric("R-Squared",   f"{attribution['r_squared']:.1%}")
            c4.metric("Observations", f"{attribution['n_obs']}")

            col_l, col_r = st.columns(2)

            with col_l:
                if ff_source == "CAPM (1-Factor)":
                    st.markdown('<div class="section-header">Security Characteristic Line (CAPM)</div>', unsafe_allow_html=True)
                    st.markdown(
                        '<div class="info-box">'
                        'The <b>Security Characteristic Line (SCL)</b> plots each day\'s portfolio return '
                        'against the market (SPY) return. The slope is <b>beta</b> — how much the portfolio '
                        'amplifies market moves. The y-intercept is <b>alpha</b> — return not explained by '
                        'the market. <b>R²</b> shows what fraction of portfolio variance is driven by the market.'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    fig_attr = capm_scatter_chart(bt["port_daily"], bt["bench_daily"], attribution)
                else:
                    st.markdown(f'<div class="section-header">Factor Loadings — {ff_source}</div>', unsafe_allow_html=True)
                    fig_attr = ff_attribution_chart(attribution, f"Factor Loadings ({ff_source})")
                st.plotly_chart(fig_attr, use_container_width=True, key="chart_attr_bars")

            with col_r:
                st.markdown('<div class="section-header">Regression Output Table</div>', unsafe_allow_html=True)

                def sig(p):
                    if p < 0.01: return "***"
                    if p < 0.05: return "**"
                    if p < 0.10: return "*"
                    return ""

                reg_rows = [{
                    "Factor":  "Alpha",
                    "Beta":    f"{attribution['alpha_ann']:.4f}  ({attribution['alpha_daily']:.6f}/day)",
                    "t-stat":  f"{attribution['alpha_tstat']:.3f}",
                    "p-value": f"{attribution['alpha_pvalue']:.4f}",
                    "Sig.":    sig(attribution["alpha_pvalue"]),
                }]
                for f in attribution["factor_names"]:
                    reg_rows.append({
                        "Factor":  f,
                        "Beta":    f"{attribution['betas'][f]:.4f}",
                        "t-stat":  f"{attribution['t_stats'][f]:.3f}",
                        "p-value": f"{attribution['p_values'][f]:.4f}",
                        "Sig.":    sig(attribution["p_values"][f]),
                    })
                st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)

                st.markdown(
                    f"**R² = {attribution['r_squared']:.3f}**  "
                    f"(Adj. R² = {attribution['adj_r_squared']:.3f})<br>"
                    f"N = {attribution['n_obs']} trading days",
                    unsafe_allow_html=True,
                )

            # Return decomposition
            if "decomposition" in attribution:
                st.markdown('<div class="section-header">Annualized Return Decomposition</div>', unsafe_allow_html=True)
                decomp = attribution["decomposition"]
                decomp_rows = [{"Component": k, "Contribution (Ann.)": f"{v:.2%}"} for k, v in decomp.items()]
                st.dataframe(pd.DataFrame(decomp_rows), use_container_width=True, hide_index=True)
        else:
            st.error("Attribution regression failed. Insufficient data.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7: OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

with tab7:
    st.markdown('<div class="section-header">Mean-Variance Portfolio Optimization</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">'
        '<b>Markowitz (1952) Mean-Variance Framework:</b><br>'
        'Finds the set of portfolios that minimize variance for a given expected return (efficient frontier). '
        'Constraints: long-only, max 15% per stock, weights sum to 1.<br><br>'
        '<b>Min Variance ◆</b>: Lowest achievable portfolio volatility.<br>'
        '<b>Max Sharpe ★</b>: Tangency portfolio — highest risk-adjusted return (Sharpe ratio).<br>'
        '<b>Equal Weight ●</b>: Naive 1/N benchmark for comparison.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Load portfolio prices for optimization
    bt_years_int = int(backtest_years.replace("Y", ""))
    with st.spinner("Computing efficient frontier..."):
        port_prices = load_portfolio_prices(port_tickers, bt_years_int)

    ef_result = None
    if not port_prices.empty and len(port_prices.columns) >= 2:
        try:
            ef_result = compute_efficient_frontier(
                port_prices,
                rf_rate=rf_rate,
                n_points=50,
                n_random=2000,
            )
        except Exception as e:
            st.warning(f"Optimization did not converge: {e}")

    if ef_result:
        ms = ef_result["max_sharpe"]
        mv = ef_result["min_var"]
        ew = ef_result["equal_weight"]

        # KPI cards
        c1, c2, c3 = st.columns(3)
        c1.metric("Max Sharpe Portfolio",
                  f"Sharpe: {ms['sharpe']:.2f}",
                  f"Return: {ms['return']:.1%} | Vol: {ms['vol']:.1%}")
        c2.metric("Min Variance Portfolio",
                  f"Vol: {mv['vol']:.1%}",
                  f"Return: {mv['return']:.1%} | Sharpe: {mv['sharpe']:.2f}")
        c3.metric("Equal Weight Benchmark",
                  f"Sharpe: {ew['sharpe']:.2f}",
                  f"Return: {ew['return']:.1%} | Vol: {ew['vol']:.1%}")

        st.markdown('<div class="section-header">Efficient Frontier</div>', unsafe_allow_html=True)
        fig_ef = efficient_frontier_chart(ef_result)
        st.plotly_chart(fig_ef, use_container_width=True, key="chart_opt_frontier")

        # Weights comparison
        st.markdown('<div class="section-header">Portfolio Weights Comparison</div>', unsafe_allow_html=True)
        wdf = weights_comparison_df(ef_result)
        if not wdf.empty:
            fig_weights = portfolio_weights_chart(wdf)
            st.plotly_chart(fig_weights, use_container_width=True, key="chart_opt_weights")

            # Weights table
            st.markdown("**Top 20 Holdings by Max Sharpe Weight**")
            top_wdf = wdf.head(20).copy()
            top_wdf["ew"]        = top_wdf["ew"].map(lambda v: f"{v:.2%}")
            top_wdf["min_var"]   = top_wdf["min_var"].map(lambda v: f"{v:.2%}")
            top_wdf["max_sharpe"]= top_wdf["max_sharpe"].map(lambda v: f"{v:.2%}")
            top_wdf.columns      = ["Ticker", "Equal Weight", "Min Variance", "Max Sharpe"]
            st.dataframe(top_wdf, use_container_width=True, hide_index=True)

        # Optimization stats
        st.markdown('<div class="section-header">Portfolio Comparison Summary</div>', unsafe_allow_html=True)
        summary_data = {
            "Portfolio":          ["Equal Weight",    "Min Variance",    "Max Sharpe"],
            "Ann. Return (%)":    [f"{ew['return']:.2%}", f"{mv['return']:.2%}", f"{ms['return']:.2%}"],
            "Ann. Volatility (%)": [f"{ew['vol']:.2%}",   f"{mv['vol']:.2%}",   f"{ms['vol']:.2%}"],
            "Sharpe Ratio":       [f"{ew['sharpe']:.2f}", f"{mv['sharpe']:.2f}", f"{ms['sharpe']:.2f}"],
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    else:
        st.info("Insufficient price history to compute efficient frontier.")
        st.markdown("Need at least 2 years of data for 2+ portfolio tickers.")
