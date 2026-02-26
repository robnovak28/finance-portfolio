"""
charts.py
---------
Professional Plotly chart builders for the factor screener.
All charts use a consistent dark institutional theme.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Design System ───────────────────────────────────────────────────────────
FONT   = "Inter, system-ui, -apple-system, sans-serif"
BG     = "#0E1117"
PAPER  = "#161B22"
PANEL  = "#1C2230"
GRID   = "#2A3441"
BORDER = "#30363D"
GOLD   = "#C9A96E"
GREEN  = "#2ECC71"
RED    = "#E74C3C"
BLUE   = "#3498DB"
PURPLE = "#9B59B6"
ORANGE = "#E67E22"
TEAL   = "#16A085"
TEXT   = "#E6EDF3"
SUBTEXT = "#8B9AB8"
MUTED   = "#484F58"

SECTOR_COLORS = [
    "#3498DB", "#2ECC71", "#E74C3C", "#F39C12", "#9B59B6",
    "#1ABC9C", "#E67E22", "#34495E", "#C0392B", "#16A085",
    "#8E44AD", "#2980B9",
]


def _apply_theme(fig, title="", height=420, show_legend=True):
    fig.update_layout(
        font=dict(family=FONT, color=TEXT, size=12),
        paper_bgcolor=PAPER,
        plot_bgcolor=PANEL,
        height=height,
        margin=dict(l=60, r=30, t=55, b=50),
        title=dict(
            text=title,
            font=dict(size=14, color=TEXT, family=FONT, weight=600),
            x=0.02, y=0.97,
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=BORDER,
            borderwidth=1,
            font=dict(color=SUBTEXT, size=11),
        ) if show_legend else dict(visible=False),
        hoverlabel=dict(
            bgcolor=PANEL,
            bordercolor=BORDER,
            font=dict(family=FONT, color=TEXT, size=12),
        ),
    )
    fig.update_xaxes(
        gridcolor=GRID, linecolor=GRID, zeroline=False,
        tickfont=dict(color=SUBTEXT, size=11),
    )
    fig.update_yaxes(
        gridcolor=GRID, linecolor=GRID, zeroline=False,
        tickfont=dict(color=SUBTEXT, size=11),
    )
    return fig


# ─── Cumulative Returns ───────────────────────────────────────────────────────
def cumulative_returns_chart(port_cum: pd.Series, bench_cum: pd.Series, title="Cumulative Performance"):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=port_cum.index, y=port_cum.values * 100,
        name="Factor Portfolio",
        line=dict(color=GOLD, width=2.5),
        fill="tozeroy",
        fillcolor="rgba(201,169,110,0.08)",
        hovertemplate="%{x|%b %d, %Y}<br>Portfolio: <b>%{y:.2f}%</b><extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=bench_cum.index, y=bench_cum.values * 100,
        name="S&P 500 (SPY)",
        line=dict(color=BLUE, width=2, dash="dash"),
        hovertemplate="%{x|%b %d, %Y}<br>S&P 500: <b>%{y:.2f}%</b><extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dot"))

    _apply_theme(fig, title, height=420)
    fig.update_yaxes(ticksuffix="%", title_text="Cumulative Return")
    fig.update_xaxes(title_text="Date")
    return fig


# ─── Drawdown ─────────────────────────────────────────────────────────────────
def drawdown_chart(drawdown: pd.Series, title="Portfolio Drawdown"):
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values * 100,
        name="Drawdown",
        line=dict(color=RED, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(231,76,60,0.20)",
        hovertemplate="%{x|%b %d, %Y}<br>Drawdown: <b>%{y:.2f}%</b><extra></extra>",
    ))

    for lvl in [-10, -20, -30]:
        fig.add_hline(y=lvl, line=dict(color=MUTED, width=0.8, dash="dot"),
                      annotation_text=f"{lvl}%", annotation_position="right",
                      annotation_font=dict(color=MUTED, size=10))

    fig.add_annotation(
        x=max_dd_date, y=max_dd * 100,
        text=f"Max DD: {max_dd:.1%}",
        showarrow=True, arrowhead=2, arrowcolor=RED,
        font=dict(color=RED, size=11), bgcolor=PANEL, bordercolor=RED,
    )

    _apply_theme(fig, title, height=320, show_legend=False)
    fig.update_yaxes(ticksuffix="%", title_text="Drawdown (%)")
    fig.update_xaxes(title_text="Date")
    return fig


# ─── Rolling Metrics ─────────────────────────────────────────────────────────
def rolling_metrics_chart(rolling_df: pd.DataFrame, title="Rolling Risk Metrics (252-Day)"):
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Rolling Sharpe Ratio", "Rolling Alpha (Ann.)", "Rolling Beta"),
        vertical_spacing=0.08,
    )

    # Sharpe
    fig.add_trace(go.Scatter(
        x=rolling_df.index, y=rolling_df["rolling_sharpe"],
        line=dict(color=GOLD, width=1.8),
        name="Sharpe",
        hovertemplate="%{x|%b %Y}<br>Sharpe: <b>%{y:.2f}</b><extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dot"), row=1, col=1)
    fig.add_hline(y=1, line=dict(color=GREEN, width=0.8, dash="dot"), row=1, col=1)

    # Alpha
    colors_alpha = [GREEN if v >= 0 else RED for v in rolling_df["rolling_alpha"]]
    fig.add_trace(go.Bar(
        x=rolling_df.index, y=rolling_df["rolling_alpha"] * 100,
        marker_color=colors_alpha, name="Alpha",
        hovertemplate="%{x|%b %Y}<br>Alpha: <b>%{y:.2f}%</b><extra></extra>",
    ), row=2, col=1)

    # Beta
    fig.add_trace(go.Scatter(
        x=rolling_df.index, y=rolling_df["rolling_beta"],
        line=dict(color=BLUE, width=1.8),
        name="Beta",
        hovertemplate="%{x|%b %Y}<br>Beta: <b>%{y:.2f}</b><extra></extra>",
    ), row=3, col=1)
    fig.add_hline(y=1, line=dict(color=MUTED, width=1, dash="dot"), row=3, col=1)

    _apply_theme(fig, title, height=550)
    fig.update_yaxes(ticksuffix="", title_text="Sharpe", row=1, col=1)
    fig.update_yaxes(ticksuffix="%", title_text="Alpha", row=2, col=1)
    fig.update_yaxes(title_text="Beta", row=3, col=1)

    for i in range(1, 4):
        fig.update_xaxes(
            gridcolor=GRID, linecolor=GRID,
            tickfont=dict(color=SUBTEXT, size=10), row=i, col=1,
        )
        fig.update_yaxes(
            gridcolor=GRID, linecolor=GRID,
            tickfont=dict(color=SUBTEXT, size=10), row=i, col=1,
        )
    for ann in fig.layout.annotations:
        ann.font.color = SUBTEXT
        ann.font.size = 12
    return fig


# ─── Return Distribution ──────────────────────────────────────────────────────
def return_distribution_chart(returns: pd.Series, var_95: float, cvar_95: float,
                               var_99: float, cvar_99: float,
                               title="Daily Return Distribution"):
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=returns * 100,
        nbinsx=80,
        name="Daily Returns",
        marker_color=BLUE,
        opacity=0.65,
        hovertemplate="Return: <b>%{x:.2f}%</b><br>Count: %{y}<extra></extra>",
    ))

    mean_r = returns.mean() * 100
    lines = [
        (mean_r,          GREEN,     f"Mean: {mean_r:.3f}%"),
        (var_95 * 100,   ORANGE,    f"VaR 95%: {var_95:.2%}"),
        (var_99 * 100,   RED,       f"VaR 99%: {var_99:.2%}"),
        (cvar_95 * 100,  "#FF6B35", f"CVaR 95%: {cvar_95:.2%}"),
    ]
    for val, color, label in lines:
        fig.add_vline(x=val, line=dict(color=color, width=1.5, dash="dash"))
        # Invisible scatter trace purely for the legend entry
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            name=label, line=dict(color=color, dash="dash", width=1.5),
            showlegend=True,
        ))

    _apply_theme(fig, title, height=380, show_legend=True)
    fig.update_layout(
        legend=dict(
            x=0.01, y=0.99, xanchor="left", yanchor="top",
            bgcolor="rgba(22,27,34,0.85)",
            bordercolor=BORDER, borderwidth=1,
        ),
    )
    fig.update_xaxes(ticksuffix="%", title_text="Daily Return (%)")
    fig.update_yaxes(title_text="Frequency")
    return fig


# ─── Monthly Returns Heatmap ─────────────────────────────────────────────────
def monthly_returns_heatmap(pivot: pd.DataFrame, title="Monthly Returns (%)"):
    # Format display text
    z_vals = pivot.values * 100
    text_vals = [
        [f"{v:.1f}%" if not np.isnan(v) else "" for v in row]
        for row in z_vals
    ]
    months = list(pivot.columns)
    years  = [str(y) for y in pivot.index]

    # Limit colorscale range to avoid outliers washing out the palette
    vmax = min(abs(np.nanpercentile(z_vals, 95)), 8)

    fig = go.Figure(go.Heatmap(
        z=z_vals,
        x=months,
        y=years,
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=10, color="black"),
        colorscale="RdYlGn",
        zmid=0,
        zmin=-vmax, zmax=vmax,
        colorbar=dict(
            title=dict(text="Return %", font=dict(color=SUBTEXT, size=11)),
            ticksuffix="%",
            tickfont=dict(color=SUBTEXT, size=10),
            bgcolor=PANEL,
        ),
        hovertemplate="<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>",
    ))

    _apply_theme(fig, title, height=max(280, len(years) * 52 + 100), show_legend=False)
    fig.update_xaxes(side="top", tickfont=dict(size=11))
    return fig


# ─── Factor Score Distributions ──────────────────────────────────────────────
def factor_distribution_chart(scored_df: pd.DataFrame, portfolio_df: pd.DataFrame,
                               title="Factor Score Distributions: Universe vs Portfolio"):
    """
    Single full-width chart: 4 horizontal violin rows (one per factor).
    The S&P 500 universe distribution is shown as a violin with box/mean line;
    portfolio stocks are overlaid as gold diamond markers (hover shows ticker).
    """
    factors = [
        ("value_score",    "Value",    BLUE),
        ("momentum_score", "Momentum", GOLD),
        ("quality_score",  "Quality",  GREEN),
        ("lowvol_score",   "Low Vol",  PURPLE),
    ]

    fig = go.Figure()
    first = True

    for col, label, color in factors:
        if col not in scored_df.columns:
            continue

        uni_vals = scored_df[col].dropna()
        r, g, b  = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

        # Universe: horizontal violin (500 stocks)
        fig.add_trace(go.Violin(
            x=uni_vals,
            y=[label] * len(uni_vals),
            orientation="h",
            name="S&P 500 Universe",
            side="both",
            box_visible=True,
            meanline_visible=True,
            points=False,
            fillcolor=f"rgba({r},{g},{b},0.22)",
            line_color=color,
            showlegend=first,
            legendgroup="universe",
            hovertemplate=f"{label}: %{{x:.3f}}<extra>S&P 500 Universe</extra>",
        ))

        # Portfolio: scatter diamonds (~20 stocks)
        if col in portfolio_df.columns:
            pvals = portfolio_df[col].dropna()
            ptxt  = (portfolio_df.loc[pvals.index, "ticker"].values
                     if "ticker" in portfolio_df.columns
                     else np.full(len(pvals), ""))
            if len(pvals) > 0:
                fig.add_trace(go.Scatter(
                    x=pvals.values,
                    y=[label] * len(pvals),
                    mode="markers",
                    name="Portfolio",
                    marker=dict(
                        color=GOLD, size=9, symbol="diamond",
                        line=dict(color=TEXT, width=0.8), opacity=0.9,
                    ),
                    text=ptxt,
                    showlegend=first,
                    legendgroup="portfolio",
                    hovertemplate="<b>%{text}</b><br>%{y} score: %{x:.3f}<extra>Portfolio</extra>",
                ))

        first = False

    _apply_theme(fig, title, height=620)
    fig.update_xaxes(range=[-0.02, 1.02], title_text="Score  (0 = worst → 1 = best)")
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=["Low Vol", "Quality", "Momentum", "Value"],
        tickfont=dict(color=TEXT, size=12),
    )
    fig.update_layout(violingap=0.25, violinmode="overlay")
    return fig


# ─── Factor Correlation Heatmap ───────────────────────────────────────────────
def factor_correlation_heatmap(scored_df: pd.DataFrame, title="Factor Correlation Matrix"):
    cols = [c for c in ["value_score", "momentum_score", "quality_score", "lowvol_score",
                         "composite_score", "pe_ratio", "roe", "gross_margin", "momentum"]
            if c in scored_df.columns]
    labels = {
        "value_score": "Value Score", "momentum_score": "Mom Score",
        "quality_score": "Qual Score", "lowvol_score": "LowVol Score",
        "composite_score": "Composite", "pe_ratio": "P/E Ratio",
        "roe": "ROE", "gross_margin": "Gross Margin", "momentum": "Momentum (raw)",
    }
    corr = scored_df[cols].corr(method="spearman").round(2)
    display_labels = [labels.get(c, c) for c in cols]

    text_vals = [[f"{v:.2f}" for v in row] for row in corr.values]

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=display_labels,
        y=display_labels,
        text=text_vals,
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorscale="RdBu",
        zmid=0, zmin=-1, zmax=1,
        colorbar=dict(
            title=dict(text="Spearman ρ", font=dict(color=SUBTEXT, size=11)),
            tickfont=dict(color=SUBTEXT, size=10),
            bgcolor=PANEL,
        ),
        hovertemplate="<b>%{y} vs %{x}</b><br>ρ = %{z:.2f}<extra></extra>",
    ))

    _apply_theme(fig, title, height=480, show_legend=False)
    fig.update_xaxes(tickangle=-35, tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    return fig


# ─── Sector Allocation ────────────────────────────────────────────────────────
def sector_donut_chart(portfolio_df: pd.DataFrame, title="Sector Allocation"):
    counts = portfolio_df["sector"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.50,
        marker=dict(colors=SECTOR_COLORS[:len(counts)], line=dict(color=PANEL, width=2)),
        textfont=dict(size=11, family=FONT),
        hovertemplate="<b>%{label}</b><br>%{value} stocks (%{percent})<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{len(portfolio_df)}</b><br>Stocks",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color=TEXT, family=FONT),
    )
    _apply_theme(fig, title, height=380)
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig


# ─── Factor Scatter ───────────────────────────────────────────────────────────
def factor_scatter_chart(scored_df: pd.DataFrame, x_col="value_score", y_col="momentum_score",
                          portfolio_df: pd.DataFrame = None,
                          title="Value vs Momentum Factor Scores"):
    labels = {
        "value_score": "Value Score", "momentum_score": "Momentum Score",
        "quality_score": "Quality Score", "lowvol_score": "Low Vol Score",
    }
    data = scored_df[[x_col, y_col, "composite_score", "ticker"]].dropna()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data[x_col], y=data[y_col],
        mode="markers",
        marker=dict(
            size=5,
            color=data["composite_score"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(
                title=dict(text="Composite", font=dict(color=SUBTEXT, size=10)),
                tickfont=dict(color=SUBTEXT, size=10),
                thickness=12, len=0.65, x=1.01,
            ),
            opacity=0.6,
        ),
        text=data["ticker"],
        hovertemplate="<b>%{text}</b><br>"
                      + f"{labels.get(x_col, x_col)}: " + "%{x:.3f}<br>"
                      + f"{labels.get(y_col, y_col)}: " + "%{y:.3f}<extra></extra>",
        name="S&P 500 Universe",
    ))

    if portfolio_df is not None and len(portfolio_df) > 0:
        pdata = portfolio_df[[x_col, y_col, "ticker"]].dropna() if x_col in portfolio_df.columns else pd.DataFrame()
        if len(pdata) > 0:
            fig.add_trace(go.Scatter(
                x=pdata[x_col], y=pdata[y_col],
                mode="markers+text",
                marker=dict(size=9, color=GOLD, symbol="star", line=dict(color=TEXT, width=1)),
                text=pdata["ticker"],
                textposition="top center",
                textfont=dict(size=8, color=GOLD),
                hovertemplate="<b>%{text}</b> (Portfolio)<br>"
                              + f"{labels.get(x_col, x_col)}: " + "%{x:.3f}<br>"
                              + f"{labels.get(y_col, y_col)}: " + "%{y:.3f}<extra></extra>",
                name="Portfolio",
            ))

    _apply_theme(fig, title, height=420)
    # Push legend to bottom-left so it doesn't clash with the right-side colorbar
    fig.update_layout(
        margin=dict(l=60, r=110, t=55, b=50),
        legend=dict(
            x=0.01, y=0.01, xanchor="left", yanchor="bottom",
            bgcolor="rgba(22,27,34,0.80)",
            bordercolor=BORDER, borderwidth=1,
        ),
    )
    fig.update_xaxes(range=[0, 1], title_text=labels.get(x_col, x_col))
    fig.update_yaxes(range=[0, 1], title_text=labels.get(y_col, y_col))
    return fig


# ─── Efficient Frontier ───────────────────────────────────────────────────────
def efficient_frontier_chart(ef_result: dict, title="Mean-Variance Efficient Frontier"):
    rand_df = ef_result["random_df"]
    front_df = ef_result["frontier_df"]
    mv      = ef_result["min_var"]
    ms      = ef_result["max_sharpe"]
    ew      = ef_result["equal_weight"]

    fig = go.Figure()

    # Random portfolios (colored by Sharpe)
    fig.add_trace(go.Scatter(
        x=rand_df["vol"] * 100, y=rand_df["return"] * 100,
        mode="markers",
        marker=dict(
            size=3.5,
            color=rand_df["sharpe"],
            colorscale="Viridis",
            showscale=True,
            opacity=0.45,
            colorbar=dict(title="Sharpe", tickfont=dict(color=SUBTEXT, size=9), x=1.01),
        ),
        name="Random Portfolios",
        hovertemplate="Vol: %{x:.1f}%<br>Return: %{y:.1f}%<br>Sharpe: %{marker.color:.2f}<extra></extra>",
    ))

    # Efficient frontier line
    if len(front_df) > 0:
        fig.add_trace(go.Scatter(
            x=front_df["vol"] * 100, y=front_df["return"] * 100,
            mode="lines",
            line=dict(color=GOLD, width=2.5),
            name="Efficient Frontier",
            hovertemplate="Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>",
        ))

    # Special portfolios
    specials = [
        (mv,  BLUE,  "Min Variance",  "square"),
        (ms,  GOLD,  "Max Sharpe",    "star"),
        (ew,  TEXT,  "Equal Weight",  "circle"),
    ]
    for pt, color, name, symbol in specials:
        fig.add_trace(go.Scatter(
            x=[pt["vol"] * 100], y=[pt["return"] * 100],
            mode="markers+text",
            marker=dict(size=14, color=color, symbol=symbol,
                        line=dict(color=TEXT, width=1.5)),
            text=[f"  {name}"],
            textposition="middle right",
            textfont=dict(color=color, size=11),
            name=f"{name} (Sharpe: {pt['sharpe']:.2f})",
            hovertemplate=f"<b>{name}</b><br>Vol: %{{x:.1f}}%<br>Return: %{{y:.1f}}%<br>Sharpe: {pt['sharpe']:.2f}<extra></extra>",
        ))

    _apply_theme(fig, title, height=480)
    fig.update_xaxes(ticksuffix="%", title_text="Annualized Volatility (%)")
    fig.update_yaxes(ticksuffix="%", title_text="Annualized Return (%)")
    return fig


# ─── FF Attribution Bar ───────────────────────────────────────────────────────
def ff_attribution_chart(result: dict, title="Fama-French 4-Factor Loadings"):
    factor_names = result["factor_names"]
    betas   = [result["betas"][f] for f in factor_names]
    tstats  = [result["t_stats"][f] for f in factor_names]
    se      = [abs(b / t) if t != 0 else 0 for b, t in zip(betas, tstats)]

    def sig_stars(t):
        t = abs(t)
        if t > 3.0:   return "***"
        if t > 2.0:   return "**"
        if t > 1.645: return "*"
        return ""

    labels = [f"{f} {sig_stars(t)}" for f, t in zip(factor_names, tstats)]
    colors = [GREEN if b >= 0 else RED for b in betas]

    fig = go.Figure()

    # Alpha bar (annualized)
    alpha_ann = result["alpha_ann"]
    alpha_t   = result["alpha_tstat"]
    fig.add_trace(go.Bar(
        x=[alpha_ann * 100],
        y=["Alpha (Ann.)"],
        orientation="h",
        marker_color=GOLD,
        error_x=dict(
            type="data", array=[abs(result["alpha_daily"] / alpha_t) * 252 * 100 if alpha_t != 0 else 0],
            color=MUTED, thickness=1.5
        ),
        name=f"Alpha {sig_stars(alpha_t)}",
        hovertemplate="<b>Alpha</b><br>%{x:.2f}% Ann.<br>t-stat: " + f"{alpha_t:.2f}<extra></extra>",
    ))

    # Factor beta bars
    fig.add_trace(go.Bar(
        x=betas,
        y=labels,
        orientation="h",
        marker_color=colors,
        error_x=dict(type="data", array=[s * 1.96 for s in se], color=MUTED, thickness=1.5),
        name="Factor Betas",
        hovertemplate="<b>%{y}</b><br>Beta: %{x:.3f}<extra></extra>",
    ))

    fig.add_vline(x=0, line=dict(color=MUTED, width=1))

    _apply_theme(fig, title, height=380, show_legend=False)
    fig.update_xaxes(title_text="Beta / Annualized Alpha (%)")
    fig.update_layout(barmode="group")
    return fig


# ─── CAPM Security Characteristic Line ───────────────────────────────────────
def capm_scatter_chart(port_daily: pd.Series, bench_daily: pd.Series,
                        attribution: dict,
                        title="Security Characteristic Line (CAPM)"):
    """
    Scatter of portfolio daily returns vs market (SPY) daily returns,
    with the OLS regression line (Security Characteristic Line) overlaid.

    Slope  = beta  (systematic / market sensitivity)
    Y-int  = alpha (skill / excess return not explained by market)
    R²     = fraction of portfolio variance explained by the market
    """
    idx   = port_daily.index.intersection(bench_daily.index)
    port  = (port_daily.loc[idx] * 100).values
    bench = (bench_daily.loc[idx] * 100).values

    beta            = attribution["betas"].get("Mkt-RF", 1.0)
    alpha_daily_pct = attribution["alpha_ann"] / 252 * 100   # daily alpha in %
    alpha_ann_pct   = attribution["alpha_ann"] * 100
    r2              = attribution["r_squared"]

    x_line = np.linspace(float(bench.min()), float(bench.max()), 200)
    y_scl  = alpha_daily_pct + beta * x_line   # regression (SCL)

    fig = go.Figure()

    # Daily return cloud
    fig.add_trace(go.Scatter(
        x=bench, y=port,
        mode="markers",
        name="Daily return pairs",
        marker=dict(color=BLUE, size=3.5, opacity=0.30,
                    line=dict(color="rgba(0,0,0,0)", width=0)),
        hovertemplate="Market: %{x:.3f}%<br>Portfolio: %{y:.3f}%<extra></extra>",
    ))

    # Security Characteristic Line
    fig.add_trace(go.Scatter(
        x=x_line, y=y_scl,
        mode="lines",
        name=f"SCL  (β = {beta:.2f},  α = {alpha_ann_pct:.1f}%/yr)",
        line=dict(color=GOLD, width=2.5),
    ))

    # β = 1 reference line
    fig.add_trace(go.Scatter(
        x=x_line, y=x_line,
        mode="lines",
        name="β = 1 reference",
        line=dict(color=MUTED, width=1, dash="dot"),
    ))

    # Stats annotation box
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.02, y=0.98, xanchor="left", yanchor="top",
        text=(
            f"<b>CAPM Regression</b><br>"
            f"Beta (β): {beta:.3f}<br>"
            f"Alpha (α): {alpha_ann_pct:.2f}%/yr<br>"
            f"R²: {r2:.1%}<br>"
            f"N = {attribution['n_obs']} obs"
        ),
        showarrow=False, align="left",
        bgcolor="rgba(22,27,34,0.85)",
        bordercolor=BORDER, borderwidth=1,
        font=dict(color=TEXT, size=11, family=FONT),
    )

    _apply_theme(fig, title, height=400)
    fig.update_xaxes(
        title_text="S&P 500 (SPY) Daily Return (%)",
        zeroline=True, zerolinecolor=GRID, zerolinewidth=1,
    )
    fig.update_yaxes(
        title_text="Portfolio Daily Return (%)",
        zeroline=True, zerolinecolor=GRID, zerolinewidth=1,
    )
    return fig


# ─── Portfolio Weights (Portfolio vs Benchmark) ──────────────────────────────
def portfolio_weights_chart(weights_df: pd.DataFrame, title="Optimized Portfolio Weights"):
    """
    weights_df: DataFrame with columns ['ticker', 'ew', 'max_sharpe', 'min_var']
    """
    top_n = weights_df.nlargest(20, "max_sharpe")
    fig = go.Figure()
    for col, name, color in [
        ("ew",        "Equal Weight",   SUBTEXT),
        ("min_var",   "Min Variance",   BLUE),
        ("max_sharpe","Max Sharpe",     GOLD),
    ]:
        if col in top_n.columns:
            fig.add_trace(go.Bar(
                x=top_n["ticker"], y=top_n[col] * 100,
                name=name, marker_color=color, opacity=0.85,
                hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:.2f}}%<extra></extra>",
            ))

    _apply_theme(fig, title, height=400)
    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Ticker")
    fig.update_yaxes(ticksuffix="%", title_text="Weight (%)")
    return fig
