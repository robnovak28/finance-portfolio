"""
charts.py
---------
Plotly chart builders for the LBO Streamlit dashboard.
All charts share a consistent institutional dark theme.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Global design tokens
# ---------------------------------------------------------------------------
COLORS = {
    "primary":   "#C9A84C",   # Gold
    "secondary": "#4ECDC4",   # Teal
    "accent":    "#FF6B6B",   # Coral
    "green":     "#27AE60",
    "yellow":    "#F4C842",
    "red":       "#E74C3C",
    "bg":        "#0E1117",
    "panel":     "#161B22",   # Slightly lighter for depth
    "panel2":    "#1C2230",   # Second-level panel
    "grid":      "#252D3A",
    "text":      "#E8EAF0",
    "subtext":   "#8A9BB0",
    "bear":      "#E74C3C",
    "base":      "#C9A84C",
    "bull":      "#27AE60",
    "border":    "#2D3748",
}

TRANCHE_COLORS = [
    "#3B82F6",  # Blue   – Revolver
    "#10B981",  # Emerald – TLA
    "#F59E0B",  # Amber   – TLB
    "#8B5CF6",  # Violet  – Sr Notes
    "#EF4444",  # Red     – HY Notes
]

FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

LAYOUT_BASE = dict(
    paper_bgcolor = COLORS["bg"],
    plot_bgcolor  = COLORS["panel"],
    font          = dict(family=FONT, size=12, color=COLORS["text"]),
    margin        = dict(l=60, r=40, t=60, b=50),
    hoverlabel    = dict(
        bgcolor    = COLORS["panel2"],
        bordercolor= COLORS["border"],
        font_size  = 12,
        font_color = COLORS["text"],
        font_family= FONT,
    ),
    legend = dict(
        bgcolor      = COLORS["panel2"],
        bordercolor  = COLORS["border"],
        borderwidth  = 1,
        font_size    = 11,
        orientation  = "h",
        yanchor      = "bottom",
        y            = 1.02,
        xanchor      = "right",
        x            = 1,
    ),
)

AXIS_STYLE = dict(
    gridcolor      = COLORS["grid"],
    gridwidth      = 1,
    zerolinecolor  = COLORS["border"],
    zerolinewidth  = 1,
    linecolor      = COLORS["border"],
    linewidth      = 1,
    showline       = True,
    tickfont       = dict(family=FONT, size=11, color=COLORS["subtext"]),
    title_font     = dict(family=FONT, size=12, color=COLORS["subtext"]),
)

TITLE_STYLE = dict(font=dict(family=FONT, size=15, color=COLORS["primary"]),
                   x=0, xanchor="left", pad=dict(l=0))


def _base(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    """Apply shared layout to a figure."""
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text=title, **TITLE_STYLE),
        height=height,
    )
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    return fig


# ---------------------------------------------------------------------------
# Historical Financials
# ---------------------------------------------------------------------------

def historical_financials_chart(hist_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=hist_df["Period"], y=hist_df["Revenue ($M)"],
        name="Revenue ($M)",
        marker=dict(color=COLORS["secondary"], opacity=0.55, line_width=0),
        hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.0f}M<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Bar(
        x=hist_df["Period"], y=hist_df["EBITDA ($M)"],
        name="EBITDA ($M)",
        marker=dict(color=COLORS["primary"], opacity=0.9, line_width=0),
        hovertemplate="<b>%{x}</b><br>EBITDA: $%{y:,.0f}M<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=hist_df["Period"], y=hist_df["EBITDA Margin"],
        name="EBITDA Margin",
        mode="lines+markers+text",
        text=[f"{v:.1%}" for v in hist_df["EBITDA Margin"]],
        textposition="top center",
        textfont=dict(size=10, color=COLORS["accent"]),
        line=dict(color=COLORS["accent"], width=2.5),
        marker=dict(size=8, symbol="circle"),
        hovertemplate="<b>%{x}</b><br>EBITDA Margin: %{y:.1%}<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(**LAYOUT_BASE, height=380, barmode="group",
                      title=dict(text="Historical Financials — Dollar General Corporation",
                                 **TITLE_STYLE))
    fig.update_yaxes(title_text="USD Millions", secondary_y=False, **AXIS_STYLE)
    fig.update_yaxes(title_text="EBITDA Margin", tickformat=".1%",
                     secondary_y=True, showgrid=False,
                     tickfont=dict(size=11, color=COLORS["subtext"]))
    return fig


# ---------------------------------------------------------------------------
# Revenue & EBITDA Projection
# ---------------------------------------------------------------------------

def revenue_ebitda_projection(is_df: pd.DataFrame) -> go.Figure:
    years   = list(is_df.columns)
    revenue = [is_df.loc["Revenue",      c] for c in years]
    ebitda  = [is_df.loc["EBITDA",       c] for c in years]
    margins = [is_df.loc["EBITDA Margin",c] for c in years]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=years, y=revenue, name="Revenue ($M)",
        marker=dict(color=COLORS["secondary"], opacity=0.45, line_width=0),
        hovertemplate="<b>%{x}</b><br>Revenue: $%{y:,.0f}M<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Bar(
        x=years, y=ebitda, name="EBITDA ($M)",
        marker=dict(color=COLORS["primary"], opacity=0.9, line_width=0),
        text=[f"${v:,.0f}" for v in ebitda],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["primary"]),
        hovertemplate="<b>%{x}</b><br>EBITDA: $%{y:,.0f}M<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=years, y=margins, name="EBITDA Margin",
        mode="lines+markers+text",
        text=[f"{v:.1%}" for v in margins],
        textposition="top center",
        textfont=dict(size=10, color=COLORS["accent"]),
        line=dict(color=COLORS["accent"], width=2.5),
        marker=dict(size=9, symbol="circle", line=dict(width=2, color=COLORS["bg"])),
        hovertemplate="<b>%{x}</b><br>EBITDA Margin: %{y:.1%}<extra></extra>",
    ), secondary_y=True)

    fig.update_layout(**LAYOUT_BASE, barmode="group", height=420,
                      title=dict(text="Projected Revenue & EBITDA", **TITLE_STYLE))
    fig.update_yaxes(title_text="USD Millions", gridcolor=COLORS["grid"],
                     **{k: v for k, v in AXIS_STYLE.items() if k != "gridcolor"},
                     secondary_y=False)
    fig.update_yaxes(title_text="EBITDA Margin", tickformat=".1%",
                     secondary_y=True, showgrid=False,
                     tickfont=dict(size=11, color=COLORS["subtext"]))
    return fig


# ---------------------------------------------------------------------------
# Debt Waterfall (stacked bar)
# ---------------------------------------------------------------------------

def debt_waterfall_chart(waterfall_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, tranche in enumerate(waterfall_df.columns):
        color = TRANCHE_COLORS[i % len(TRANCHE_COLORS)]
        fig.add_trace(go.Bar(
            name=tranche,
            x=waterfall_df.index,
            y=waterfall_df[tranche],
            marker=dict(color=color, line_width=0, opacity=0.88),
            hovertemplate=f"<b>{tranche}</b><br>%{{x}}: $%{{y:,.0f}}M<extra></extra>",
        ))

    fig.update_layout(**LAYOUT_BASE, barmode="stack", height=420,
                      title=dict(text="Debt Paydown by Tranche", **TITLE_STYLE))
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(title_text="Debt Balance ($M)", **AXIS_STYLE,
                     tickformat="$,.0f")
    return fig


# ---------------------------------------------------------------------------
# Leverage & Coverage
# ---------------------------------------------------------------------------

def leverage_coverage_chart(credit_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=credit_df["Year"].astype(str), y=credit_df["Gross Leverage (x)"],
        name="Gross Leverage",
        mode="lines+markers",
        line=dict(color=COLORS["accent"], width=2.5),
        marker=dict(size=9, symbol="circle", line=dict(width=2, color=COLORS["bg"])),
        hovertemplate="<b>Year %{x}</b><br>Gross Leverage: %{y:.2f}x<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=credit_df["Year"].astype(str), y=credit_df["Net Leverage (x)"],
        name="Net Leverage",
        mode="lines+markers",
        line=dict(color=COLORS["yellow"], width=2, dash="dot"),
        marker=dict(size=7),
        hovertemplate="<b>Year %{x}</b><br>Net Leverage: %{y:.2f}x<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=credit_df["Year"].astype(str), y=credit_df["Interest Coverage (x)"],
        name="Interest Coverage",
        mode="lines+markers",
        line=dict(color=COLORS["green"], width=2.5),
        marker=dict(size=9, symbol="diamond", line=dict(width=2, color=COLORS["bg"])),
        hovertemplate="<b>Year %{x}</b><br>Coverage: %{y:.2f}x<extra></extra>",
    ), secondary_y=True)

    # Covenant reference line (max 6.5x leverage)
    fig.add_hline(y=6.5, line_dash="dash", line_color=COLORS["red"],
                  line_width=1, opacity=0.4,
                  annotation_text="Max Leverage Covenant",
                  annotation_font_color=COLORS["red"],
                  annotation_font_size=10,
                  secondary_y=False)

    fig.update_layout(**LAYOUT_BASE, height=440,
                      title=dict(text="Credit Metrics — Leverage & Coverage Trajectory",
                                 **TITLE_STYLE))
    fig.update_yaxes(title_text="Leverage (x)", secondary_y=False, **AXIS_STYLE)
    fig.update_yaxes(title_text="Interest Coverage (x)", secondary_y=True,
                     showgrid=False,
                     tickfont=dict(size=11, color=COLORS["subtext"]))
    return fig


# ---------------------------------------------------------------------------
# Returns Attribution Bridge  (uses native go.Waterfall)
# ---------------------------------------------------------------------------

def returns_bridge_chart(bridge: dict) -> go.Figure:
    entry_eq    = bridge["Entry Equity Value ($M)"]
    ebitda_g    = bridge["EBITDA Growth ($M)"]
    mult_exp    = bridge["Multiple Expansion / (Contraction) ($M)"]
    delev       = bridge["Deleveraging ($M)"]
    exit_eq     = bridge["Exit Equity Value ($M)"]

    fig = go.Figure(go.Waterfall(
        name        = "Value Bridge",
        orientation = "v",
        measure     = ["absolute", "relative", "relative", "relative", "total"],
        x           = ["Entry Equity", "EBITDA Growth", "Multiple Exp.", "Deleveraging", "Exit Equity"],
        y           = [entry_eq, ebitda_g, mult_exp, delev, 0],
        text        = [
            f"${entry_eq:,.0f}M",
            f"+${ebitda_g:,.0f}M" if ebitda_g >= 0 else f"-${abs(ebitda_g):,.0f}M",
            f"+${mult_exp:,.0f}M" if mult_exp >= 0 else f"${mult_exp:,.0f}M",
            f"+${delev:,.0f}M",
            f"${exit_eq:,.0f}M",
        ],
        textposition  = "outside",
        textfont      = dict(size=11, family=FONT),
        connector     = dict(line=dict(color=COLORS["border"], width=1.5, dash="dot")),
        increasing    = dict(marker=dict(color=COLORS["green"],   line_width=0)),
        decreasing    = dict(marker=dict(color=COLORS["red"],     line_width=0)),
        totals        = dict(marker=dict(color=COLORS["primary"], line_width=0)),
        hovertemplate = "<b>%{x}</b><br>Value: %{text}<extra></extra>",
    ))

    moic = exit_eq / entry_eq if entry_eq > 0 else 0
    fig.update_layout(
        **LAYOUT_BASE, height=460, showlegend=False,
        title=dict(text=f"Returns Attribution Bridge  |  MOIC {moic:.2f}x",
                   **TITLE_STYLE),
    )
    fig.update_yaxes(title_text="Equity Value ($M)", tickformat="$,.0f", **AXIS_STYLE)
    fig.update_xaxes(**AXIS_STYLE)
    return fig


# ---------------------------------------------------------------------------
# Scenario Comparison
# ---------------------------------------------------------------------------

def scenario_comparison_chart(scenario_results: dict) -> go.Figure:
    names  = list(scenario_results.keys())
    irrs   = [scenario_results[n]["returns"]["irr"]  for n in names]
    moics  = [scenario_results[n]["returns"]["moic"] for n in names]
    colors_map = {"Bear": COLORS["bear"], "Base": COLORS["base"], "Bull": COLORS["bull"]}
    bar_colors = [colors_map.get(n, COLORS["primary"]) for n in names]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["IRR by Scenario", "MOIC by Scenario"],
        horizontal_spacing=0.12,
    )

    fig.add_trace(go.Bar(
        x=names, y=irrs,
        marker=dict(color=bar_colors, line_width=0, opacity=0.9),
        text=[f"{v:.1%}" if not np.isnan(v) else "N/A" for v in irrs],
        textposition="outside",
        textfont=dict(size=12, family=FONT, color=COLORS["text"]),
        hovertemplate="<b>%{x}</b><br>IRR: %{text}<extra></extra>",
        name="IRR",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=names, y=moics,
        marker=dict(color=bar_colors, line_width=0, opacity=0.9),
        text=[f"{v:.2f}x" for v in moics],
        textposition="outside",
        textfont=dict(size=12, family=FONT, color=COLORS["text"]),
        hovertemplate="<b>%{x}</b><br>MOIC: %{text}<extra></extra>",
        name="MOIC",
    ), row=1, col=2)

    # Hurdle rate reference line
    fig.add_hline(y=0.20, line_dash="dash", line_color=COLORS["yellow"],
                  line_width=1, opacity=0.6,
                  annotation_text="20% Hurdle",
                  annotation_font_color=COLORS["yellow"],
                  annotation_font_size=10,
                  row=1, col=1)

    fig.update_layout(
        **LAYOUT_BASE, height=420, showlegend=False,
        title=dict(text="Bull / Base / Bear — Return Comparison", **TITLE_STYLE),
    )
    for style_dict in [{"row": 1, "col": 1}, {"row": 1, "col": 2}]:
        fig.update_xaxes(**AXIS_STYLE, **style_dict)
        fig.update_yaxes(**AXIS_STYLE, **style_dict)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(tickformat=".2f", row=1, col=2)
    for ann in fig.layout.annotations:
        ann.font.color = COLORS["subtext"]
        ann.font.size  = 12
    return fig


# ---------------------------------------------------------------------------
# Scenario EBITDA lines
# ---------------------------------------------------------------------------

def scenario_ebitda_chart(scenario_results: dict) -> go.Figure:
    colors_map = {"Bear": COLORS["bear"], "Base": COLORS["base"], "Bull": COLORS["bull"]}
    dash_map   = {"Bear": "dot", "Base": "solid", "Bull": "solid"}
    width_map  = {"Bear": 2.0, "Base": 2.5, "Bull": 2.0}

    fig = go.Figure()
    for name, res in scenario_results.items():
        is_df  = res["is_df"]
        years  = list(is_df.columns)
        ebitda = [is_df.loc["EBITDA", c] for c in years]
        color  = colors_map.get(name, COLORS["primary"])

        fig.add_trace(go.Scatter(
            x=years, y=ebitda, name=name,
            mode="lines+markers",
            line=dict(color=color, width=width_map.get(name, 2),
                      dash=dash_map.get(name, "solid")),
            marker=dict(size=8, symbol="circle",
                        line=dict(width=2, color=COLORS["bg"])),
            fill="tozeroy" if name == "Bull" else None,
            fillcolor="rgba(39,174,96,0.06)" if name == "Bull" else None,
            hovertemplate=f"<b>{name} — %{{x}}</b><br>EBITDA: $%{{y:,.0f}}M<extra></extra>",
        ))

    fig.update_layout(
        **LAYOUT_BASE, height=400,
        title=dict(text="EBITDA Trajectory by Scenario", **TITLE_STYLE),
    )
    fig.update_yaxes(title_text="EBITDA ($M)", tickformat="$,.0f", **AXIS_STYLE)
    fig.update_xaxes(**AXIS_STYLE)
    return fig


# ---------------------------------------------------------------------------
# Monte Carlo — IRR Histogram
# ---------------------------------------------------------------------------

def monte_carlo_irr_histogram(raw_df: pd.DataFrame) -> go.Figure:
    irr_vals = raw_df["IRR"].dropna()
    p10  = np.percentile(irr_vals, 10)
    p50  = np.percentile(irr_vals, 50)
    p90  = np.percentile(irr_vals, 90)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=irr_vals, nbinsx=70,
        marker=dict(color=COLORS["secondary"], opacity=0.7, line_width=0),
        name="Simulated IRR",
        histnorm="probability density",
        hovertemplate="IRR bin: %{x:.1%}<br>Density: %{y:.4f}<extra></extra>",
    ))

    for val, color, label, pos in [
        (p10,  COLORS["red"],     f"P10: {p10:.1%}",    "top right"),
        (p50,  COLORS["primary"], f"Median: {p50:.1%}", "top right"),
        (p90,  COLORS["green"],   f"P90: {p90:.1%}",    "top left"),
        (0.20, COLORS["yellow"],  "20% Hurdle",          "top right"),
    ]:
        fig.add_vline(
            x=val, line_dash="dash" if val != 0.20 else "dot",
            line_color=color, line_width=1.5,
            annotation_text=label,
            annotation_font_color=color,
            annotation_font_size=10,
            annotation_position=pos,
        )

    fig.update_layout(
        **LAYOUT_BASE, height=420, showlegend=False,
        title=dict(text=f"IRR Distribution  ({len(irr_vals):,} Simulations)", **TITLE_STYLE),
    )
    fig.update_xaxes(title_text="IRR", tickformat=".0%", **AXIS_STYLE)
    fig.update_yaxes(title_text="Probability Density", **AXIS_STYLE)
    return fig


def monte_carlo_scatter(raw_df: pd.DataFrame) -> go.Figure:
    sample = raw_df.sample(min(2000, len(raw_df)), random_state=42)

    fig = px.scatter(
        sample,
        x="Exit EV/EBITDA", y="IRR",
        color="Exit EBITDA Margin",
        color_continuous_scale="Plasma",
        opacity=0.45,
        labels={"Exit EV/EBITDA": "Exit Multiple (x)", "IRR": "IRR"},
        hover_data={"Exit EV/EBITDA": ":.1f", "IRR": ":.1%", "Exit EBITDA Margin": ":.1%"},
    )
    fig.add_hline(y=0.20, line_dash="dash", line_color=COLORS["yellow"],
                  line_width=1.5, opacity=0.7,
                  annotation_text="20% Hurdle",
                  annotation_font_color=COLORS["yellow"],
                  annotation_font_size=10)

    fig.update_layout(
        **LAYOUT_BASE, height=420,
        title=dict(text="Exit Multiple vs. IRR (Monte Carlo Sample)", **TITLE_STYLE),
        coloraxis_colorbar=dict(
            title="Exit EBITDA Margin",
            tickformat=".0%",
            title_font=dict(size=11, color=COLORS["subtext"]),
            tickfont=dict(size=10, color=COLORS["subtext"]),
            bgcolor=COLORS["panel2"],
            bordercolor=COLORS["border"],
        ),
    )
    fig.update_yaxes(tickformat=".0%", **AXIS_STYLE)
    fig.update_xaxes(**AXIS_STYLE)
    return fig


# ---------------------------------------------------------------------------
# Free Cash Flow Summary
# ---------------------------------------------------------------------------

def fcf_waterfall_chart(cfs_df: pd.DataFrame) -> go.Figure:
    years  = list(cfs_df.columns)
    ocf    = [cfs_df.loc["Operating CF",   c] for c in years]
    capex  = [abs(cfs_df.loc["(-) CapEx",  c]) for c in years]
    fcf    = [cfs_df.loc["Free Cash Flow", c] for c in years]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=years, y=ocf,
        name="Operating CF",
        marker=dict(color=COLORS["green"], opacity=0.75, line_width=0),
        hovertemplate="<b>%{x}</b><br>Operating CF: $%{y:,.0f}M<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        x=years, y=[-v for v in capex],
        name="CapEx",
        marker=dict(color=COLORS["red"], opacity=0.75, line_width=0),
        hovertemplate="<b>%{x}</b><br>CapEx: $%{y:,.0f}M<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=years, y=fcf,
        name="Free Cash Flow",
        mode="lines+markers+text",
        text=[f"${v:,.0f}" for v in fcf],
        textposition="top center",
        textfont=dict(size=10, color=COLORS["primary"]),
        line=dict(color=COLORS["primary"], width=2.5),
        marker=dict(size=9, symbol="circle",
                    line=dict(width=2, color=COLORS["bg"])),
        hovertemplate="<b>%{x}</b><br>FCF: $%{y:,.0f}M<extra></extra>",
    ))

    fig.update_layout(
        **LAYOUT_BASE, barmode="relative", height=420,
        title=dict(text="Free Cash Flow Bridge", **TITLE_STYLE),
    )
    fig.update_yaxes(title_text="$M", tickformat="$,.0f", **AXIS_STYLE)
    fig.update_xaxes(**AXIS_STYLE)
    return fig
