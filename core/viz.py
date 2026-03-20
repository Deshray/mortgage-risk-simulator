"""
core/viz.py — Visualizations for the Canadian Mortgage Portfolio Risk Simulator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Optional

# ─────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────
BG      = "rgba(0,0,0,0)"
SURFACE = "#FDFAF4"
PAPER   = "#FFF8EE"
GRID    = "#E8DCC8"
TEXT    = "#1E1408"
MUTED   = "#7A5A30"
LABEL   = "#9A7A50"
GREEN   = "#2C6E3F"
RED     = "#8B1A1A"
AMBER   = "#C8860A"
BROWN   = "#6B3410"
ORANGE  = "#B85C10"
TEAL    = "#2C5F6E"


# ─────────────────────────────────────────────
# Axis / layout helpers — NO tickformat in base
# so there are no duplicate keyword conflicts
# ─────────────────────────────────────────────
def _xax(**kw) -> dict:
    return dict(gridcolor=GRID, linecolor=GRID, tickcolor=GRID,
                tickfont=dict(size=10, color=LABEL), zeroline=False,
                showgrid=True, **kw)


def _yax(**kw) -> dict:
    return dict(gridcolor=GRID, linecolor=GRID, tickcolor=GRID,
                tickfont=dict(size=10, color=LABEL), zeroline=False,
                showgrid=True, **kw)


def _layout(**kw) -> dict:
    return dict(
        paper_bgcolor=BG,
        plot_bgcolor=SURFACE,
        font=dict(family="Georgia, 'Times New Roman', serif",
                  color=MUTED, size=11),
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor=PAPER, bordercolor=GRID,
            font=dict(family="'Courier New', monospace", size=10, color=TEXT),
            namelength=-1,
        ),
        legend=dict(
            bgcolor="rgba(255,252,240,0.9)",
            bordercolor=GRID, borderwidth=1,
            font=dict(size=10, color=MUTED),
            x=0.01, y=0.99, xanchor="left", yanchor="top",
        ),
        **kw,
    )


def _title(text: str, size: int = 13) -> dict:
    return dict(
        text=text,
        font=dict(family="'Palatino Linotype', Palatino, 'Book Antiqua', serif",
                  size=size, color=TEXT),
        x=0.0, xanchor="left", pad=dict(l=4),
    )


# ─────────────────────────────────────────────
# 1. Loss distribution histogram
# ─────────────────────────────────────────────
def loss_distribution_chart(
    losses: np.ndarray,
    var95: float,
    var99: float,
    cvar95: float,
    title: str = "Simulated Portfolio Loss Distribution",
) -> go.Figure:
    fig = go.Figure()
    losses_m = losses / 1e6
    el = losses_m.mean()

    fig.add_trace(go.Histogram(
        x=losses_m,
        nbinsx=55,
        marker=dict(
            color=AMBER,
            line=dict(color=BROWN, width=0.4),
            opacity=0.78,
        ),
        name="Loss scenarios",
        hovertemplate="Loss: $%{x:.2f}M<br>Count: %{y}<extra></extra>",
    ))

    # EL line
    fig.add_vline(
        x=el, line_color=TEAL, line_width=1.8, line_dash="dot",
        annotation_text=f"  EL = ${el:.1f}M",
        annotation_font=dict(color=TEAL, size=9,
                             family="'Courier New', monospace"),
        annotation_position="top left",
    )

    # Risk lines
    for val, col, lbl, pos in [
        (var95 / 1e6,  ORANGE, f"VaR 95% = ${var95/1e6:.1f}M",  "top right"),
        (var99 / 1e6,  RED,    f"VaR 99% = ${var99/1e6:.1f}M",  "top right"),
        (cvar95 / 1e6, BROWN,  f"CVaR 95% = ${cvar95/1e6:.1f}M","top right"),
    ]:
        fig.add_vline(
            x=val, line_color=col, line_width=1.6, line_dash="dash",
            annotation_text=f"  {lbl}",
            annotation_font=dict(color=col, size=8,
                                 family="'Courier New', monospace"),
            annotation_position=pos,
        )

    # Tail shading
    fig.add_vrect(
        x0=var95 / 1e6, x1=losses_m.max() * 1.05,
        fillcolor="rgba(139,26,26,0.05)",
        layer="below", line_width=0,
    )

    fig.update_layout(
        **_layout(title=_title(title), height=390, bargap=0.04),
        xaxis=_xax(title="Portfolio Loss (CAD Millions)",
                   tickprefix="$", ticksuffix="M", tickformat=".1f"),
        yaxis=_yax(title="Number of Scenarios", tickformat=",d"),
    )
    return fig


# ─────────────────────────────────────────────
# 2. Rate sensitivity curve
# ─────────────────────────────────────────────
def rate_sensitivity_chart(
    df_sweep: pd.DataFrame,
    metric: str = "loss_rate",
    baseline_val: Optional[float] = None,
) -> go.Figure:
    label_map = {
        "loss_rate":         ("Portfolio Loss Rate vs Rate Shock",
                              "Loss Rate (%)", ".4f"),
        "mean_default_prob": ("Mean Default Probability vs Rate Shock",
                              "Default Probability (%)", ".4f"),
        "expected_loss":     ("Expected Loss vs Rate Shock",
                              "Expected Loss ($M)", ".2f"),
    }
    title_text, y_label, yfmt = label_map.get(metric, (metric, metric, ".3f"))

    raw = df_sweep[metric].values.copy()
    if metric in ("loss_rate", "mean_default_prob"):
        y  = raw * 100
        bv = baseline_val * 100 if baseline_val is not None else None
    elif metric == "expected_loss":
        y  = raw / 1e6
        bv = baseline_val / 1e6 if baseline_val is not None else None
    else:
        y  = raw
        bv = baseline_val

    x = df_sweep["delta_rate"].values * 100

    fig = go.Figure()

    # Fill under curve
    fig.add_trace(go.Scatter(
        x=x, y=y,
        fill="tozeroy", fillcolor="rgba(200,134,10,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))

    # Main curve
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines+markers",
        marker=dict(size=4, color=AMBER,
                    line=dict(color=BROWN, width=0.8)),
        line=dict(color=AMBER, width=2.5),
        name=y_label,
        hovertemplate=f"+%{{x:.2f}}pp shock → {y_label}: %{{y:{yfmt}}}<extra></extra>",
    ))

    if bv is not None:
        fig.add_hline(
            y=bv, line_dash="dot", line_color=GREEN, line_width=1.5,
            annotation_text=f"  Baseline: {bv:{yfmt}}",
            annotation_font=dict(color=GREEN, size=9,
                                 family="'Courier New', monospace"),
            annotation_position="top left",
        )

    # +100bps / +200bps reference marks
    for bps in [1.0, 2.0]:
        fig.add_vrect(
            x0=bps - 0.12, x1=bps + 0.12,
            fillcolor="rgba(139,69,19,0.05)",
            layer="below", line_width=0,
        )

    fig.update_layout(
        **_layout(title=_title(title_text), height=370),
        xaxis=_xax(title="Rate Shock (percentage points)",
                   ticksuffix="pp", tickformat=".1f"),
        yaxis=_yax(title=y_label, tickformat=yfmt),
    )
    return fig


# ─────────────────────────────────────────────
# 3. Rate paths fan chart
# ─────────────────────────────────────────────
def rate_paths_fan_chart(
    rate_paths: np.ndarray,
    current_rate: float,
    n_sample_paths: int = 60,
) -> go.Figure:
    n_q = rate_paths.shape[1]
    quarters = [f"Q+{i}" for i in range(n_q)]
    p5, p25, p50, p75, p95 = [
        np.percentile(rate_paths, p, axis=0) * 100
        for p in [5, 25, 50, 75, 95]
    ]

    fig = go.Figure()

    # Sample paths
    rng = np.random.default_rng(7)
    idx = rng.choice(len(rate_paths),
                     size=min(n_sample_paths, len(rate_paths)), replace=False)
    for i in idx:
        fig.add_trace(go.Scatter(
            x=quarters, y=rate_paths[i] * 100,
            mode="lines",
            line=dict(color="rgba(200,134,10,0.06)", width=0.8),
            showlegend=False, hoverinfo="skip",
        ))

    # 5–95 band
    fig.add_trace(go.Scatter(
        x=quarters + quarters[::-1],
        y=list(p95) + list(p5)[::-1],
        fill="toself", fillcolor="rgba(139,69,19,0.06)",
        line=dict(color="rgba(0,0,0,0)"),
        name="5–95th pct", hoverinfo="skip",
    ))

    # 25–75 band
    fig.add_trace(go.Scatter(
        x=quarters + quarters[::-1],
        y=list(p75) + list(p25)[::-1],
        fill="toself", fillcolor="rgba(200,134,10,0.14)",
        line=dict(color="rgba(0,0,0,0)"),
        name="25–75th pct", hoverinfo="skip",
    ))

    # P25 / P75 outlines (dashed)
    for pv, lbl in [(p25, "P25"), (p75, "P75")]:
        fig.add_trace(go.Scatter(
            x=quarters, y=pv,
            mode="lines",
            line=dict(color=AMBER, width=1, dash="dot"),
            showlegend=False,
            hovertemplate=f"{lbl}: %{{y:.2f}}%<extra></extra>",
        ))

    # Median
    fig.add_trace(go.Scatter(
        x=quarters, y=p50,
        mode="lines+markers",
        marker=dict(size=5, color=AMBER,
                    line=dict(color=BROWN, width=1)),
        line=dict(color=AMBER, width=2.5),
        name="Median path",
        hovertemplate="Quarter: %{x}<br>Median: %{y:.2f}%<extra></extra>",
    ))

    # Current rate
    fig.add_hline(
        y=current_rate * 100,
        line_dash="dash", line_color=GREEN, line_width=1.8,
        annotation_text=f"  Current BoC: {current_rate*100:.2f}%",
        annotation_font=dict(color=GREEN, size=9,
                             family="'Courier New', monospace"),
        annotation_position="top left",
    )

    fig.update_layout(
        **_layout(title=_title("Stochastic BoC Rate Paths — Vasicek Model"),
                  height=390),
        xaxis=_xax(title="Forecast Quarter"),
        yaxis=_yax(title="BoC Overnight Rate (%)",
                   tickformat=".2f", ticksuffix="%"),
    )
    return fig


# ─────────────────────────────────────────────
# 4. Portfolio breakdown — horizontal bars
# ─────────────────────────────────────────────
def portfolio_breakdown_chart(
    df: pd.DataFrame,
    group_col: str,
    metric: str = "default_prob",
    title: str = "",
) -> go.Figure:
    grouped = (df.groupby(group_col)[metric]
                 .agg(["mean", "std"])
                 .sort_values("mean", ascending=True))
    vals = grouped["mean"].values * 100
    errs = grouped["std"].values * 100
    cats = grouped.index.tolist()

    # Risk-scaled colours: green → amber → red
    norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
    colors = [
        f"rgba({int(44 + n*(180-44))},{int(110 - n*90)},{int(63 - n*63)},0.85)"
        for n in norm
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=cats, x=vals,
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color="rgba(60,30,10,0.3)", width=0.6),
        ),
        error_x=dict(type="data", array=errs, visible=True,
                     color="rgba(100,60,20,0.35)",
                     thickness=1.2, width=4),
        text=[f"{v:.3f}%" for v in vals],
        textposition="outside",
        textfont=dict(size=9, color=MUTED,
                      family="'Courier New', monospace"),
        hovertemplate="%{y}<br>Mean: %{x:.4f}%<extra></extra>",
        name="",
    ))

    fig.update_layout(
        **_layout(
            title=_title(title or f"Default Risk by {group_col.replace('_',' ').title()}"),
            height=max(300, len(cats) * 42),
            showlegend=False,
        ),
        xaxis=_xax(title="Avg Default Probability (%)",
                   tickformat=".3f", ticksuffix="%"),
        yaxis=_yax(tickformat=""),
    )
    return fig


# ─────────────────────────────────────────────
# 5. Quarterly loss chart
# ─────────────────────────────────────────────
def quarterly_loss_chart(quarterly_losses: np.ndarray) -> go.Figure:
    n_q = quarterly_losses.shape[1]
    q_labels = [f"Q+{i+1}" for i in range(n_q)]
    q_mean = quarterly_losses.mean(axis=0) / 1e6
    q_p25  = np.percentile(quarterly_losses, 25, axis=0) / 1e6
    q_p75  = np.percentile(quarterly_losses, 75, axis=0) / 1e6
    q_p95  = np.percentile(quarterly_losses, 95, axis=0) / 1e6

    fig = go.Figure()

    # IQR band
    fig.add_trace(go.Scatter(
        x=q_labels + q_labels[::-1],
        y=list(q_p75) + list(q_p25)[::-1],
        fill="toself", fillcolor="rgba(200,134,10,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="25–75th pct", hoverinfo="skip",
    ))

    # Bars — slight gradient effect
    fig.add_trace(go.Bar(
        x=q_labels, y=q_mean,
        name="Mean loss",
        marker=dict(
            color=[f"rgba({int(180+i*3)},{int(100-i*4)},10,0.82)"
                   for i in range(n_q)],
            line=dict(color=BROWN, width=0.5),
        ),
        hovertemplate="Quarter %{x}<br>Mean: $%{y:.2f}M<extra></extra>",
    ))

    # P95 line
    fig.add_trace(go.Scatter(
        x=q_labels, y=q_p95,
        mode="lines+markers",
        line=dict(color=RED, width=2, dash="dash"),
        marker=dict(size=6, color=RED, symbol="diamond",
                    line=dict(color=BROWN, width=1)),
        name="P95 loss",
        hovertemplate="Quarter %{x}<br>P95: $%{y:.2f}M<extra></extra>",
    ))

    fig.update_layout(
        **_layout(title=_title("Expected Loss by Quarter"), height=330,
                  bargap=0.25),
        xaxis=_xax(title="Forecast Quarter"),
        yaxis=_yax(title="Loss (CAD Millions)",
                   tickprefix="$", ticksuffix="M", tickformat=".2f"),
    )
    return fig


# ─────────────────────────────────────────────
# 6. Risk metrics table
# ─────────────────────────────────────────────
def risk_metrics_table(metrics: Dict[str, float]) -> go.Figure:
    rows = [
        ("Expected Loss (EL)",           f"${metrics['EL']/1e6:.2f}M",         "normal"),
        ("EL / Total Exposure",          f"{metrics['EL_rate']*100:.3f}%",      "normal"),
        ("VaR 95%  (1-yr horizon)",      f"${metrics['VaR_95']/1e6:.2f}M",      "normal"),
        ("VaR 99%  (1-yr horizon)",      f"${metrics['VaR_99']/1e6:.2f}M",      "normal"),
        ("CVaR 95%  (Exp. Shortfall)",   f"${metrics['CVaR_95']/1e6:.2f}M",     "highlight"),
        ("CVaR 99%",                     f"${metrics['CVaR_99']/1e6:.2f}M",     "highlight"),
        ("Max Simulated Loss",           f"${metrics['max_loss']/1e6:.2f}M",    "muted"),
        ("Total Exposure",               f"${metrics['exposure']/1e9:.2f}B",    "muted"),
        ("Mean Defaults / Scenario",     f"{metrics['mean_defaults']:.0f}",     "muted"),
        ("P99 Defaults / Scenario",      f"{metrics['p99_defaults']:.0f}",      "muted"),
    ]
    labels = [r[0] for r in rows]
    values = [r[1] for r in rows]
    kinds  = [r[2] for r in rows]

    fill  = [("#F5E8D0" if k == "highlight"
               else PAPER if k == "muted"
               else SURFACE) for k in kinds]
    vcols = [(BROWN if k == "highlight"
               else LABEL if k == "muted"
               else TEXT) for k in kinds]

    fig = go.Figure(go.Table(
        columnwidth=[2.2, 1],
        header=dict(
            values=["<b>Risk Metric</b>", "<b>Value</b>"],
            fill_color="#1E1408",
            font=dict(color=AMBER,
                      family="'Palatino Linotype', Palatino, serif",
                      size=12),
            align=["left", "right"],
            height=34,
            line=dict(color=BROWN, width=1),
        ),
        cells=dict(
            values=[labels, values],
            fill_color=[fill, fill],
            font=dict(
                color=[["#3A2A14"] * len(labels), vcols],
                family="'Courier New', monospace",
                size=11,
            ),
            align=["left", "right"],
            height=30,
            line=dict(color=GRID, width=0.5),
        ),
    ))
    fig.update_layout(
        paper_bgcolor=BG,
        margin=dict(l=0, r=0, t=0, b=0),
        height=len(rows) * 32 + 46,
    )
    return fig


# ─────────────────────────────────────────────
# 7. LTV vs Default probability scatter
# ─────────────────────────────────────────────
def ltv_default_scatter(df: pd.DataFrame, sample_n: int = 2000) -> go.Figure:
    s = df.sample(min(sample_n, len(df)), random_state=42)
    nd = s[s["defaulted"] == 0]
    d  = s[s["defaulted"] == 1]

    fig = go.Figure()

    # Non-defaulted — coloured by credit score
    fig.add_trace(go.Scatter(
        x=nd["ltv"] * 100,
        y=nd["default_prob"] * 100,
        mode="markers",
        marker=dict(
            color=nd["credit_score"],
            colorscale=[[0, RED], [0.5, AMBER], [1, GREEN]],
            size=4, opacity=0.4,
            colorbar=dict(
                title=dict(text="Credit Score",
                           font=dict(size=9, color=MUTED)),
                tickfont=dict(size=8, color=MUTED),
                thickness=10, len=0.55,
                outlinecolor=GRID, outlinewidth=0.5,
            ),
            line=dict(width=0),
        ),
        name="No Default",
        hovertemplate=(
            "LTV: %{x:.1f}%<br>"
            "P(default): %{y:.3f}%<br>"
            "Credit: %{marker.color:.0f}"
            "<extra></extra>"
        ),
    ))

    # Defaulted
    fig.add_trace(go.Scatter(
        x=d["ltv"] * 100,
        y=d["default_prob"] * 100,
        mode="markers",
        marker=dict(
            color=RED, size=5, opacity=0.7,
            symbol="x",
            line=dict(color=BROWN, width=0.5),
        ),
        name="Defaulted",
        hovertemplate=(
            "LTV: %{x:.1f}%<br>"
            "P(default): %{y:.3f}%"
            "<extra></extra>"
        ),
    ))

    # 80% threshold
    fig.add_vline(
        x=80, line_dash="dot", line_color=MUTED, line_width=1.2,
        annotation_text="  80% LTV",
        annotation_font=dict(size=8, color=MUTED,
                             family="'Courier New', monospace"),
        annotation_position="top right",
    )

    fig.update_layout(
        **_layout(
            title=_title("LTV vs Default Probability  —  coloured by credit score"),
            height=375,
        ),
        xaxis=_xax(title="Loan-to-Value Ratio (%)",
                   ticksuffix="%", tickformat=".0f"),
        yaxis=_yax(title="Default Probability (%)",
                   ticksuffix="%", tickformat=".2f"),
    )
    return fig