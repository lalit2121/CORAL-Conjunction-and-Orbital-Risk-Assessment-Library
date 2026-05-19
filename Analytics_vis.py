"""
CASAS – Analytics & Visualisation
===================================
Plotly-based visualisations for CDM analysis, risk trends,
covariance ellipses, close-approach geometry, and space weather correlation.
"""

from __future__ import annotations

import math
import json
from typing import List, Dict, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

RISK_COLORS = {
    "CRITICAL":   "#FF0000",
    "HIGH":       "#FF6600",
    "MEDIUM":     "#FFAA00",
    "LOW":        "#FFE066",
    "NEGLIGIBLE": "#00CC66",
}

RISK_ORDER = ["NEGLIGIBLE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]


# ---------------------------------------------------------------------------
# KPI Cards
# ---------------------------------------------------------------------------

def kpi_cards(stats: Dict) -> str:
    """Return an HTML snippet with KPI cards (for use in Streamlit html())."""
    cards = [
        ("Total CDM Records",   stats.get("total_records", 0),  "#3B82F6"),
        ("Unique Events",       stats.get("total_events", 0),   "#8B5CF6"),
        ("Critical Events",     stats.get("critical_events", 0), "#EF4444"),
        ("High-Risk Events",    stats.get("high_events", 0),    "#F97316"),
        ("Unacked Alerts",      stats.get("unacked_alerts", 0), "#F59E0B"),
    ]
    html = '<div style="display:flex;gap:16px;flex-wrap:wrap;">'
    for label, value, color in cards:
        html += f"""
        <div style="background:{color}15;border-left:4px solid {color};
                    padding:16px 20px;border-radius:8px;min-width:150px;flex:1;">
            <div style="font-size:2em;font-weight:700;color:{color};">{value}</div>
            <div style="font-size:0.85em;color:#888;margin-top:4px;">{label}</div>
        </div>"""
    html += '</div>'
    return html


# ---------------------------------------------------------------------------
# Risk Distribution
# ---------------------------------------------------------------------------

def plot_risk_distribution(rows: List[Dict]) -> go.Figure:
    """Donut chart of event count by risk level."""
    counts = {level: 0 for level in RISK_ORDER}
    for r in rows:
        lv = r.get("risk_level", "NEGLIGIBLE")
        counts[lv] = counts.get(lv, 0) + 1

    labels = [k for k in RISK_ORDER if counts[k] > 0]
    values = [counts[k] for k in labels]
    colors = [RISK_COLORS[k] for k in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        marker_colors=colors,
        hole=0.55,
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        title="Risk Level Distribution",
        legend=dict(orientation="h"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0"),
    )
    return fig


# ---------------------------------------------------------------------------
# Risk Trend
# ---------------------------------------------------------------------------

def plot_risk_trend(trend_rows: List[Dict], event_id: int) -> go.Figure:
    """
    Dual-axis chart: log10(Pc) and miss distance vs time-to-TCA.
    """
    if not trend_rows:
        return go.Figure().update_layout(title="No data")

    tcas  = [r["time_to_tca_days"] * 24 for r in trend_rows]   # hours
    risks = [r.get("risk", -99) for r in trend_rows]
    miss  = [r.get("miss_distance_m", 0) / 1000.0 for r in trend_rows]  # km
    mah   = [r.get("mahalanobis_distance", 0) for r in trend_rows]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("log₁₀(Pc) Evolution", "Miss Distance (km)"),
        vertical_spacing=0.08,
    )

    # Risk lines
    fig.add_trace(go.Scatter(
        x=tcas, y=risks,
        mode="lines+markers",
        name="log₁₀(Pc)",
        line=dict(color="#EF4444", width=2),
        marker=dict(size=8),
    ), row=1, col=1)

    # Threshold lines
    for lvl, val, color in [
        ("CRITICAL (-4)", -4, "#FF0000"),
        ("HIGH (-5)", -5, "#FF6600"),
        ("MEDIUM (-6)", -6, "#FFAA00"),
    ]:
        fig.add_hline(y=val, line_dash="dash", line_color=color,
                      annotation_text=lvl, row=1, col=1)

    # Miss distance
    fig.add_trace(go.Scatter(
        x=tcas, y=miss,
        mode="lines+markers",
        name="Miss Distance (km)",
        line=dict(color="#3B82F6", width=2),
        marker=dict(size=8),
    ), row=2, col=1)

    fig.update_xaxes(title_text="Time to TCA (hours)", row=2, col=1, autorange="reversed")
    fig.update_yaxes(title_text="log₁₀(Pc)", row=1, col=1)
    fig.update_yaxes(title_text="Miss Distance (km)", row=2, col=1)
    fig.update_layout(
        title=f"Risk Trend – Event {event_id}",
        height=550,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,17,27,0.8)",
        font=dict(color="#E0E0E0"),
        legend=dict(orientation="h", y=-0.12),
    )
    return fig


# ---------------------------------------------------------------------------
# Close-Approach Geometry
# ---------------------------------------------------------------------------

def plot_close_approach_geometry(rows: List[Dict]) -> go.Figure:
    """
    3-D RTN scatter of relative position vectors across CDM series.
    """
    if not rows:
        return go.Figure()

    r_vals = [row.get("rel_pos_r", 0) / 1000.0 for row in rows]
    t_vals = [row.get("rel_pos_t", 0) / 1000.0 for row in rows]
    n_vals = [row.get("rel_pos_n", 0) / 1000.0 for row in rows]
    tca_h  = [row.get("time_to_tca_days", 0) * 24 for row in rows]
    risk   = [row.get("risk", -99) for row in rows]

    fig = go.Figure(go.Scatter3d(
        x=r_vals, y=t_vals, z=n_vals,
        mode="markers+lines",
        marker=dict(
            size=8,
            color=risk,
            colorscale="RdYlGn_r",
            colorbar=dict(title="log₁₀(Pc)"),
            showscale=True,
        ),
        text=[f"TCA-{h:.1f}h" for h in tca_h],
        hovertemplate="R: %{x:.3f} km<br>T: %{y:.3f} km<br>N: %{z:.3f} km<extra></extra>",
        line=dict(color="#60A5FA", width=2),
    ))

    # Collision sphere
    theta = np.linspace(0, 2*np.pi, 40)
    phi   = np.linspace(0, np.pi, 20)
    R_hbr = 0.005  # km
    xs = R_hbr * np.outer(np.cos(theta), np.sin(phi))
    ys = R_hbr * np.outer(np.sin(theta), np.sin(phi))
    zs = R_hbr * np.outer(np.ones(40), np.cos(phi))
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        opacity=0.3,
        colorscale=[[0,"red"],[1,"red"]],
        showscale=False,
        name="HBR Sphere",
    ))

    fig.update_layout(
        title="RTN Relative Position Evolution",
        scene=dict(
            xaxis_title="R (km)",
            yaxis_title="T (km)",
            zaxis_title="N (km)",
            bgcolor="rgba(17,17,27,0.9)",
        ),
        height=550,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E0E0E0"),
    )
    return fig


# ---------------------------------------------------------------------------
# Covariance Ellipse
# ---------------------------------------------------------------------------
def plot_covariance_ellipse_2d(rows: List[Dict], plane: str = "RT") -> go.Figure:
    """
    Draw 1,2,3-sigma covariance ellipses in chosen RTN plane.
    """
    plane_map = {
        "RT": ("t_sigma_r", "t_sigma_t", "R", "T"),
        "RN": ("t_sigma_r", "t_sigma_n", "R", "N"),
        "TN": ("t_sigma_t", "t_sigma_n", "T", "N"),
    }
    col_a, col_b, label_a, label_b = plane_map.get(plane, plane_map["RT"])
    
    fig = go.Figure()
    colors = px.colors.sequential.Plasma
    
    for i, row in enumerate(rows[:10]):
        sa = float(row.get(col_a, 1) or 1)
        sb = float(row.get(col_b, 1) or 1)
        tca_h = float(row.get("time_to_tca_days", 0) * 24)
        color = colors[i % len(colors)]
        
        for sigma_lv in [1, 2, 3]:
            theta = np.linspace(0, 2*np.pi, 120)
            xe = sa * sigma_lv * np.cos(theta)
            ye = sb * sigma_lv * np.sin(theta)
            
            opacity = 0.5 + 0.15 * sigma_lv   # Fixed: max = 0.95
            opacity = min(opacity, 0.95)       # Safety clamp
            
            fig.add_trace(go.Scatter(
                x=xe, y=ye,
                mode="lines",
                line=dict(color=color, width=1, dash="dot" if sigma_lv < 3 else "solid"),
                name=f"TCA-{tca_h:.0f}h / {sigma_lv}σ" if sigma_lv == 3 else None,
                showlegend=(sigma_lv == 3),
                opacity=opacity,
            ))
    
    # Miss distance marker
    if rows:
        miss_km = float(rows[0].get("miss_distance_km", 0) or 0)
        fig.add_trace(go.Scatter(
            x=[miss_km], y=[0],
            mode="markers",
            marker=dict(symbol="x", size=14, color="#FF0000"),
            name="Miss Point",
        ))
    
    fig.update_layout(
        title=f"Position Uncertainty Ellipses – {plane} Plane (Target)",
        xaxis_title=f"{label_a} (km)",
        yaxis_title=f"{label_b} (km)",
        yaxis_scaleanchor="x",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,17,27,0.8)",
        font=dict(color="#E0E0E0"),
        height=450,
    )
    return fig


# ---------------------------------------------------------------------------
# Mahalanobis Distance Trend
# ---------------------------------------------------------------------------

def plot_mahalanobis(rows: List[Dict]) -> go.Figure:
    tcas = [r["time_to_tca_days"] * 24 for r in rows]
    mah  = [r.get("mahalanobis_distance", 0) for r in rows]

    fig = go.Figure(go.Scatter(
        x=tcas, y=mah,
        mode="lines+markers",
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.15)",
        line=dict(color="#3B82F6", width=2),
        marker=dict(size=7),
        name="Mahalanobis Distance",
    ))
    fig.add_hline(y=3.0, line_dash="dash", line_color="#F97316",
                  annotation_text="3σ threshold")

    fig.update_xaxes(title_text="Time to TCA (hours)", autorange="reversed")
    fig.update_yaxes(title_text="Mahalanobis Distance (σ)")
    fig.update_layout(
        title="Mahalanobis Distance vs Time to TCA",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,17,27,0.8)",
        font=dict(color="#E0E0E0"),
        height=380,
    )
    return fig


# ---------------------------------------------------------------------------
# Pc Method Comparison
# ---------------------------------------------------------------------------

def plot_pc_comparison(pc_dict: Dict) -> go.Figure:
    """Bar chart comparing Pc estimates from different methods."""
    methods = ["Alfano", "Foster", "Chan", "Monte Carlo", "Consensus"]
    # Accept both key styles: "pc_alfano" (API) and "alfano" (dashboard)
    key_pairs = [
        ("pc_alfano",      "alfano"),
        ("pc_foster",      "foster"),
        ("pc_chan",        "chan"),
        ("pc_monte_carlo", "monte_carlo"),
        ("pc_consensus",   "consensus"),
    ]
    values = [pc_dict.get(k1, pc_dict.get(k2, 0)) or 0 for k1, k2 in key_pairs]
    log_vals = [math.log10(v) if v > 0 else -15 for v in values]

    colors = ["#3B82F6", "#8B5CF6", "#EC4899", "#F97316", "#10B981"]

    fig = go.Figure(go.Bar(
        x=methods, y=log_vals,
        marker_color=colors,
        text=[f"{v:.2e}" for v in values],
        textposition="outside",
        hovertemplate="%{x}<br>log₁₀(Pc) = %{y:.2f}<br>Pc = %{text}<extra></extra>",
    ))

    for lvl, val, color in [
        ("CRITICAL", -4, "#FF0000"),
        ("HIGH", -5, "#FF6600"),
        ("MEDIUM", -6, "#FFAA00"),
    ]:
        fig.add_hline(y=val, line_dash="dash", line_color=color, annotation_text=lvl)

    fig.update_layout(
        title="Collision Probability – Method Comparison",
        xaxis_title="Method",
        yaxis_title="log₁₀(Pc)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,17,27,0.8)",
        font=dict(color="#E0E0E0"),
        height=420,
    )
    return fig


# ---------------------------------------------------------------------------
# Space Weather Correlation
# ---------------------------------------------------------------------------

def plot_space_weather(rows: List[Dict]) -> go.Figure:
    """Scatter matrix of F10/AP vs sigma_r (drag-induced uncertainty)."""
    f10   = [r.get("F10", 0) for r in rows]
    ap    = [r.get("AP", 0) for r in rows]
    sig_r = [r.get("t_sigma_r", 0) for r in rows]
    risk  = [r.get("risk", -99) for r in rows]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("F10.7 vs Target σ_R", "Ap Index vs Target σ_R"),
    )

    scatter_kwargs = dict(
        mode="markers",
        marker=dict(
            color=risk, colorscale="RdYlGn_r",
            colorbar=dict(title="log₁₀Pc"),
            size=8, opacity=0.7,
        ),
    )

    fig.add_trace(go.Scatter(
        x=f10, y=sig_r, **scatter_kwargs, name="F10.7",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=ap, y=sig_r, **scatter_kwargs, name="Ap", showlegend=False,
    ), row=1, col=2)

    fig.update_xaxes(title_text="F10.7 Solar Flux", row=1, col=1)
    fig.update_xaxes(title_text="Ap Geomagnetic Index", row=1, col=2)
    fig.update_yaxes(title_text="Target σ_R (km)", row=1, col=1)
    fig.update_layout(
        title="Space Weather vs Uncertainty Correlation",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,17,27,0.8)",
        font=dict(color="#E0E0E0"),
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# Orbital Regime Chart
# ---------------------------------------------------------------------------

def plot_orbital_regime(rows: List[Dict]) -> go.Figure:
    """Scatter of apogee vs perigee altitude, coloured by risk."""
    apo   = [r.get("t_apogee_km", 0) for r in rows]
    per   = [r.get("t_perigee_km", 0) for r in rows]
    risk  = [r.get("risk", -99) for r in rows]
    ev_id = [r.get("event_id", 0) for r in rows]

    fig = go.Figure(go.Scatter(
        x=per, y=apo,
        mode="markers",
        marker=dict(
            color=risk, colorscale="RdYlGn_r",
            size=10, opacity=0.8,
            colorbar=dict(title="log₁₀(Pc)"),
            line=dict(width=0.5, color="white"),
        ),
        text=[f"Event {e}" for e in ev_id],
        hovertemplate="Event %{text}<br>Perigee: %{x:.0f} km<br>Apogee: %{y:.0f} km<extra></extra>",
    ))

    # Regime boundaries
    for alt, name in [(2000, "LEO/MEO"), (20200, "MEO/GEO")]:
        fig.add_hline(y=alt + 6378, line_dash="dot", line_color="#888",
                      annotation_text=name)
        fig.add_vline(x=alt + 6378, line_dash="dot", line_color="#888")

    fig.update_layout(
        title="Orbital Regime – Target Apogee vs Perigee",
        xaxis_title="Perigee Altitude (km)",
        yaxis_title="Apogee Altitude (km)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,17,27,0.8)",
        font=dict(color="#E0E0E0"),
        height=420,
    )
    return fig


# ---------------------------------------------------------------------------
# Export Utilities
# ---------------------------------------------------------------------------

class ExportUtils:
    @staticmethod
    def to_csv(rows: List[Dict], filepath: str):
        import csv
        if not rows:
            return
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def to_json(rows: List[Dict], filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, default=str)