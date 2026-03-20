"""
app.py — Canadian Mortgage Portfolio Risk Simulator
Streamlit application. Run: streamlit run app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import streamlit as st

from core.data import (
    generate_portfolio, PortfolioConfig,
    BOC_RATE_HISTORY, get_current_boc_rate,
    generate_rate_scenarios, apply_rate_shock,
)
from core.models import (
    load_bundle, predict_default_prob,
    rate_sensitivity_sweep, monte_carlo_simulation,
    compute_risk_metrics, train, save_bundle,
)
from core.viz import (
    loss_distribution_chart, rate_sensitivity_chart,
    rate_paths_fan_chart, portfolio_breakdown_chart,
    quarterly_loss_chart, risk_metrics_table,
    ltv_default_scatter,
)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Canadian Mortgage Risk Simulator",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS — same warm academic style as Model Risk Lab
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,900;1,400&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Inconsolata:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Libre Baskerville', Georgia, serif; font-size: 15px; }

[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"]        { display: none !important; }

.stApp {
    background-color: #FBF6EC;
    background-image:
        linear-gradient(rgba(180,150,100,0.07) 1px, transparent 1px),
        linear-gradient(90deg, rgba(180,150,100,0.07) 1px, transparent 1px),
        linear-gradient(rgba(180,150,100,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(180,150,100,0.03) 1px, transparent 1px);
    background-size: 80px 80px, 80px 80px, 20px 20px, 20px 20px;
}
.block-container { padding: 0 !important; max-width: 100% !important; }

/* Cover strip */
.cover { background:#2A1F14; background-image:linear-gradient(135deg,#2A1F14 0%,#4A2E1A 50%,#2A1F14 100%); padding:2.2rem 4rem 1.8rem; border-bottom:5px solid #C8860A; }
.cover-title { font-family:'Playfair Display',serif; font-size:2.2rem; font-weight:900; color:#F5E6C8; letter-spacing:-0.02em; line-height:1; margin-bottom:0.3rem; }
.cover-sub   { font-family:'Inconsolata',monospace; font-size:0.75rem; letter-spacing:0.14em; text-transform:uppercase; color:#C8860A; }
.cover-meta  { font-family:'Inconsolata',monospace; font-size:0.65rem; color:#6A4A2A; margin-top:0.5rem; letter-spacing:0.05em; }

/* Instrument panel */
.panel { background:#1E1408; border-bottom:3px solid #C8860A; padding:1rem 4rem; }
.panel-group-label { font-family:'Inconsolata',monospace; font-size:0.58rem; letter-spacing:0.2em; text-transform:uppercase; color:#C8860A; margin-bottom:0.5rem; border-bottom:1px solid #2A1A08; padding-bottom:0.25rem; }
.panel-divider { width:1px; background:#2A1A08; margin:0 0.5rem; }

/* Body */
.body { padding:2rem 4rem; max-width:1350px; margin:0 auto; }

/* Section heading */
.sec { margin:2.5rem 0 1rem; display:flex; align-items:center; gap:1rem; }
.sec-num   { font-family:'Playfair Display',serif; font-style:italic; font-size:1.4rem; color:#C8860A; flex-shrink:0; width:2rem; text-align:right; }
.sec-title { font-family:'Playfair Display',serif; font-size:1.1rem; font-weight:700; color:#2A1F14; border-bottom:2px solid #C8860A; flex:1; padding-bottom:0.25rem; }

/* Metrics */
[data-testid="stMetric"] { background:#FFF8EE !important; border:1px solid #D4B896 !important; border-top:3px solid #8B4513 !important; border-radius:1px !important; padding:0.9rem 1rem !important; box-shadow:2px 2px 0 rgba(139,69,19,0.1) !important; }
[data-testid="stMetricLabel"] p { font-family:'Inconsolata',monospace !important; font-size:0.58rem !important; letter-spacing:0.14em !important; text-transform:uppercase !important; color:#8B6A40 !important; }
[data-testid="stMetricValue"] { font-family:'Inconsolata',monospace !important; font-size:1.5rem !important; font-weight:700 !important; color:#2A1F14 !important; }
[data-testid="stMetricDelta"] { font-family:'Inconsolata',monospace !important; font-size:0.72rem !important; }

/* Tabs */
[data-testid="stTabs"] { border-bottom:2px solid #D4B896 !important; }
[data-testid="stTabs"] button { font-family:'Inconsolata',monospace !important; font-size:0.65rem !important; font-weight:700 !important; letter-spacing:0.1em !important; text-transform:uppercase !important; color:#8B6A40 !important; padding:0.6rem 1.2rem !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color:#8B4513 !important; border-bottom:3px solid #8B4513 !important; }

/* Sliders */
[data-testid="stSlider"] label p { font-family:'Inconsolata',monospace !important; font-size:0.58rem !important; letter-spacing:0.14em !important; text-transform:uppercase !important; color:#7A6040 !important; }
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p { font-family:'Inconsolata',monospace !important; font-size:0.7rem !important; color:#C8A060 !important; }

/* Selectbox */
[data-testid="stSelectbox"] label p { font-family:'Inconsolata',monospace !important; font-size:0.58rem !important; letter-spacing:0.14em !important; text-transform:uppercase !important; color:#7A6040 !important; }
[data-testid="stSelectbox"] > div > div { background:#130D06 !important; border:1px solid #2A1A08 !important; color:#C8A060 !important; font-family:'Inconsolata',monospace !important; font-size:0.82rem !important; }

/* Button */
.stButton > button { background:#C8860A !important; border:2px solid #A06A05 !important; color:#1A1008 !important; font-family:'Inconsolata',monospace !important; font-size:0.72rem !important; font-weight:700 !important; letter-spacing:0.12em !important; text-transform:uppercase !important; border-radius:1px !important; padding:0.6rem 1.8rem !important; box-shadow:3px 3px 0 #7A5005 !important; transition:all 0.15s !important; }
.stButton > button:hover { background:#A06A05 !important; box-shadow:1px 1px 0 #7A5005 !important; transform:translate(2px,2px) !important; }

/* Expander */
[data-testid="stExpander"] { background:#FFF8EE !important; border:1px solid #D4B896 !important; border-radius:1px !important; }
[data-testid="stExpander"] summary { font-family:'Inconsolata',monospace !important; font-size:0.65rem !important; letter-spacing:0.1em !important; text-transform:uppercase !important; color:#8B6A40 !important; }

/* Radio */
[data-testid="stRadio"] label p { font-family:'Inconsolata',monospace !important; font-size:0.72rem !important; color:#5A3A1A !important; }

/* Divider */
hr { border:none !important; border-bottom:1px solid #D4B896 !important; margin:1.5rem 0 !important; }

/* Dataframe */
[data-testid="stDataFrame"] * { font-family:'Inconsolata',monospace !important; font-size:0.75rem !important; }
[data-testid="stDataFrame"] { border:1px solid #D4B896 !important; }

/* Custom */
.remark { background:#FFF8EE; border:1px solid #D4B896; border-left:5px solid #8B4513; padding:1rem 1.3rem; margin:1rem 0; }
.remark-type { font-family:'Inconsolata',monospace; font-size:0.6rem; letter-spacing:0.16em; text-transform:uppercase; color:#8B4513; margin-bottom:0.4rem; }
.remark-body { font-family:'Libre Baskerville',serif; font-size:0.92rem; color:#2A1F14; line-height:1.8; margin:0; }
.result-box { background:#F0E8D8; border:2px solid #C8860A; padding:0.8rem 1.2rem; margin:1rem 0; }
.result-label { font-family:'Inconsolata',monospace; font-size:0.58rem; letter-spacing:0.18em; text-transform:uppercase; color:#C8860A; margin-bottom:0.3rem; }
.result-body { font-family:'Libre Baskerville',serif; font-style:italic; font-size:0.9rem; color:#2A1F14; line-height:1.7; }
.exp-tag { display:inline-block; background:#F0E0C0; border:1px solid #C8860A; color:#6A3A0A; font-family:'Inconsolata',monospace; font-size:0.62rem; letter-spacing:0.08em; padding:3px 9px; margin:2px; }
.field-label { font-family:'Inconsolata',monospace; font-size:0.58rem; letter-spacing:0.14em; text-transform:uppercase; color:#8B6A40; border-bottom:1px dashed #D4B896; padding-bottom:0.3rem; margin-bottom:0.5rem; }
.boc-rate-box { background:#2A1F14; border:2px solid #C8860A; padding:0.6rem 1rem; display:inline-block; }
.boc-rate-num { font-family:'Inconsolata',monospace; font-size:2rem; font-weight:700; color:#C8860A; line-height:1; }
.boc-rate-label { font-family:'Inconsolata',monospace; font-size:0.55rem; letter-spacing:0.18em; text-transform:uppercase; color:#6A4A2A; margin-top:2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Cover
# ─────────────────────────────────────────────
current_boc = get_current_boc_rate()
st.markdown(f"""
<div class="cover">
    <div class="cover-title">🏦 Canadian Mortgage Portfolio Risk Simulator</div>
    <div class="cover-sub">
        Default Probability Modelling · Rate Stress Testing · Monte Carlo Loss Simulation
    </div>
    <div class="cover-meta">
        Data: CMHC Residential Mortgage Dashboard · Bank of Canada Rate History (StatsCan 10-10-0139-01)
        &nbsp;·&nbsp; Current BoC Rate: {current_boc*100:.2f}%
        &nbsp;·&nbsp; Portfolio calibrated to CMHC 2024 Q3 statistics
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Instrument panel
# ─────────────────────────────────────────────
st.markdown('<div class="panel">', unsafe_allow_html=True)

col_port, div1, col_rate, div2, col_mc, div3, col_run = st.columns(
    [2, 0.04, 2.5, 0.04, 2, 0.04, 1.2]
)

with col_port:
    st.markdown('<div class="panel-group-label">I. Portfolio Configuration</div>', unsafe_allow_html=True)
    n_borrowers = st.slider("n  Borrowers",        1000, 20000, 5000, 1000)
    variable_pct = st.slider("ρ  Variable Rate %",    10, 50, 28, 2,
                              help="% of portfolio on variable rate mortgages")
    insured_pct  = st.slider("θ  Insured %",           20, 70, 42, 2,
                              help="% CMHC-insured mortgages")

with div1:
    st.markdown('<div style="height:80px;width:1px;background:#2A1A08;margin:1.2rem auto 0"></div>', unsafe_allow_html=True)

with col_rate:
    st.markdown('<div class="panel-group-label">II. Rate Stress Scenarios</div>', unsafe_allow_html=True)
    shock_basis_pts = st.slider("Δr  Instantaneous Shock (bps)",  0, 400, 100, 25,
                                 help="Immediate rate shock applied to full portfolio")
    n_rate_scenarios = st.slider("N  Monte Carlo Paths",           200, 2000, 1000, 100)
    horizon_quarters = st.slider("T  Forecast Horizon (quarters)", 4, 16, 8, 2)

with div2:
    st.markdown('<div style="height:80px;width:1px;background:#2A1A08;margin:1.2rem auto 0"></div>', unsafe_allow_html=True)

with col_mc:
    st.markdown('<div class="panel-group-label">III. Model Settings</div>', unsafe_allow_html=True)
    model_choice = st.selectbox("Model", ["Gradient Boosting", "Logistic Regression"],
                                 label_visibility="collapsed")
    lgd_pct = st.slider("λ  Loss Given Default (%)", 10, 60, 35, 5,
                         help="% of loan lost on default (CMHC historical ~35%)")
    st.markdown(
        '<div style="font-family:\'Inconsolata\',monospace;font-size:0.6rem;'
        'color:#5A4020;line-height:1.7;margin-top:0.3rem">'
        'Insured mortgages: LGD=0<br>for lender (CMHC absorbs loss)</div>',
        unsafe_allow_html=True
    )

with div3:
    st.markdown('<div style="height:80px;width:1px;background:#2A1A08;margin:1.2rem auto 0"></div>', unsafe_allow_html=True)

with col_run:
    st.markdown('<div class="panel-group-label">IV. Execute</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("▶  Simulate", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Body
# ─────────────────────────────────────────────
st.markdown('<div class="body">', unsafe_allow_html=True)

if not run_btn and "sim_done" not in st.session_state:
    # Landing
    st.markdown("""
        <div style="background:#FFF8EE;border:1px solid #D4B896;border-top:5px solid #2A1F14;
                    padding:2rem 2.5rem;margin:1.5rem 0;max-width:820px">
            <div style="font-family:'Inconsolata',monospace;font-size:0.6rem;letter-spacing:0.2em;
                        text-transform:uppercase;color:#8B6A40;margin-bottom:0.8rem">Abstract</div>
            <div style="font-family:'Libre Baskerville',serif;font-size:1rem;line-height:1.9;color:#2A1F14">
                This simulator models a synthetic Canadian residential mortgage portfolio 
                calibrated to CMHC published statistics (2024 Q3). It predicts individual 
                borrower default probabilities using gradient-boosted trees trained on 
                LTV, credit score, debt service ratios, mortgage type, and macroeconomic 
                exposure. A Vasicek mean-reverting model generates stochastic Bank of Canada 
                rate paths, which are used to stress borrower default probabilities and 
                simulate portfolio-level losses. Output metrics include Expected Loss, 
                Value-at-Risk (95%/99%), and Conditional VaR — the same framework used 
                by Canadian financial institutions for OSFI stress testing compliance.
            </div>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    for col, (title, body) in zip([c1,c2,c3], [
        ("Default Probability Model",
         "Gradient-boosted classifier trained on borrower LTV, credit score, GDS/TDS ratios, "
         "mortgage type, province, and employment. Calibrated to CMHC mortgage arrears data."),
        ("Rate Stress Testing",
         "Vasicek stochastic model generates 1,000 Bank of Canada rate paths. "
         "Variable-rate borrowers receive immediate payment shocks; "
         "fixed-rate borrowers face renewal exposure."),
        ("Monte Carlo Loss Simulation",
         "Quarterly default draws across all rate paths. Outputs VaR 95%/99%, "
         "CVaR (Expected Shortfall), and quarterly loss waterfall — "
         "the standard OSFI stress testing framework."),
    ]):
        col.markdown(f"""
            <div style="background:#FFF8EE;border:1px solid #D4B896;border-top:3px solid #C8860A;
                        padding:1.2rem;height:100%">
                <div style="font-family:'Playfair Display',serif;font-size:0.95rem;
                            font-weight:700;color:#2A1F14;margin-bottom:0.5rem">{title}</div>
                <div style="font-family:'Libre Baskerville',serif;font-size:0.85rem;
                            font-weight:300;color:#5A4A30;line-height:1.7">{body}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# Run simulation
# ─────────────────────────────────────────────
with st.spinner("Generating portfolio and training models…"):
    cfg = PortfolioConfig(
        n_borrowers=n_borrowers,
        variable_rate_fraction=variable_pct / 100,
        insured_fraction=insured_pct / 100,
    )
    df = generate_portfolio(cfg)

    # Train or retrain
    bundle = train(df)
    save_bundle(bundle)
    model_key = "gbt" if model_choice == "Gradient Boosting" else "lr"
    df["predicted_prob"] = predict_default_prob(bundle, df, model=model_key)

with st.spinner("Running Monte Carlo simulation…"):
    rate_paths = generate_rate_scenarios(
        current_rate=current_boc,
        n_scenarios=n_rate_scenarios,
        horizon_quarters=horizon_quarters,
    )
    mc_results = monte_carlo_simulation(
        df, rate_paths, lgd=lgd_pct / 100
    )
    risk_metrics = compute_risk_metrics(mc_results)

# Rate sensitivity sweep
delta_rates = np.linspace(0, 0.04, 20)
df_sweep = rate_sensitivity_sweep(bundle, df, delta_rates.tolist(), model=model_key)
shock_dr = shock_basis_pts / 10000
stressed_probs = apply_rate_shock(df, shock_dr)

st.session_state.sim_done = True

# ─────────────────────────────────────────────
# §1 — Portfolio summary
# ─────────────────────────────────────────────
st.markdown("""
    <div class="sec"><span class="sec-num">1</span>
    <span class="sec-title">Portfolio Summary</span></div>
""", unsafe_allow_html=True)

tags = "".join([
    f'<span class="exp-tag">n = {n_borrowers:,} borrowers</span>',
    f'<span class="exp-tag">Variable = {variable_pct}%</span>',
    f'<span class="exp-tag">Insured = {insured_pct}%</span>',
    f'<span class="exp-tag">BoC Rate = {current_boc*100:.2f}%</span>',
    f'<span class="exp-tag">Shock = +{shock_basis_pts}bps</span>',
    f'<span class="exp-tag">Model = {model_choice}</span>',
])
st.markdown(f'<div style="margin-bottom:1.2rem">{tags}</div>', unsafe_allow_html=True)

c1,c2,c3,c4,c5,c6 = st.columns(6)
total_exp = df["loan_amount"].sum()
baseline_dr = df["defaulted"].mean()
stressed_dr = (stressed_probs > 0.05).mean()
el = risk_metrics["EL"]
var95 = risk_metrics["VaR_95"]

c1.metric("Total Exposure",      f"${total_exp/1e9:.2f}B")
c2.metric("Baseline Default Rate",f"{baseline_dr:.2%}")
c3.metric(f"Stressed Rate (+{shock_basis_pts}bps)",
          f"{stressed_dr:.2%}",
          f"{stressed_dr-baseline_dr:+.2%}")
c4.metric("Expected Loss (EL)",  f"${el/1e6:.1f}M")
c5.metric("VaR 95%",             f"${var95/1e6:.1f}M")
c6.metric("GBT AUC",             f"{bundle.metrics['gbt_auc']:.4f}")

# Summary table
st.markdown("<br>", unsafe_allow_html=True)
cl, _, cr = st.columns([2.5, 0.2, 2.5])
with cl:
    st.markdown('<div class="field-label">Portfolio Characteristics</div>', unsafe_allow_html=True)
    port_stats = pd.DataFrame({
        "Metric": ["Avg Loan Amount","Avg Annual Income","Avg LTV","Avg Credit Score",
                   "Avg GDS Ratio","Avg TDS Ratio","Avg Mortgage Rate"],
        "Value":  [
            f"${df['loan_amount'].mean():,.0f}",
            f"${df['annual_income'].mean():,.0f}",
            f"{df['ltv'].mean():.1%}",
            f"{df['credit_score'].mean():.0f}",
            f"{df['gds_ratio'].mean():.1%}",
            f"{df['tds_ratio'].mean():.1%}",
            f"{df['mortgage_rate'].mean():.2%}",
        ],
    })
    st.dataframe(port_stats, hide_index=True, use_container_width=True)

with cr:
    st.markdown('<div class="field-label">Model Performance</div>', unsafe_allow_html=True)
    model_stats = pd.DataFrame({
        "Metric": ["GBT AUC (in-sample)","GBT OOF AUC (5-fold CV)",
                   "GBT PR-AUC","GBT Brier Score",
                   "LR AUC","Portfolio Default Rate"],
        "Value":  [
            f"{bundle.metrics['gbt_auc']:.4f}",
            f"{bundle.metrics['gbt_oof_auc']:.4f}",
            f"{bundle.metrics['gbt_pr_auc']:.4f}",
            f"{bundle.metrics['gbt_brier']:.4f}",
            f"{bundle.metrics['lr_auc']:.4f}",
            f"{bundle.metrics['default_rate']:.3%}",
        ],
    })
    st.dataframe(model_stats, hide_index=True, use_container_width=True)

st.divider()

# ════════════════════════════════════
# 2. Monte Carlo
# ════════════════════════════════════
st.markdown("""
    <div class="sec"><span class="sec-num">2</span>
    <span class="sec-title">Monte Carlo Loss Distribution</span></div>
""", unsafe_allow_html=True)

col_fan, col_hist = st.columns(2)
with col_fan:
    st.markdown('<div class="field-label">2.1 — Stochastic Rate Paths (Vasicek)</div>', unsafe_allow_html=True)
    st.plotly_chart(rate_paths_fan_chart(rate_paths, current_boc), use_container_width=True)
with col_hist:
    st.markdown('<div class="field-label">2.2 — Portfolio Loss Distribution</div>', unsafe_allow_html=True)
    st.plotly_chart(loss_distribution_chart(
        mc_results["total_losses"], risk_metrics["VaR_95"],
        risk_metrics["VaR_99"], risk_metrics["CVaR_95"],
    ), use_container_width=True)

st.markdown('<div class="field-label" style="margin-top:0.5rem">2.3 — Loss by Quarter</div>', unsafe_allow_html=True)
st.plotly_chart(quarterly_loss_chart(mc_results["quarterly_losses"]), use_container_width=True)

st.markdown(f"""
    <div class="remark">
        <div class="remark-type">Remark 2.1 — Loss Simulation</div>
        <div class="remark-body">
            Across {n_rate_scenarios:,} simulated Bank of Canada rate paths over {horizon_quarters} quarters,
            the portfolio's expected loss is <strong>${risk_metrics['EL']/1e6:.1f}M</strong>
            ({risk_metrics['EL_rate']*100:.3f}% of total exposure).
            The 95th percentile loss (VaR 95%) reaches <strong>${risk_metrics['VaR_95']/1e6:.1f}M</strong>,
            while CVaR 95% — the expected loss <em>given that we are in the worst 5% of scenarios</em> — is
            <strong>${risk_metrics['CVaR_95']/1e6:.1f}M</strong>.
            Canadian banks are required under OSFI Guideline B-20 to demonstrate portfolio resilience
            under scenarios of this severity.
        </div>
    </div>
""", unsafe_allow_html=True)

st.divider()

# ════════════════════════════════════
# 3. Rate stress testing
# ════════════════════════════════════
st.markdown("""
    <div class="sec"><span class="sec-num">3</span>
    <span class="sec-title">Rate Sensitivity &amp; Stress Testing</span></div>
""", unsafe_allow_html=True)

metric_c = st.radio(
    "Metric", ["loss_rate", "mean_default_prob", "expected_loss"],
    format_func=lambda x: {"loss_rate": "Portfolio Loss Rate",
                            "mean_default_prob": "Mean Default Probability",
                            "expected_loss": "Expected Loss ($)"}[x],
    horizontal=True, label_visibility="visible"
)
baseline_val = df_sweep[metric_c].iloc[0]

col_curve, col_table = st.columns([2, 1])
with col_curve:
    st.markdown('<div class="field-label">3.1 — Sensitivity Curve</div>', unsafe_allow_html=True)
    st.plotly_chart(rate_sensitivity_chart(df_sweep, metric_c, baseline_val), use_container_width=True)
with col_table:
    st.markdown('<div class="field-label">3.2 — Scenario Table</div>', unsafe_allow_html=True)
    scenario_table = pd.DataFrame({"Shock (bps)": [0, 25, 50, 100, 150, 200, 300, 400]})
    for col_name, label in [("loss_rate","Loss Rate (%)"),("mean_default_prob","Default Prob (%)"),("expected_loss","Exp. Loss ($M)")]:
        vals = []
        for bps in scenario_table["Shock (bps)"]:
            row = df_sweep[df_sweep["delta_rate"].between(bps/10000-0.002, bps/10000+0.002)]
            if len(row):
                v = row[col_name].iloc[0]
                vals.append(f"${v/1e6:.1f}M" if col_name == "expected_loss" else f"{v*100:.3f}%")
            else:
                vals.append("—")
        scenario_table[label] = vals
    st.dataframe(scenario_table, hide_index=True, use_container_width=True)

st.divider()
st.markdown(f'<div class="field-label">3.3 — Instantaneous Shock Analysis (+{shock_basis_pts}bps)</div>', unsafe_allow_html=True)

col_a, col_b, col_c, col_d = st.columns(4)
stressed_el = (stressed_probs * df["loan_amount"] * np.where(df["is_insured"]==1, 0, lgd_pct/100)).sum()
baseline_el = (df["default_prob"] * df["loan_amount"] * np.where(df["is_insured"]==1, 0, lgd_pct/100)).sum()
col_a.metric("Stressed Mean P(Default)", f"{stressed_probs.mean():.4%}", f"{(stressed_probs.mean()-df['default_prob'].mean()):+.4%}")
col_b.metric("Stressed Expected Loss", f"${stressed_el/1e6:.1f}M", f"${(stressed_el-baseline_el)/1e6:+.1f}M")
col_c.metric("Variable Rate Borrowers at Risk", f"{(stressed_probs[df['is_variable']==1] > 0.05).sum():,}")
col_d.metric("Loss Increase vs Baseline", f"{stressed_el/baseline_el - 1:.1%}")

st.markdown(f"""
    <div class="result-box">
        <div class="result-label">Result 3.1 — Rate Sensitivity</div>
        <div class="result-body">
            A +{shock_basis_pts}bps instantaneous shock increases the portfolio expected loss
            from ${baseline_el/1e6:.1f}M to ${stressed_el/1e6:.1f}M (+{stressed_el/baseline_el-1:.1%}).
            Variable-rate borrowers — {variable_pct}% of this portfolio — absorb the full shock immediately
            through higher monthly payments, while fixed-rate borrowers face exposure only at renewal.
            This asymmetry is a key structural vulnerability in Canadian mortgage portfolios,
            particularly given the proportion of 5-year fixed mortgages originated at 2020–2021 rates
            now entering renewal.
        </div>
    </div>
""", unsafe_allow_html=True)

st.divider()

# ════════════════════════════════════
# 4. Portfolio analysis
# ════════════════════════════════════
st.markdown("""
    <div class="sec"><span class="sec-num">4</span>
    <span class="sec-title">Portfolio Composition &amp; Risk Drivers</span></div>
""", unsafe_allow_html=True)

col_l, col_r = st.columns(2)
with col_l:
    st.markdown('<div class="field-label">4.1 — Default Risk by Province</div>', unsafe_allow_html=True)
    st.plotly_chart(portfolio_breakdown_chart(df, "province", "default_prob", "Avg Default Probability by Province"), use_container_width=True)
with col_r:
    st.markdown('<div class="field-label">4.2 — Default Risk by Mortgage Type</div>', unsafe_allow_html=True)
    st.plotly_chart(portfolio_breakdown_chart(df, "mortgage_type", "default_prob", "Avg Default Probability by Mortgage Type"), use_container_width=True)

col_l2, col_r2 = st.columns(2)
with col_l2:
    st.markdown('<div class="field-label">4.3 — LTV vs Default Probability</div>', unsafe_allow_html=True)
    st.plotly_chart(ltv_default_scatter(df), use_container_width=True)
with col_r2:
    st.markdown('<div class="field-label">4.4 — Default Risk by Employment Type</div>', unsafe_allow_html=True)
    st.plotly_chart(portfolio_breakdown_chart(df, "employment", "default_prob", "Avg Default Probability by Employment"), use_container_width=True)

st.markdown("""
    <div class="remark">
        <div class="remark-type">Remark 4.1 — Risk Drivers</div>
        <div class="remark-body">
            LTV is the dominant predictor of default probability — a well-established finding in
            mortgage default literature (Deng, Quigley &amp; Van Order, 1996). Borrowers with LTV
            above 80% face nonlinearly higher default risk, as negative equity removes the financial
            incentive to continue servicing the mortgage. Employment type shows a clear gradient:
            contract and self-employed borrowers carry materially higher risk due to income volatility.
            Provincial variation reflects both house price levels and local economic conditions.
        </div>
    </div>
""", unsafe_allow_html=True)

st.divider()

# ════════════════════════════════════
# 5. Risk metrics
# ════════════════════════════════════
st.markdown("""
    <div class="sec"><span class="sec-num">5</span>
    <span class="sec-title">Risk Metrics — OSFI Stress Testing Framework</span></div>
""", unsafe_allow_html=True)

col_tbl, col_notes = st.columns([1.6, 1])
with col_tbl:
    st.markdown('<div class="field-label">5.1 — Summary Risk Table</div>', unsafe_allow_html=True)
    st.plotly_chart(risk_metrics_table(risk_metrics), use_container_width=True)
with col_notes:
    st.markdown("""
        <div style="margin-top:1rem">
        <div class="field-label">Metric Definitions</div>
        <div style="font-family:'Libre Baskerville',serif;font-size:0.85rem;color:#4A3010;line-height:1.85">
            <strong>EL (Expected Loss)</strong> — probability-weighted average loss across all simulated
            scenarios. The cost the portfolio <em>expects</em> to bear.<br><br>
            <strong>VaR 95%/99%</strong> — the loss not exceeded in 95%/99% of simulated scenarios.<br><br>
            <strong>CVaR 95% (Expected Shortfall)</strong> — the average loss <em>conditional on being in
            the worst 5%</em> of scenarios. More informative than VaR for tail risk.<br><br>
            <strong>OSFI Guideline B-20</strong> requires Canadian lenders to demonstrate resilience under
            severe but plausible scenarios. CVaR 99% approximates this threshold.
        </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
    <div class="result-box" style="margin-top:1rem">
        <div class="result-label">Result 5.1 — Capital Adequacy Assessment</div>
        <div class="result-body">
            Under {n_rate_scenarios:,} stochastic rate scenarios, the portfolio's CVaR at 99% confidence
            is ${risk_metrics['CVaR_99']/1e6:.1f}M — the expected loss in the worst 1% of macroeconomic
            outcomes. Expressed as {risk_metrics['CVaR_99']/risk_metrics['exposure']*100:.2f}% of total
            portfolio exposure, this figure represents the minimum capital buffer a lender would need
            to hold under an OSFI-equivalent stress test framework.
        </div>
    </div>
""", unsafe_allow_html=True)

with st.expander("Methodology — Data sources and model assumptions"):
    st.markdown("""
    <div style="font-family:'Libre Baskerville',serif;font-size:0.88rem;color:#4A3010;line-height:1.85">
    <strong>Portfolio generation:</strong> Synthetic borrower records calibrated to CMHC Residential
    Mortgage Industry Data Dashboard (2024 Q3). Key statistics matched: average LTV ~72%,
    variable rate share ~28%, insured fraction ~42%, average loan ~$350k, average income ~$95k.<br><br>
    <strong>Default model:</strong> Gradient-boosted classifier (scikit-learn GBT, 200 estimators,
    isotonic calibration via 3-fold CV). Features: LTV, credit score, GDS/TDS ratios, mortgage rate,
    employment type, province, mortgage type, amortization remaining, and seven engineered interaction terms.<br><br>
    <strong>Rate model:</strong> Vasicek mean-reverting model: kappa=0.35, theta=3.5% long-run mean,
    sigma=0.8%/quarter, calibrated to 2015-2025 BoC rate history.<br><br>
    <strong>Loss Given Default:</strong> 35% for uninsured mortgages (CMHC historical). 0% for insured.<br><br>
    <strong>Limitations:</strong> Portfolio is synthetic. Default correlations (contagion) not modelled.
    House price dynamics not incorporated into LGD. Government intervention programs not modelled.
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)