"""
core/data.py
Synthetic Canadian mortgage portfolio generator calibrated to CMHC published statistics.

Sources used for calibration:
- CMHC Residential Mortgage Industry Data Dashboard (2024 Q3)
- CMHC Mortgage and Debt Data Tables
- Bank of Canada Financial System Review 2023-2024

The portfolio is synthetic (no real borrower PII) but the distributions match
published Canadian mortgage market characteristics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────
# Portfolio configuration
# ─────────────────────────────────────────────
@dataclass
class PortfolioConfig:
    n_borrowers: int = 5000
    random_state: int = 42

    # Calibrated to CMHC 2024 Q3 data
    # Average Canadian mortgage ~$350k, avg income ~$95k
    mean_loan_amount: float = 350_000
    std_loan_amount: float = 120_000
    mean_income: float = 95_000
    std_income: float = 35_000
    mean_ltv: float = 0.72       # CMHC reports avg LTV ~72%
    std_ltv: float = 0.12
    mean_credit_score: float = 720
    std_credit_score: float = 65
    variable_rate_fraction: float = 0.28  # ~28% variable rate (CMHC 2024)
    insured_fraction: float = 0.42        # ~42% CMHC-insured


PROVINCE_WEIGHTS = {
    "Ontario":           0.385,
    "British Columbia":  0.175,
    "Quebec":            0.155,
    "Alberta":           0.115,
    "Manitoba":          0.035,
    "Saskatchewan":      0.030,
    "Nova Scotia":       0.025,
    "New Brunswick":     0.020,
    "Other":             0.060,
}

# Provincial house price index adjustment (vs national average)
PROVINCE_HPI_FACTOR = {
    "Ontario": 1.25, "British Columbia": 1.35, "Quebec": 0.90,
    "Alberta": 0.95, "Manitoba": 0.78, "Saskatchewan": 0.72,
    "Nova Scotia": 0.88, "New Brunswick": 0.70, "Other": 0.82,
}

# Mortgage type delinquency base rates (CMHC arrears data 2024)
# Fixed rate is baseline; variable rates carry higher sensitivity
MORTGAGE_TYPE_BASE_DEFAULT = {
    "Fixed 5Y":   0.0018,
    "Fixed 3Y":   0.0021,
    "Fixed 1Y":   0.0024,
    "Variable":   0.0035,
}


# ─────────────────────────────────────────────
# Portfolio generator
# ─────────────────────────────────────────────
def generate_portfolio(cfg: PortfolioConfig = PortfolioConfig()) -> pd.DataFrame:
    """
    Generate a synthetic Canadian mortgage portfolio.
    All distributions calibrated to CMHC published statistics.
    """
    rng = np.random.default_rng(cfg.random_state)
    n = cfg.n_borrowers

    # Province
    provinces = list(PROVINCE_WEIGHTS.keys())
    weights = list(PROVINCE_WEIGHTS.values())
    province = rng.choice(provinces, size=n, p=weights)
    hpi_factor = np.array([PROVINCE_HPI_FACTOR[p] for p in province])

    # Borrower financials
    annual_income = np.clip(
        rng.normal(cfg.mean_income, cfg.std_income, n), 30_000, 400_000
    )
    loan_amount = np.clip(
        rng.normal(cfg.mean_loan_amount, cfg.std_loan_amount, n) * hpi_factor,
        80_000, 2_000_000
    )
    property_value = loan_amount / np.clip(
        rng.normal(cfg.mean_ltv, cfg.std_ltv, n), 0.35, 0.95
    )
    ltv = loan_amount / property_value

    # Credit score (capped 300–900, Canadian scale)
    credit_score = np.clip(
        rng.normal(cfg.mean_credit_score, cfg.std_credit_score, n), 300, 900
    ).astype(int)

    # Debt service ratio (GDS ~28-32% is typical; TDS up to 44%)
    other_monthly_debt = np.clip(rng.exponential(400, n), 0, 3000)
    mortgage_rate_base = rng.uniform(0.045, 0.065, n)  # origination rate
    monthly_payment = loan_amount * (mortgage_rate_base / 12) / (
        1 - (1 + mortgage_rate_base / 12) ** -300  # 25yr amortization
    )
    gds = (monthly_payment * 12) / annual_income
    tds = (monthly_payment * 12 + other_monthly_debt * 12) / annual_income

    # Mortgage type
    mortgage_types = ["Fixed 5Y", "Fixed 3Y", "Fixed 1Y", "Variable"]
    type_weights = [0.55, 0.12, 0.05, cfg.variable_rate_fraction]
    mortgage_type = rng.choice(mortgage_types, size=n, p=type_weights)
    is_variable = (mortgage_type == "Variable").astype(int)

    # CMHC insured (required for LTV > 80%)
    is_insured = ((ltv > 0.80) | (rng.random(n) < cfg.insured_fraction * 0.6)).astype(int)
    # High LTV must be insured
    is_insured = np.where(ltv > 0.80, 1, is_insured)

    # Employment type
    employment = rng.choice(
        ["Employed", "Self-Employed", "Contract", "Retired"],
        size=n, p=[0.72, 0.14, 0.09, 0.05]
    )

    # Amortization remaining (years)
    amort_remaining = np.clip(rng.normal(20, 6, n), 1, 30)

    # Origination year (affects renewal exposure)
    orig_year = rng.choice(range(2016, 2025), size=n,
                           p=[0.05, 0.06, 0.07, 0.09, 0.12, 0.15, 0.18, 0.14, 0.14])

    # ── Default label construction ────────────────────────────────────
    # Based on CMHC arrears rates and academic mortgage default literature
    # Key drivers: LTV, credit score, DTI, mortgage type, rate environment
    base_default_prob = np.array([
        MORTGAGE_TYPE_BASE_DEFAULT[mt] for mt in mortgage_type
    ])

    # Credit score effect (logistic scaling)
    credit_score_effect = np.exp(-(credit_score - 600) / 80)

    # LTV effect (nonlinear — equity cushion matters most at extremes)
    ltv_effect = np.where(ltv > 0.80, (ltv - 0.80) * 8 + 1.0,
                  np.where(ltv > 0.65, 1.0, np.maximum(0.4, ltv / 0.65)))

    # DTI effect
    tds_effect = np.where(tds > 0.44, (tds - 0.44) * 6 + 1.5,
                  np.where(tds > 0.32, 1.2, 1.0))

    # Employment effect
    emp_effect = np.where(employment == "Employed", 1.0,
                  np.where(employment == "Self-Employed", 1.4,
                  np.where(employment == "Contract", 1.6, 0.9)))

    default_prob = np.clip(
        base_default_prob * credit_score_effect * ltv_effect * tds_effect * emp_effect,
        0.0001, 0.50
    )
    default_label = rng.random(n) < default_prob

    df = pd.DataFrame({
        "borrower_id":       np.arange(n),
        "province":          province,
        "annual_income":     annual_income.round(0),
        "loan_amount":       loan_amount.round(0),
        "property_value":    property_value.round(0),
        "ltv":               ltv.round(4),
        "credit_score":      credit_score,
        "mortgage_type":     mortgage_type,
        "is_variable":       is_variable,
        "is_insured":        is_insured,
        "employment":        employment,
        "mortgage_rate":     mortgage_rate_base.round(4),
        "monthly_payment":   monthly_payment.round(2),
        "other_monthly_debt":other_monthly_debt.round(2),
        "gds_ratio":         gds.round(4),
        "tds_ratio":         tds.round(4),
        "amort_remaining":   amort_remaining.round(1),
        "orig_year":         orig_year,
        "default_prob":      default_prob.round(6),
        "defaulted":         default_label.astype(int),
    })

    return df


# ─────────────────────────────────────────────
# Bank of Canada rate history (hardcoded from
# StatsCan Table 10-10-0139-01, updated to 2025)
# ─────────────────────────────────────────────
BOC_RATE_HISTORY = pd.DataFrame({
    "date": pd.to_datetime([
        "2020-03-27","2020-03-13","2020-01-22",
        "2022-03-02","2022-04-13","2022-06-01","2022-07-13",
        "2022-09-07","2022-10-26","2022-12-07",
        "2023-01-25","2023-03-08","2023-06-07","2023-07-12",
        "2024-06-05","2024-07-24","2024-09-04",
        "2024-10-23","2024-12-11",
        "2025-01-29","2025-03-12",
    ]),
    "rate": [
        0.0025, 0.0075, 0.0175,
        0.0050, 0.0100, 0.0150, 0.0250,
        0.0325, 0.0375, 0.0425,
        0.0450, 0.0450, 0.0500, 0.0500,
        0.0475, 0.0450, 0.0425,
        0.0375, 0.0325,
        0.0300, 0.0275,
    ],
}).sort_values("date").reset_index(drop=True)


def get_current_boc_rate() -> float:
    return float(BOC_RATE_HISTORY["rate"].iloc[-1])


# ─────────────────────────────────────────────
# Rate scenario generator
# ─────────────────────────────────────────────
def generate_rate_scenarios(
    current_rate: float,
    n_scenarios: int = 1000,
    horizon_quarters: int = 8,
    random_state: int = 42,
) -> np.ndarray:
    """
    Generate stochastic interest rate paths using a mean-reverting
    Vasicek model calibrated to BoC rate history.

    Returns array of shape (n_scenarios, horizon_quarters).
    """
    rng = np.random.default_rng(random_state)

    # Vasicek parameters (calibrated to BoC 2015-2025 history)
    kappa = 0.35   # mean reversion speed
    theta = 0.035  # long-run mean (~3.5% neutral rate)
    sigma = 0.008  # volatility per quarter
    dt    = 0.25   # quarterly

    paths = np.zeros((n_scenarios, horizon_quarters))
    r = np.full(n_scenarios, current_rate)

    for t in range(horizon_quarters):
        dr = kappa * (theta - r) * dt + sigma * np.sqrt(dt) * rng.standard_normal(n_scenarios)
        r = np.clip(r + dr, 0.001, 0.15)
        paths[:, t] = r

    return paths


# ─────────────────────────────────────────────
# Rate sensitivity: how does default prob shift
# when mortgage rate increases by Δr?
# ─────────────────────────────────────────────
def apply_rate_shock(
    df: pd.DataFrame,
    delta_rate: float,
    variable_sensitivity: float = 3.5,
    fixed_renewal_fraction: float = 0.20,
) -> pd.Series:
    """
    Apply a rate shock of `delta_rate` (e.g., 0.01 = +100bps) to the portfolio.

    Variable rate borrowers feel the full shock immediately.
    Fixed rate borrowers: `fixed_renewal_fraction` are coming up for renewal
    (realistic — ~20% of fixed mortgages renew each year).

    Returns a Series of stressed default probabilities.
    """
    stress_factor = np.ones(len(df))

    # Variable: full payment shock
    variable_mask = df["is_variable"] == 1
    new_rate_var = df.loc[variable_mask, "mortgage_rate"] + delta_rate
    payment_increase_pct = (
        new_rate_var / df.loc[variable_mask, "mortgage_rate"] - 1
    ).clip(0, 2)
    stress_factor[variable_mask] = (
        1 + payment_increase_pct * variable_sensitivity
    )

    # Fixed at renewal: partial shock (fraction renewing)
    fixed_mask = df["is_variable"] == 0
    renewal_mask = fixed_mask & (np.random.default_rng(42).random(len(df)) < fixed_renewal_fraction)
    new_rate_fix = df.loc[renewal_mask, "mortgage_rate"] + delta_rate
    payment_increase_fix = (
        new_rate_fix / df.loc[renewal_mask, "mortgage_rate"] - 1
    ).clip(0, 2)
    stress_factor[renewal_mask] = (
        1 + payment_increase_fix * variable_sensitivity * 0.6
    )

    stressed_prob = np.clip(df["default_prob"] * stress_factor, 0.0001, 0.95)
    return pd.Series(stressed_prob, index=df.index)