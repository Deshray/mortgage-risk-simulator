"""
core/models.py
Default probability model + rate sensitivity analysis + Monte Carlo simulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, log_loss,
)
from sklearn.calibration import CalibratedClassifierCV
import joblib
from pathlib import Path


MODELS_DIR = Path("models")
FEATURE_COLS = [
    "annual_income", "loan_amount", "ltv", "credit_score",
    "gds_ratio", "tds_ratio", "mortgage_rate", "is_variable",
    "is_insured", "amort_remaining", "orig_year",
    "province_enc", "employment_enc", "mortgage_type_enc",
    # Engineered
    "income_loan_ratio", "log_income", "log_loan",
    "credit_ltv_interact", "dti_rate_interact",
]


# ─────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Encoders (label encoding — tree models handle this fine)
    for col, enc_col in [
        ("province",      "province_enc"),
        ("employment",    "employment_enc"),
        ("mortgage_type", "mortgage_type_enc"),
    ]:
        le = LabelEncoder()
        df[enc_col] = le.fit_transform(df[col])

    # Ratio features
    df["income_loan_ratio"]  = df["annual_income"] / (df["loan_amount"] + 1)
    df["log_income"]         = np.log1p(df["annual_income"])
    df["log_loan"]           = np.log1p(df["loan_amount"])
    df["credit_ltv_interact"]= df["credit_score"] * (1 - df["ltv"])
    df["dti_rate_interact"]  = df["tds_ratio"] * df["mortgage_rate"]

    return df


# ─────────────────────────────────────────────
# Model bundle
# ─────────────────────────────────────────────
@dataclass
class ModelBundle:
    gbt: CalibratedClassifierCV
    lr:  CalibratedClassifierCV
    scaler: StandardScaler
    feature_cols: List[str]
    metrics: Dict[str, float] = field(default_factory=dict)


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────
def train(df: pd.DataFrame) -> ModelBundle:
    df = engineer_features(df)
    X = df[FEATURE_COLS].fillna(0)
    y = df["defaulted"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Gradient Boosting (primary model)
    gbt_base = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, random_state=42,
    )
    gbt = CalibratedClassifierCV(gbt_base, cv=3, method="isotonic")
    gbt.fit(X, y)

    # Logistic Regression (interpretable baseline)
    lr_base = LogisticRegression(max_iter=1000, C=0.5, random_state=42)
    lr = CalibratedClassifierCV(lr_base, cv=3, method="sigmoid")
    lr.fit(X_scaled, y)

    # OOF AUC for GBT
    oof = np.zeros(len(y))
    for tr_idx, val_idx in cv.split(X, y):
        m = GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=42,
        )
        m.fit(X.iloc[tr_idx], y[tr_idx])
        oof[val_idx] = m.predict_proba(X.iloc[val_idx])[:, 1]

    gbt_proba = gbt.predict_proba(X)[:, 1]
    lr_proba  = lr.predict_proba(X_scaled)[:, 1]

    metrics = {
        "gbt_auc":     float(roc_auc_score(y, gbt_proba)),
        "gbt_oof_auc": float(roc_auc_score(y, oof)),
        "gbt_pr_auc":  float(average_precision_score(y, gbt_proba)),
        "gbt_brier":   float(brier_score_loss(y, gbt_proba)),
        "lr_auc":      float(roc_auc_score(y, lr_proba)),
        "lr_brier":    float(brier_score_loss(y, lr_proba)),
        "default_rate":float(y.mean()),
        "n_borrowers": int(len(y)),
    }

    bundle = ModelBundle(
        gbt=gbt, lr=lr, scaler=scaler,
        feature_cols=FEATURE_COLS, metrics=metrics,
    )
    return bundle


def save_bundle(bundle: ModelBundle) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle.gbt,    MODELS_DIR / "gbt.pkl")
    joblib.dump(bundle.lr,     MODELS_DIR / "lr.pkl")
    joblib.dump(bundle.scaler, MODELS_DIR / "scaler.pkl")
    joblib.dump(bundle.feature_cols, MODELS_DIR / "features.pkl")
    joblib.dump(bundle.metrics,      MODELS_DIR / "metrics.pkl")


def load_bundle() -> ModelBundle:
    return ModelBundle(
        gbt=joblib.load(MODELS_DIR / "gbt.pkl"),
        lr=joblib.load(MODELS_DIR / "lr.pkl"),
        scaler=joblib.load(MODELS_DIR / "scaler.pkl"),
        feature_cols=joblib.load(MODELS_DIR / "features.pkl"),
        metrics=joblib.load(MODELS_DIR / "metrics.pkl"),
    )


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────
def predict_default_prob(
    bundle: ModelBundle,
    df: pd.DataFrame,
    model: str = "gbt",
) -> np.ndarray:
    df = engineer_features(df)
    X = df[bundle.feature_cols].fillna(0)
    if model == "lr":
        X_s = bundle.scaler.transform(X)
        return bundle.lr.predict_proba(X_s)[:, 1]
    return bundle.gbt.predict_proba(X)[:, 1]


# ─────────────────────────────────────────────
# Rate sensitivity sweep
# ─────────────────────────────────────────────
def rate_sensitivity_sweep(
    bundle: ModelBundle,
    df: pd.DataFrame,
    delta_rates: List[float],
    model: str = "gbt",
) -> pd.DataFrame:
    """
    For each Δrate in delta_rates, compute portfolio-level metrics.
    Returns DataFrame with columns:
        delta_rate, mean_default_prob, p95_default_prob,
        expected_loss, loss_rate, n_stressed
    """
    from core.data import apply_rate_shock

    rows = []
    loan_amounts = df["loan_amount"].values
    lgd = 0.35  # Loss Given Default — CMHC historical ~35% for uninsured
    insured = df["is_insured"].values

    for dr in delta_rates:
        stressed_prob = apply_rate_shock(df, dr).values
        # Insured mortgages: LGD effectively ~0 for lender (CMHC absorbs)
        effective_lgd = np.where(insured, 0.0, lgd)
        el = stressed_prob * loan_amounts * effective_lgd
        rows.append({
            "delta_rate":       dr,
            "mean_default_prob":float(stressed_prob.mean()),
            "p95_default_prob": float(np.percentile(stressed_prob, 95)),
            "expected_loss":    float(el.sum()),
            "loss_rate":        float(el.sum() / loan_amounts.sum()),
            "n_stressed":       int((stressed_prob > 0.05).sum()),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Monte Carlo portfolio simulation
# ─────────────────────────────────────────────
def monte_carlo_simulation(
    df: pd.DataFrame,
    rate_paths: np.ndarray,
    lgd: float = 0.35,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Simulate portfolio losses across stochastic rate paths.

    Args:
        df:          borrower portfolio DataFrame
        rate_paths:  (n_scenarios, n_quarters) from generate_rate_scenarios()
        lgd:         loss given default for uninsured mortgages

    Returns dict with:
        total_losses     — (n_scenarios,) total portfolio loss per path
        loss_rates       — (n_scenarios,) loss / total_exposure
        peak_rates       — (n_scenarios,) max rate reached per path
        default_counts   — (n_scenarios,) number of defaults per path
        quarterly_losses — (n_scenarios, n_quarters) loss by quarter
    """
    from core.data import apply_rate_shock

    rng = np.random.default_rng(random_state)
    n_scenarios, n_quarters = rate_paths.shape
    n_borrowers = len(df)
    total_exposure = df["loan_amount"].sum()
    insured = df["is_insured"].values
    effective_lgd = np.where(insured, 0.0, lgd)

    total_losses    = np.zeros(n_scenarios)
    default_counts  = np.zeros(n_scenarios, dtype=int)
    quarterly_losses = np.zeros((n_scenarios, n_quarters))
    peak_rates      = rate_paths.max(axis=1)

    # Baseline rate (first column = current rate)
    baseline_rate = rate_paths[:, 0].mean()

    for s in range(n_scenarios):
        already_defaulted = np.zeros(n_borrowers, dtype=bool)

        for q in range(n_quarters):
            r = rate_paths[s, q]
            delta_r = max(0, r - baseline_rate)

            stressed_prob = apply_rate_shock(df, delta_r).values

            # Quarterly default: annualised prob → quarterly
            q_prob = 1 - (1 - stressed_prob) ** 0.25
            q_prob[already_defaulted] = 0  # can't default twice

            new_defaults = (rng.random(n_borrowers) < q_prob) & ~already_defaulted
            already_defaulted |= new_defaults

            q_loss = (
                df["loan_amount"].values * effective_lgd * new_defaults
            ).sum()
            quarterly_losses[s, q] = q_loss
            total_losses[s] += q_loss
            default_counts[s] += new_defaults.sum()

    return {
        "total_losses":     total_losses,
        "loss_rates":       total_losses / total_exposure,
        "peak_rates":       peak_rates,
        "default_counts":   default_counts,
        "quarterly_losses": quarterly_losses,
        "total_exposure":   total_exposure,
    }


# ─────────────────────────────────────────────
# Risk metrics
# ─────────────────────────────────────────────
def compute_risk_metrics(mc_results: Dict) -> Dict[str, float]:
    losses = mc_results["total_losses"]
    rates  = mc_results["loss_rates"]
    exposure = mc_results["total_exposure"]

    return {
        "EL":           float(losses.mean()),
        "EL_rate":      float(rates.mean()),
        "VaR_95":       float(np.percentile(losses, 95)),
        "VaR_99":       float(np.percentile(losses, 99)),
        "CVaR_95":      float(losses[losses >= np.percentile(losses, 95)].mean()),
        "CVaR_99":      float(losses[losses >= np.percentile(losses, 99)].mean()),
        "VaR_95_rate":  float(np.percentile(rates, 95)),
        "VaR_99_rate":  float(np.percentile(rates, 99)),
        "max_loss":     float(losses.max()),
        "exposure":     float(exposure),
        "mean_defaults":float(mc_results["default_counts"].mean()),
        "p99_defaults": float(np.percentile(mc_results["default_counts"], 99)),
    }