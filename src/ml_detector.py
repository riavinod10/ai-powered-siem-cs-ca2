"""
Phase 3 — Isolation Forest with Tiered Scoring (v2)
====================================================

Changes from v1:
  * Uses the 7 context features from Phase 1.5, total 16 features.
  * Scores each event on a THREE-LEVEL scale relative to the distribution
    of normal training scores:
        - "strong" anomaly : > 97th percentile of normal training scores
        - "weak"   anomaly : > 85th percentile of normal training scores
        - "none"           : otherwise
  * The risk engine maps weak -> +5 points, strong -> +15 points.
    This is far less punitive than a binary flag that added +10 to every
    borderline normal row.

Because the tier thresholds are computed from the *normal training
distribution*, they are stable and reproducible (seeded).

Usage:
    python -m src.ml_detector
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DEFAULT_CONFIG = os.path.join("config", "config.yaml")

MFA_ENCODING = {
    "none": 0,
    "push": 1,
    "push_denied": 2,
    "new_device_registered": 3,
}

# The full 16-feature vector the model is trained on.
NUMERIC_FEATURES: List[str] = [
    "failed_login_count",
    "unique_host_count",
    "unique_ip_count",
    "login_frequency",
    "login_hour",
    "hour_deviation",
    "time_since_last_event_min",
    "failed_success_ratio",
]
BINARY_FEATURES: List[str] = [
    "new_device",
    "remote_login",
    "privilege_escalation",
    "credential_access",
    "is_sensitive_host",
    "is_dc_access",
]
ORDINAL_FEATURES: List[str] = [
    "event_type_code",
    "mfa_event_encoded",
]

ALL_FEATURES: List[str] = NUMERIC_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES

# Tier thresholds (percentiles of NORMAL training scores)
WEAK_PCT = 85
STRONG_PCT = 97


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------
def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in NUMERIC_FEATURES + BINARY_FEATURES + ["event_type_code"]
               if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing feature columns: {missing}. "
            f"Run `python -m src.feature_engineering` first."
        )
    if "mfa_event" not in df.columns:
        raise ValueError("Input is missing 'mfa_event' column.")

    X = pd.DataFrame(index=df.index)
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in BINARY_FEATURES:
        X[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    X["event_type_code"] = (
        pd.to_numeric(df["event_type_code"], errors="coerce").fillna(0).astype(int)
    )
    X["mfa_event_encoded"] = (
        df["mfa_event"].astype(str).str.lower()
        .map(MFA_ENCODING).fillna(0).astype(int)
    )
    return X[ALL_FEATURES]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(
    X_normal: pd.DataFrame, cfg: dict
) -> Tuple[IsolationForest, StandardScaler, float, float]:
    if X_normal.empty:
        raise ValueError("No normal rows available to train on.")

    ml_cfg = cfg["ml"]
    scaler = StandardScaler().fit(X_normal.values)

    model = IsolationForest(
        n_estimators=int(ml_cfg["n_estimators"]),
        contamination=float(ml_cfg.get("contamination", 0.03)),
        random_state=int(cfg["project"]["random_seed"]),
        n_jobs=-1,
    )
    model.fit(scaler.transform(X_normal.values))

    # --- percentile thresholds from NORMAL training scores ---
    train_scores = -model.decision_function(scaler.transform(X_normal.values))
    weak_thr = float(np.percentile(train_scores, WEAK_PCT))
    strong_thr = float(np.percentile(train_scores, STRONG_PCT))

    return model, scaler, weak_thr, strong_thr


def _tier(score: float, weak_thr: float, strong_thr: float) -> str:
    if score >= strong_thr:
        return "strong"
    if score >= weak_thr:
        return "weak"
    return "none"


def score_events(
    df: pd.DataFrame,
    model: IsolationForest,
    scaler: StandardScaler,
    weak_thr: float,
    strong_thr: float,
) -> pd.DataFrame:
    X = build_feature_matrix(df)
    Xs = scaler.transform(X.values)

    raw = -model.decision_function(Xs)
    preds = model.predict(Xs)

    out = df.copy()
    out["anomaly_score"] = np.round(raw, 6)
    out["anomaly_prediction"] = preds
    out["is_anomaly"] = (preds == -1).astype(int)  # keep for backwards compat
    out["ml_tier"] = [_tier(s, weak_thr, strong_thr) for s in raw]
    return out


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def _print_summary(scored: pd.DataFrame, weak_thr: float, strong_thr: float) -> None:
    total = len(scored)
    tier_counts = scored["ml_tier"].value_counts().to_dict()
    weak = tier_counts.get("weak", 0)
    strong = tier_counts.get("strong", 0)

    # Treat strong OR weak as "flagged" for legacy metrics
    flagged = weak + strong
    tp = int(((scored["ml_tier"] != "none") & (scored["label"] == 1)).sum())
    fp = int(((scored["ml_tier"] != "none") & (scored["label"] == 0)).sum())
    fn = int(((scored["ml_tier"] == "none") & (scored["label"] == 1)).sum())
    tn = int(((scored["ml_tier"] == "none") & (scored["label"] == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    print("\n" + "=" * 66)
    print("  PHASE 3 — TIERED ISOLATION FOREST")
    print("=" * 66)
    print(f"  Total events              : {total}")
    print(f"  Weak threshold            : {weak_thr:.4f}  ({WEAK_PCT}th pct of normal)")
    print(f"  Strong threshold          : {strong_thr:.4f}  ({STRONG_PCT}th pct of normal)")
    print("-" * 66)
    print(f"  Tier distribution:")
    print(f"    strong anomaly          : {strong:>6}")
    print(f"    weak   anomaly          : {weak:>6}")
    print(f"    no anomaly              : {total - flagged:>6}")
    print("-" * 66)
    print("  Combined (weak + strong) vs ground truth:")
    print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"    Precision : {precision:.3f}")
    print(f"    Recall    : {recall:.3f}")
    print(f"    F1        : {f1:.3f}")
    print(f"    FPR       : {fpr:.3f}")
    print("=" * 66 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 3 — tiered Isolation Forest.")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--input", default="data/processed/events_with_rules.csv")
    ap.add_argument("--output", default="data/processed/events_scored.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"Input '{args.input}' not found. Run `python -m src.rule_engine` first."
        )

    print(f"[*] Loading events from {args.input} ...")
    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    print(f"[*] Loaded {len(df)} events.")

    X_all = build_feature_matrix(df)
    X_normal = X_all.loc[df["label"] == 0]
    print(f"[*] Training on {len(X_normal)} normal-only rows "
          f"({len(ALL_FEATURES)} features) ...")

    model, scaler, weak_thr, strong_thr = train_model(X_normal, cfg)

    print("[*] Scoring all events ...")
    scored = score_events(df, model, scaler, weak_thr, strong_thr)

    model_path = cfg["ml"]["model_path"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "features": ALL_FEATURES,
            "weak_threshold": weak_thr,
            "strong_threshold": strong_thr,
        },
        model_path,
    )
    print(f"[+] Model saved to:        {model_path}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    scored.to_csv(args.output, index=False)
    print(f"[+] Scored events saved to: {args.output}")

    _print_summary(scored, weak_thr, strong_thr)


if __name__ == "__main__":
    main()