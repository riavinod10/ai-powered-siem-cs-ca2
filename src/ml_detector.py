"""
Phase 3 — Machine-Learning Anomaly Detection (Isolation Forest)
===============================================================

Trains an Isolation Forest on predominantly-normal behaviour and scores
every event in the dataset.

Design decisions (documented for the case study report):

  * Only `label == 0` rows are used to fit the model. This mirrors the
    reality that a SIEM must baseline "normal" without having labels for
    tomorrow's attack. The model therefore never sees an attack during
    training.

  * Features are the 10 behavioural columns from Phase 1/2 plus an
    ordinal encoding of the `mfa_event` category. `event_id`, `scenario`
    and `label` are excluded to prevent target leakage.

  * Features are standardised (zero mean, unit variance) because
    Isolation Forest partitions on raw numeric ranges — unscaled columns
    like `login_frequency` (0-30) would dominate `login_hour` (0-23).

  * Anomaly scores are inverted so that **higher = more anomalous**,
    matching SIEM convention and the risk engine that consumes them.

IMPORTANT
---------
Isolation Forest does NOT prove an attack occurred. It flags events whose
feature vector is a statistical outlier relative to the learned normal
baseline. Every alert it contributes to must be interpreted as
"suspicious, needs investigation", never as "confirmed compromise".

Usage
-----
    python -m src.ml_detector
    python -m src.ml_detector --input data/processed/events_with_rules.csv
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

DEFAULT_CONFIG = os.path.join("config", "config.yaml")

# Ordinal mapping for MFA events — increasing "suspicion weight"
MFA_ENCODING = {
    "none": 0,
    "push": 1,
    "push_denied": 2,
    "new_device_registered": 3,
}

# Exact feature list the model expects, in order.
NUMERIC_FEATURES: List[str] = [
    "failed_login_count",
    "unique_host_count",
    "unique_ip_count",
    "login_frequency",
    "login_hour",
]
BINARY_FEATURES: List[str] = [
    "new_device",
    "remote_login",
    "privilege_escalation",
    "credential_access",
]
# MFA becomes one more numeric column after encoding
MFA_FEATURE = "mfa_event_encoded"

ALL_FEATURES: List[str] = NUMERIC_FEATURES + BINARY_FEATURES + [MFA_FEATURE]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path: str = DEFAULT_CONFIG) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the raw event table into the 10-column numeric feature matrix."""
    missing = [c for c in NUMERIC_FEATURES + BINARY_FEATURES
               if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe is missing required columns: {missing}")
    if "mfa_event" not in df.columns:
        raise ValueError("Input dataframe is missing required column: 'mfa_event'")

    X = pd.DataFrame(index=df.index)

    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    for col in BINARY_FEATURES:
        X[col] = (
            pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        )

    X[MFA_FEATURE] = (
        df["mfa_event"].astype(str).str.lower()
        .map(MFA_ENCODING).fillna(0).astype(int)
    )

    return X[ALL_FEATURES]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_isolation_forest(
    X_normal: pd.DataFrame, cfg: Dict
) -> Tuple[IsolationForest, StandardScaler]:
    """Fit a scaler and an Isolation Forest on normal-only feature rows."""
    if X_normal.empty:
        raise ValueError("No normal rows available to train on.")

    ml_cfg = cfg["ml"]
    scaler = StandardScaler().fit(X_normal.values)

    model = IsolationForest(
        n_estimators=int(ml_cfg["n_estimators"]),
        contamination=float(ml_cfg["contamination"]),
        random_state=int(cfg["project"]["random_seed"]),
        n_jobs=-1,
    )
    model.fit(scaler.transform(X_normal.values))
    return model, scaler


def score_events(
    df: pd.DataFrame, model: IsolationForest, scaler: StandardScaler
) -> pd.DataFrame:
    """Score every row and append anomaly columns to a copy of the input."""
    X = build_feature_matrix(df)
    Xs = scaler.transform(X.values)

    # decision_function: higher = more normal. Negate so higher = more anomalous.
    raw = -model.decision_function(Xs)
    predictions = model.predict(Xs)          # -1 anomaly, 1 normal

    out = df.copy()
    out["anomaly_score"] = np.round(raw, 6)
    out["anomaly_prediction"] = predictions
    out["is_anomaly"] = (predictions == -1).astype(int)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_summary(scored: pd.DataFrame) -> None:
    total = len(scored)
    anomalies = int(scored["is_anomaly"].sum())
    tp = int(((scored["is_anomaly"] == 1) & (scored["label"] == 1)).sum())
    fp = int(((scored["is_anomaly"] == 1) & (scored["label"] == 0)).sum())
    fn = int(((scored["is_anomaly"] == 0) & (scored["label"] == 1)).sum())
    tn = int(((scored["is_anomaly"] == 0) & (scored["label"] == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    print("\n" + "=" * 66)
    print("  PHASE 3 — ISOLATION FOREST ANOMALY DETECTION")
    print("=" * 66)
    print(f"  Total events scored       : {total}")
    print(f"  Flagged as anomalies      : {anomalies}  "
          f"({anomalies / total:.1%})")
    print("-" * 66)
    print("  vs. ground-truth labels:")
    print(f"    True positives   (TP)   : {tp}")
    print(f"    False positives  (FP)   : {fp}")
    print(f"    False negatives  (FN)   : {fn}")
    print(f"    True negatives   (TN)   : {tn}")
    print("-" * 66)
    print(f"    Precision               : {precision:.3f}")
    print(f"    Recall                  : {recall:.3f}")
    print(f"    F1-score                : {f1:.3f}")
    print(f"    False-positive rate     : {fpr:.3f}")
    print("=" * 66)
    print("\nNOTE: Anomalies are STATISTICAL OUTLIERS, not proven attacks.")
    print("      Labels are used only for post-hoc evaluation, never during training.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 — Isolation Forest.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--input", default="data/processed/events_with_rules.csv",
        help="Input events file (defaults to Phase 2 output)."
    )
    parser.add_argument(
        "--output", default="data/processed/events_scored.csv",
        help="Where to write the scored events."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"Input file '{args.input}' not found. "
            f"Run `python -m src.rule_engine` first."
        )
    print(f"[*] Loading events from {args.input} ...")
    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    print(f"[*] Loaded {len(df)} events.")

    # --- train on normal-only rows ---
    normal_mask = df["label"] == 0
    X_all = build_feature_matrix(df)
    X_normal = X_all.loc[normal_mask]
    print(f"[*] Training Isolation Forest on {len(X_normal)} normal-only rows "
          f"({len(ALL_FEATURES)} features) ...")

    model, scaler = train_isolation_forest(X_normal, cfg)

    # --- score every row ---
    print("[*] Scoring all events ...")
    scored = score_events(df, model, scaler)

    # --- persist model + scaler + scored data ---
    model_path = cfg["ml"]["model_path"]
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "features": ALL_FEATURES},
                model_path)
    print(f"[+] Model saved to        : {model_path}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    scored.to_csv(args.output, index=False)
    print(f"[+] Scored events saved to: {args.output}")

    _print_summary(scored)


if __name__ == "__main__":
    main()