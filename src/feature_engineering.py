"""
Phase 1.5 — Feature Engineering
===============================

Sits between the raw synthetic logs (Phase 1) and detection (Phases 2-3).
Adds six context features that make the ML detector far more discriminative:

  is_sensitive_host       — is the destination a sensitive system?
  is_dc_access            — is the destination the domain controller?
  event_type_code         — ordinal severity of the event type
  user_avg_login_hour     — this user's median login hour (baseline)
  hour_deviation          — |login_hour - user_avg_login_hour|
  time_since_last_event_min — velocity feature (attack bursts are fast)
  failed_success_ratio    — failed / (failed + total logins) in the window

Also assigns a stable, chronological `event_id` (EVT-000001, ...) which is
the join key used by every downstream phase (rules, ML, risk, alerts).

CRITICAL DESIGN NOTES — no label leakage:
  * user_avg_login_hour is computed as a MEDIAN across ALL of a user's
    events. Because ~85% of rows are normal, the median is the normal
    baseline. This is standard unsupervised baseline estimation.
  * time_since_last_event_min uses only the PRECEDING event per user.
  * No label, scenario, or future data is ever consulted.

Output:
    data/processed/events_enriched.csv

Usage:
    python -m src.feature_engineering
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import pandas as pd
import yaml

DEFAULT_CONFIG = os.path.join("config", "config.yaml")

# Ordinal severity per event type — higher = more inherently suspicious
EVENT_TYPE_CODE: Dict[str, int] = {
    "login": 1,
    "logout": 1,
    "email_access": 2,
    "file_access": 2,
    "server_access": 4,
    "mfa_event": 5,
    "privilege_change": 7,
    "process_start": 7,
    "account_creation": 8,
}


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _sensitive_host_set(cfg: dict) -> set:
    hosts = cfg["hosts"]
    return set(hosts.get("sensitive", []) + ["dc-01", "admin-jump-host"])


def enrich(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add context features and a stable event_id. Never touches label/scenario."""
    out = df.copy()
    sensitive = _sensitive_host_set(cfg)

    # --- host context ------------------------------------------------
    out["is_sensitive_host"] = (
        out["destination_host"].astype(str).isin(sensitive).astype(int)
    )
    out["is_dc_access"] = (
        (out["destination_host"].astype(str) == "dc-01").astype(int)
    )

    # --- event type severity -----------------------------------------
    out["event_type_code"] = (
        out["event_type"].astype(str).str.lower()
        .map(EVENT_TYPE_CODE).fillna(0).astype(int)
    )

    # --- per-user login-hour baseline --------------------------------
    login_rows = out["event_type"].astype(str).str.lower() == "login"
    hourly = out.loc[login_rows, ["username", "login_hour"]]
    if hourly.empty:
        out["user_avg_login_hour"] = out["login_hour"].astype(float)
    else:
        baseline = hourly.groupby("username")["login_hour"].median()
        out["user_avg_login_hour"] = (
            out["username"].map(baseline)
            .fillna(out["login_hour"])
            .astype(float)
        )
    out["hour_deviation"] = (
        out["login_hour"].astype(float) - out["user_avg_login_hour"]
    ).abs().round(2)

    # --- event velocity (backward-looking only) ----------------------
    out = out.sort_values(["username", "timestamp"]).reset_index(drop=True)
    out["time_since_last_event_min"] = (
        out.groupby("username")["timestamp"].diff()
        .dt.total_seconds().div(60).fillna(9999).round(2)
    )

    # --- failed/success ratio in the window --------------------------
    safe_login_freq = out["login_frequency"].astype(float).replace(0, np.nan)
    out["failed_success_ratio"] = (
        (out["failed_login_count"].astype(float) / safe_login_freq)
        .fillna(0.0).clip(0, 1).round(3)
    )

    # --- stable event_id, chronological ------------------------------
    # Every downstream phase (rule_engine, ml_detector, risk_engine)
    # joins on this column, so it must exist and be deterministic.
    out = out.sort_values("timestamp").reset_index(drop=True)
    if "event_id" in out.columns:
        out = out.drop(columns=["event_id"])
    out.insert(
        0, "event_id",
        [f"EVT-{i + 1:06d}" for i in range(len(out))],
    )

    return out


def _print_summary(df: pd.DataFrame, out_path: str) -> None:
    print("\n" + "=" * 66)
    print("  PHASE 1.5 — FEATURE ENGINEERING")
    print("=" * 66)
    print(f"  Output file          : {out_path}")
    print(f"  Rows                 : {len(df)}")
    print(f"  Columns              : {len(df.columns)}")
    new_cols = [
        "is_sensitive_host", "is_dc_access", "event_type_code",
        "user_avg_login_hour", "hour_deviation",
        "time_since_last_event_min", "failed_success_ratio",
    ]
    print("  New features added (normal mean vs attack mean):")
    for c in new_cols:
        n_mean = df.loc[df["label"] == 0, c].mean()
        a_mean = df.loc[df["label"] == 1, c].mean()
        print(f"    {c:<28} normal={n_mean:>8.3f}  attack={a_mean:>8.3f}")
    print("=" * 66 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1.5 — feature engineering.")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--input", default="data/raw/security_logs.csv")
    ap.add_argument("--output", default="data/processed/events_enriched.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"Input '{args.input}' not found. "
            f"Run `python -m src.data_generator` first."
        )

    print(f"[*] Loading raw events from {args.input} ...")
    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    print(f"[*] Loaded {len(df)} events.")

    print("[*] Engineering context features ...")
    enriched = enrich(df, cfg)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    enriched.to_csv(args.output, index=False)
    print(f"[+] Saved to {args.output}")

    _print_summary(enriched, args.output)


if __name__ == "__main__":
    main()