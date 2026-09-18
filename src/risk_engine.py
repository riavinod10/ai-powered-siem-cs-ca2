"""
Phase 4a — Hybrid Risk Scoring + Incident Correlation (v2)
==========================================================

Two levels of risk:

  1. PER-EVENT risk (data/processed/events_with_risk.csv)
     Combines rule_score + tiered ML bonus (weak=+5, strong=+15).
     Used by: incident timeline (dashboard page 5).

  2. PER-INCIDENT risk (data/processed/incidents_with_risk.csv)
     Groups each user's events into sessions whenever consecutive events
     are within `alerts.session_gap_minutes`. Rules fired across the whole
     session are UNIONED (not summed) to avoid double-counting. Session ML
     tier = highest tier seen in the session.
     Used by: alert engine (Phase 4b) and SOC dashboard.

Usage:
    python -m src.risk_engine
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import pandas as pd
import yaml

DEFAULT_CONFIG = os.path.join("config", "config.yaml")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path: str = DEFAULT_CONFIG) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _rule_id_to_points(cfg: dict) -> Dict[str, int]:
    w = cfg["risk_weights"]
    return {
        "R1": int(w["multiple_failed_logins"]),
        "R2": int(w["unusual_login_time"]),
        "R3": int(w["new_device"]),
        "R4": int(w["new_mfa_device"]),
        "R5": int(w["privilege_escalation"]),
        "R6": int(w["multiple_host_access"]),
        "R7": int(w["new_privileged_account"]),
        "R8": int(w["credential_access"]),
    }


def severity_from_score(score: int, bands: dict) -> str:
    for level, (lo, hi) in bands.items():
        if lo <= score <= hi:
            return level.upper()
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Per-event scoring — tiered ML bonus
# ---------------------------------------------------------------------------
def compute_event_risk(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    required = {"rule_score", "ml_tier"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input is missing required columns: {sorted(missing)}. "
            f"Run Phase 2 and Phase 3 first."
        )

    out = df.copy()
    weights = cfg["risk_weights"]

    def _ml_bonus(tier: str) -> int:
        if tier == "strong":
            return int(weights["ml_anomaly_strong"])
        if tier == "weak":
            return int(weights["ml_anomaly_weak"])
        return 0

    out["ml_bonus"] = out["ml_tier"].astype(str).apply(_ml_bonus).astype(int)
    out["rule_score"] = pd.to_numeric(
        out["rule_score"], errors="coerce"
    ).fillna(0).astype(int)
    out["risk_score"] = (
        out["rule_score"] + out["ml_bonus"]
    ).clip(upper=100).astype(int)
    out["severity"] = out["risk_score"].apply(
        lambda s: severity_from_score(s, cfg["severity_bands"])
    )
    return out


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------
def _split_sessions(ts_series: pd.Series, gap_minutes: int) -> List[tuple]:
    if len(ts_series) == 0:
        return []
    gap = pd.Timedelta(minutes=gap_minutes)
    sessions = []
    start = 0
    ts_values = ts_series.reset_index(drop=True)
    for i in range(1, len(ts_values)):
        if (ts_values.iloc[i] - ts_values.iloc[i - 1]) > gap:
            sessions.append((start, i))
            start = i
    sessions.append((start, len(ts_values)))
    return sessions


def _union_rules(cells) -> List[str]:
    fired = set()
    for cell in cells:
        for r in str(cell).split(","):
            r = r.strip()
            if r:
                fired.add(r)
    return sorted(fired)


def correlate_incidents(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    gap = int(cfg["alerts"]["session_gap_minutes"])
    rule_points = _rule_id_to_points(cfg)
    weights = cfg["risk_weights"]

    df = df.sort_values(["username", "timestamp"]).reset_index(drop=True)
    incidents = []

    for username, grp in df.groupby("username", sort=False):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        for s_start, s_end in _split_sessions(grp["timestamp"], gap):
            session = grp.iloc[s_start:s_end]

            fired = _union_rules(session["triggered_rules"])
            rule_score = sum(rule_points.get(r, 0) for r in fired)

            # Session ML tier = highest tier seen anywhere in the session
            tiers_present = set(session["ml_tier"].astype(str))
            if "strong" in tiers_present:
                ml_tier = "strong"
                ml_bonus = int(weights["ml_anomaly_strong"])
            elif "weak" in tiers_present:
                ml_tier = "weak"
                ml_bonus = int(weights["ml_anomaly_weak"])
            else:
                ml_tier = "none"
                ml_bonus = 0

            risk_score = min(100, rule_score + ml_bonus)

            try:
                primary_ip = session["source_ip"].mode().iloc[0]
            except Exception:
                primary_ip = session["source_ip"].iloc[0] if len(session) else ""

            incidents.append({
                "username": username,
                "first_seen": session["timestamp"].min(),
                "last_seen": session["timestamp"].max(),
                "duration_minutes": round(
                    (session["timestamp"].max() - session["timestamp"].min())
                    .total_seconds() / 60.0, 2
                ),
                "event_count": len(session),
                "event_ids": ",".join(session["event_id"].astype(str)),
                "primary_source_ip": primary_ip,
                "affected_hosts": ",".join(
                    sorted(set(session["destination_host"].astype(str)))
                ),
                "triggered_rules": ",".join(fired),
                "rule_score": rule_score,
                "ml_tier": ml_tier,
                "ml_bonus": ml_bonus,
                "max_anomaly_score": round(
                    float(session["anomaly_score"].max()), 6
                ),
                "risk_score": risk_score,
                "severity": severity_from_score(
                    risk_score, cfg["severity_bands"]
                ),
                "label": int(session["label"].max()),
                "scenarios": ",".join(
                    sorted(set(session["scenario"].astype(str)))
                ),
            })

    incidents_df = pd.DataFrame(incidents).sort_values(
        ["risk_score", "first_seen"], ascending=[False, True]
    ).reset_index(drop=True)
    incidents_df.insert(
        0, "incident_id",
        [f"INC-{i + 1:06d}" for i in range(len(incidents_df))],
    )
    return incidents_df


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def _print_event_summary(df: pd.DataFrame) -> None:
    total = len(df)
    flagged = int((df["risk_score"] > 0).sum())
    print("\n" + "=" * 66)
    print("  PHASE 4a — PER-EVENT RISK SCORING")
    print("=" * 66)
    print(f"  Events                    : {total}")
    print(f"  Events with risk > 0      : {flagged}  ({flagged / total:.1%})")
    print("  Severity distribution:")
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        n = int((df["severity"] == level).sum())
        print(f"    {level:<9} : {n:>6}")
    print("=" * 66)


def _print_incident_summary(df: pd.DataFrame) -> None:
    total = len(df)
    flagged = int((df["risk_score"] > 0).sum())
    actionable = int(
        df["severity"].isin(["MEDIUM", "HIGH", "CRITICAL"]).sum()
    )

    print("\n" + "=" * 66)
    print("  PHASE 4a — CORRELATED INCIDENTS")
    print("=" * 66)
    print(f"  Total user sessions       : {total}")
    print(f"  Sessions with risk > 0    : {flagged}  ({flagged / total:.1%})")
    print(f"  Actionable (MEDIUM+)      : {actionable}")
    print("  Severity distribution:")
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        n = int((df["severity"] == level).sum())
        print(f"    {level:<9} : {n:>6}")

    if actionable > 0:
        med_plus = df[df["severity"].isin(["MEDIUM", "HIGH", "CRITICAL"])]
        tp = int((med_plus["label"] == 1).sum())
        fp = int((med_plus["label"] == 0).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        print("-" * 66)
        print("  Precision at MEDIUM+ severity (the number that matters):")
        print(f"    True positives          : {tp}")
        print(f"    False positives         : {fp}")
        print(f"    Precision               : {precision:.3f}")
    print("=" * 66 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 4a — risk scoring + correlation."
    )
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--input", default="data/processed/events_scored.csv")
    ap.add_argument("--events-out", default="data/processed/events_with_risk.csv")
    ap.add_argument("--incidents-out", default="data/processed/incidents_with_risk.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"Input '{args.input}' not found. Run `python -m src.ml_detector` first."
        )

    print(f"[*] Loading scored events from {args.input} ...")
    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    print(f"[*] Loaded {len(df)} events.")

    print("[*] Computing per-event risk scores ...")
    events = compute_event_risk(df, cfg)
    os.makedirs(os.path.dirname(args.events_out), exist_ok=True)
    events.to_csv(args.events_out, index=False)
    print(f"[+] Events saved to:   {args.events_out}")

    print(f"[*] Correlating per-user sessions "
          f"(gap = {cfg['alerts']['session_gap_minutes']} min) ...")
    incidents = correlate_incidents(events, cfg)
    incidents.to_csv(args.incidents_out, index=False)
    print(f"[+] Incidents saved to: {args.incidents_out}")

    _print_event_summary(events)
    _print_incident_summary(incidents)


if __name__ == "__main__":
    main()