"""
Phase 4b — Alert Generation (Incident-Level) — v3
=================================================

Consumes the correlated incidents from Phase 4a and produces one alert
per incident. Handles NaN/empty rule strings cleanly so pure-ML alerts
are honestly labelled "ML Anomaly (Unclassified)".

Usage:
    python -m src.alert_engine
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
import yaml

DEFAULT_CONFIG = os.path.join("config", "config.yaml")


THREAT_TYPE_BY_RULE = [
    ("R8", "Credential Access"),
    ("R7", "Persistence: New Privileged Account"),
    ("R6", "Lateral Movement"),
    ("R5", "Privilege Escalation"),
    ("R4", "MFA Abuse"),
    ("R1", "Credential Abuse"),
    ("R3", "Unrecognized Device"),
    ("R2", "Off-Hours Access"),
]

RULE_NAME = {
    "R1": "Multiple Failed Logins",
    "R2": "Unusual Login Time",
    "R3": "New / Unrecognised Device",
    "R4": "New MFA Device Registered",
    "R5": "Privilege Escalation",
    "R6": "Multiple Host Access (Lateral Movement)",
    "R7": "New Privileged Account Created",
    "R8": "Credential-Access Activity",
}

MITRE_BY_RULE = {
    "R1": "T1110 - Brute Force",
    "R2": "T1078 - Valid Accounts",
    "R3": "T1078 - Valid Accounts",
    "R4": "T1098.005 - Account Manipulation: Device Registration",
    "R5": "T1098 - Account Manipulation",
    "R6": "T1021 - Remote Services",
    "R7": "T1136.001 - Create Account: Local Account",
    "R8": "T1003 - OS Credential Dumping",
}

RESPONSE_BY_SEVERITY = {
    "LOW":      "Log for review. No immediate action required.",
    "MEDIUM":   "Investigate account activity and source IP. Verify with the user.",
    "HIGH":     "Reset credentials, enforce MFA re-enrollment, isolate the affected host.",
    "CRITICAL": "Immediate incident response: disable account, isolate all affected hosts, "
                "escalate to SOC lead, begin forensic capture.",
}


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _clean_rule_tokens(triggered_rules_csv) -> set:
    """Split the CSV rule list, drop empty and 'nan'/'none' tokens."""
    fired = set()
    for r in str(triggered_rules_csv or "").split(","):
        r = r.strip()
        if not r:
            continue
        if r.lower() in ("nan", "none"):
            continue
        fired.add(r)
    return fired


def _threat_type(triggered_rules_csv) -> str:
    fired = _clean_rule_tokens(triggered_rules_csv)
    if not fired:
        return "ML Anomaly (Unclassified)"
    for rule_id, label in THREAT_TYPE_BY_RULE:
        if rule_id in fired:
            return label
    return "Suspicious Activity"


def _human_indicators(triggered_rules_csv) -> str:
    fired = _clean_rule_tokens(triggered_rules_csv)
    if not fired:
        return "Machine-learning anomaly only (no rule fired)"
    ordered = [rid for rid, _ in THREAT_TYPE_BY_RULE if rid in fired]
    ordered += sorted(fired - set(ordered))
    names = [RULE_NAME.get(rid, rid) for rid in ordered]
    return " | ".join(names)


def _mitre_union(triggered_rules_csv) -> str:
    fired = _clean_rule_tokens(triggered_rules_csv)
    tags = sorted({MITRE_BY_RULE[r] for r in fired if r in MITRE_BY_RULE})
    return "; ".join(tags)


def build_alerts(incidents: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    min_score = int(cfg["alerts"]["min_risk_score"])
    flagged = incidents[incidents["risk_score"] >= min_score].copy()
    flagged = flagged.sort_values(
        ["risk_score", "first_seen"], ascending=[False, True]
    )

    rows = []
    for i, inc in flagged.reset_index(drop=True).iterrows():
        sev = inc["severity"]
        ml_tier = str(inc.get("ml_tier", "none"))
        rows.append({
            "alert_id": f"ALT-{i + 1:06d}",
            "incident_id": inc["incident_id"],
            "timestamp": inc["first_seen"],
            "last_seen": inc["last_seen"],
            "duration_minutes": inc["duration_minutes"],
            "event_count": int(inc["event_count"]),
            "username": inc["username"],
            "source_ip": inc.get("primary_source_ip", ""),
            "affected_hosts": inc.get("affected_hosts", ""),
            "threat_type": _threat_type(inc["triggered_rules"]),
            "triggered_indicators": _human_indicators(inc["triggered_rules"]),
            "triggered_rule_ids": str(inc["triggered_rules"]) if pd.notna(
                inc["triggered_rules"]) else "",
            "mitre_techniques": _mitre_union(inc["triggered_rules"]),
            "ml_anomaly_score": inc["max_anomaly_score"],
            "ml_tier": ml_tier,
            "rule_score": int(inc["rule_score"]),
            "ml_bonus": int(inc["ml_bonus"]),
            "risk_score": int(inc["risk_score"]),
            "severity": sev,
            "status": "New",
            "recommended_response": RESPONSE_BY_SEVERITY.get(sev, "Investigate."),
            "label": int(inc.get("label", 0)),
            "scenarios": str(inc.get("scenarios", "")),
        })

    return pd.DataFrame(rows)


def _print_summary(alerts: pd.DataFrame) -> None:
    print("\n" + "=" * 66)
    print("  PHASE 4b — INCIDENT-LEVEL ALERT GENERATION")
    print("=" * 66)
    print(f"  Alerts generated        : {len(alerts)}")
    if alerts.empty:
        print("=" * 66 + "\n")
        return

    print("-" * 66)
    print("  By severity:")
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        n = int((alerts["severity"] == level).sum())
        print(f"    {level:<9} : {n:>6}")

    print("-" * 66)
    print("  Top threat types:")
    for tt, n in alerts["threat_type"].value_counts().head(8).items():
        print(f"    {tt:<40} {n:>6}")

    if "label" in alerts.columns:
        tp = int((alerts["label"] == 1).sum())
        fp = int((alerts["label"] == 0).sum())
        print("-" * 66)
        print(f"  Alerts on true suspicious incidents : {tp}")
        print(f"  Alerts on true normal     incidents : {fp}")

    actionable = alerts[
        alerts["severity"].isin(["MEDIUM", "HIGH", "CRITICAL"])
    ]
    if len(actionable) > 0:
        tp = int((actionable["label"] == 1).sum())
        fp = int((actionable["label"] == 0).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        print("-" * 66)
        print(f"  Actionable alerts (MEDIUM+)         : {len(actionable)}")
        print(f"  Precision at MEDIUM+                : {precision:.3f}")

    if (alerts["severity"] == "CRITICAL").any():
        print("-" * 66)
        print("  Example CRITICAL alert:")
        row = alerts[alerts["severity"] == "CRITICAL"].iloc[0]
        print(f"    {row['alert_id']}  user={row['username']}  "
              f"risk={row['risk_score']}  events={row['event_count']}")
        print(f"    Threat   : {row['threat_type']}")
        print(f"    MITRE    : {row['mitre_techniques']}")
        print(f"    Response : {row['recommended_response']}")
    print("=" * 66 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Phase 4b — incident-level alerts."
    )
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument(
        "--incidents",
        default="data/processed/incidents_with_risk.csv",
    )
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_path = args.output or cfg["alerts"]["output_path"]

    if not os.path.exists(args.incidents):
        raise FileNotFoundError(
            f"Input '{args.incidents}' not found. "
            f"Run `python -m src.risk_engine` first."
        )

    print(f"[*] Loading incidents from {args.incidents} ...")
    incidents = pd.read_csv(
        args.incidents, parse_dates=["first_seen", "last_seen"]
    )
    print(f"[*] Loaded {len(incidents)} correlated sessions.")

    print(f"[*] Building alerts (min risk_score = "
          f"{cfg['alerts']['min_risk_score']}) ...")
    alerts = build_alerts(incidents, cfg)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    alerts.to_csv(out_path, index=False)
    print(f"[+] Saved {len(alerts)} alerts to {out_path}")

    _print_summary(alerts)


if __name__ == "__main__":
    main()