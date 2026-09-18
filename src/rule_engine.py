"""
Phase 2 — Rule-Based Detection Engine
=====================================

Applies 8 configurable behavioural rules to a synthetic security log
dataset and emits:

    data/processed/rule_hits.csv         # one row per (event x rule hit)
    data/processed/events_with_rules.csv # events enriched with rule summary

Every rule is a small, named function that returns an evidence string when
it fires and None otherwise. Thresholds come from config/config.yaml — no
magic numbers are hard-coded in the rule bodies.

Rules 1-8 correspond to behaviours documented in the Cisco 2022 incident,
mapped to observable log signals (see README for the full mapping table).

Usage
-----
    python -m src.rule_engine
    python -m src.rule_engine --input data/raw/security_logs.csv
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import pandas as pd
import yaml

from src.preprocessing import load_logs

DEFAULT_CONFIG = os.path.join("config", "config.yaml")


# ---------------------------------------------------------------------------
# Rule model
# ---------------------------------------------------------------------------
@dataclass
class Rule:
    """A single, testable detection rule."""
    rule_id: str
    name: str
    description: str
    risk_points: int
    mitre: List[str]
    check: Callable[[pd.Series, Dict], Optional[str]]

    def evaluate(self, row: pd.Series, cfg: Dict) -> Optional[str]:
        """Return an evidence string if the rule fires for this row, else None."""
        try:
            return self.check(row, cfg)
        except Exception:
            # A malformed row must never crash the whole pipeline.
            return None


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def load_config(path: str = DEFAULT_CONFIG) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Rule checks — each returns evidence string or None
# ---------------------------------------------------------------------------
def _rule_multiple_failed_logins(row: pd.Series, cfg: Dict) -> Optional[str]:
    threshold = int(cfg["rules"]["failed_login_threshold"])
    window = int(cfg["data_generation"]["feature_window_minutes"])
    count = int(row.get("failed_login_count", 0))
    if count >= threshold and row.get("event_type") == "login":
        return f"{count} failed logins for '{row['username']}' in the last {window} min"
    return None


def _rule_unusual_login_time(row: pd.Series, cfg: Dict) -> Optional[str]:
    start = int(cfg["rules"]["unusual_hour_start"])
    end = int(cfg["rules"]["unusual_hour_end"])
    wh_start = int(cfg["working_hours"]["start"])
    wh_end = int(cfg["working_hours"]["end"])
    hour = int(row.get("login_hour", -1))
    if (
        start <= hour <= end
        and row.get("event_type") == "login"
        and row.get("login_status") == "success"
    ):
        return (
            f"Successful login at {hour:02d}:00 "
            f"(outside working hours {wh_start:02d}-{wh_end:02d})"
        )
    return None


def _rule_new_device(row: pd.Series, cfg: Dict) -> Optional[str]:
    if int(row.get("new_device", 0)) == 1 and row.get("event_type") == "login":
        return f"Login from a previously unseen device '{row.get('device_id')}'"
    return None


def _rule_new_mfa_device(row: pd.Series, cfg: Dict) -> Optional[str]:
    if row.get("mfa_event") == "new_device_registered":
        return "A new MFA device was registered for this account"
    return None


def _rule_privilege_escalation(row: pd.Series, cfg: Dict) -> Optional[str]:
    if int(row.get("privilege_escalation", 0)) == 1:
        return (
            f"Privilege escalation to '{row.get('privilege_level')}' "
            f"via '{row.get('event_type')}' on '{row.get('destination_host')}'"
        )
    return None


def _rule_multiple_host_access(row: pd.Series, cfg: Dict) -> Optional[str]:
    threshold = int(cfg["rules"]["host_access_threshold"])
    window = int(cfg["rules"]["host_access_window_minutes"])
    count = int(row.get("unique_host_count", 0))
    if count >= threshold and row.get("event_type") == "server_access":
        return (
            f"Access to {count} distinct hosts by '{row['username']}' "
            f"within {window} min (lateral-movement pattern)"
        )
    return None


def _rule_new_privileged_account(row: pd.Series, cfg: Dict) -> Optional[str]:
    if (
        row.get("event_type") == "account_creation"
        and str(row.get("privilege_level", "")).lower() == "admin"
    ):
        return (
            f"New privileged account created on '{row.get('destination_host')}' "
            f"by '{row['username']}'"
        )
    return None


def _rule_credential_access(row: pd.Series, cfg: Dict) -> Optional[str]:
    if int(row.get("credential_access", 0)) == 1:
        return (
            f"Simulated credential-access activity: process "
            f"'{row.get('process_name')}' on '{row.get('destination_host')}'"
        )
    return None


# ---------------------------------------------------------------------------
# Rule registry (built from config so risk points stay configurable)
# ---------------------------------------------------------------------------
def build_rules(cfg: Dict) -> List[Rule]:
    w = cfg["risk_weights"]
    return [
        Rule(
            rule_id="R1",
            name="Multiple Failed Logins",
            description="User exceeded the failed-login threshold inside a short window.",
            risk_points=int(w["multiple_failed_logins"]),
            mitre=["T1110 - Brute Force"],
            check=_rule_multiple_failed_logins,
        ),
        Rule(
            rule_id="R2",
            name="Unusual Login Time",
            description="Successful login outside the user's normal working hours.",
            risk_points=int(w["unusual_login_time"]),
            mitre=["T1078 - Valid Accounts"],
            check=_rule_unusual_login_time,
        ),
        Rule(
            rule_id="R3",
            name="New / Unrecognised Device",
            description="Login from a device not previously associated with the user.",
            risk_points=int(w["new_device"]),
            mitre=["T1078 - Valid Accounts"],
            check=_rule_new_device,
        ),
        Rule(
            rule_id="R4",
            name="New MFA Device Registered",
            description="A new MFA device was registered — classic MFA-fatigue follow-on.",
            risk_points=int(w["new_mfa_device"]),
            mitre=["T1098.005 - Account Manipulation: Device Registration"],
            check=_rule_new_mfa_device,
        ),
        Rule(
            rule_id="R5",
            name="Privilege Escalation",
            description="User account gained higher privileges during the session.",
            risk_points=int(w["privilege_escalation"]),
            mitre=["T1098 - Account Manipulation"],
            check=_rule_privilege_escalation,
        ),
        Rule(
            rule_id="R6",
            name="Multiple Host Access (Lateral Movement)",
            description="Rapid access to many distinct hosts — lateral-movement pattern.",
            risk_points=int(w["multiple_host_access"]),
            mitre=["T1021 - Remote Services"],
            check=_rule_multiple_host_access,
        ),
        Rule(
            rule_id="R7",
            name="New Privileged Account Created",
            description="Creation of a new administrative account — persistence pattern.",
            risk_points=int(w["new_privileged_account"]),
            mitre=["T1136.001 - Create Account: Local Account"],
            check=_rule_new_privileged_account,
        ),
        Rule(
            rule_id="R8",
            name="Credential-Access Activity",
            description="Simulated credential-access process observed on a host.",
            risk_points=int(w["credential_access"]),
            mitre=["T1003 - OS Credential Dumping"],
            check=_rule_credential_access,
        ),
    ]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
def evaluate_rules(df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Apply every rule to every row and return a long-form hits table."""
    rules = build_rules(cfg)
    hits: List[dict] = []

    for _, row in df.iterrows():
        for rule in rules:
            evidence = rule.evaluate(row, cfg)
            if evidence is None:
                continue
            hits.append({
                "event_id": row["event_id"],
                "timestamp": row["timestamp"],
                "username": row["username"],
                "source_ip": row.get("source_ip"),
                "destination_host": row.get("destination_host"),
                "event_type": row.get("event_type"),
                "rule_id": rule.rule_id,
                "rule_name": rule.name,
                "evidence": evidence,
                "risk_points": rule.risk_points,
                "mitre": "; ".join(rule.mitre),
                "label": int(row.get("label", 0)),
            })

    return pd.DataFrame(hits)


def summarise_per_event(df: pd.DataFrame, hits: pd.DataFrame) -> pd.DataFrame:
    """Enrich the original events with a compact rule summary per event."""
    if hits.empty:
        out = df.copy()
        out["triggered_rules"] = ""
        out["rule_count"] = 0
        out["rule_score"] = 0
        return out

    grouped = hits.groupby("event_id").agg(
        triggered_rules=("rule_id", lambda s: ",".join(sorted(set(s)))),
        rule_count=("rule_id", "count"),
        rule_score=("risk_points", "sum"),
    ).reset_index()

    return df.merge(grouped, on="event_id", how="left").fillna(
        {"triggered_rules": "", "rule_count": 0, "rule_score": 0}
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_summary(hits: pd.DataFrame, events: pd.DataFrame) -> None:
    total_events = len(events)
    events_with_hit = int((events["rule_count"] > 0).sum())
    total_hits = len(hits)
    flagged_susp = int(
        ((events["rule_count"] > 0) & (events["label"] == 1)).sum()
    )
    flagged_norm = int(
        ((events["rule_count"] > 0) & (events["label"] == 0)).sum()
    )

    print("\n" + "=" * 66)
    print("  PHASE 2 — RULE-BASED DETECTION")
    print("=" * 66)
    print(f"  Events scanned            : {total_events}")
    print(f"  Events with >=1 rule hit  : {events_with_hit}  "
          f"({events_with_hit / total_events:.1%})")
    print(f"  Total rule hits           : {total_hits}")
    print(f"  ... on suspicious events  : {flagged_susp}")
    print(f"  ... on normal   events    : {flagged_norm}")
    print("-" * 66)
    print("  Rule breakdown:")
    if hits.empty:
        print("    (no rule hits)")
    else:
        breakdown = hits.groupby(["rule_id", "rule_name"]).size().reset_index(name="hits")
        for _, r in breakdown.iterrows():
            print(f"    {r['rule_id']}  {r['rule_name']:<38} {r['hits']:>6} hits")
    print("=" * 66 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 — rule-based detection.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--input", default="data/raw/security_logs.csv")
    parser.add_argument("--hits-out", default="data/processed/rule_hits.csv")
    parser.add_argument("--events-out", default="data/processed/events_with_rules.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f"[*] Loading logs from {args.input} ...")
    df = load_logs(args.input)
    print(f"[*] Loaded {len(df)} events.")

    print("[*] Applying rules ...")
    hits = evaluate_rules(df, cfg)
    events = summarise_per_event(df, hits)

    os.makedirs(os.path.dirname(args.hits_out), exist_ok=True)
    hits.to_csv(args.hits_out, index=False)
    events.to_csv(args.events_out, index=False)

    _print_summary(hits, events)
    print(f"[+] Rule hits saved to   : {args.hits_out}")
    print(f"[+] Enriched events to   : {args.events_out}\n")


if __name__ == "__main__":
    main()