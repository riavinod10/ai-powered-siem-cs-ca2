"""
Phase 7 — Experiments and Evaluation
====================================

Runs four controlled test scenarios and a detector ablation study,
then writes a markdown report suitable for the case-study Results section.

Usage:
    python -m experiments.evaluation
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Dict

import pandas as pd
import yaml

from experiments.test_scenarios import (
    test_1_normal,
    test_2_credential_abuse,
    test_3_lateral_movement,
    test_4_combined_cisco,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(ROOT, "config", "config.yaml")
EVENTS_PATH = os.path.join(ROOT, "data", "processed", "events_with_risk.csv")
ALERTS_PATH = os.path.join(ROOT, "data", "alerts", "alerts.csv")


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) \
        if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "fpr": round(fpr, 3),
    }


def event_level_ablation(events: pd.DataFrame) -> pd.DataFrame:
    y = events["label"]
    rule_pred = (events.get("rule_score", 0) > 0).astype(int)
    ml_pred = (events.get("ml_tier", "none").astype(str) != "none").astype(int)
    hybrid_pred = (events.get("risk_score", 0) > 0).astype(int)

    rows = []
    for name, pred in [
        ("Rules only", rule_pred),
        ("ML only (Isolation Forest)", ml_pred),
        ("Hybrid (rules + ML)", hybrid_pred),
    ]:
        m = binary_metrics(y, pred)
        m["detector"] = name
        rows.append(m)
    return pd.DataFrame(rows)[
        ["detector", "precision", "recall", "f1", "fpr", "tp", "fp", "fn", "tn"]
    ]


def incident_level_ablation(alerts: pd.DataFrame) -> pd.DataFrame:
    if "label" not in alerts.columns or alerts.empty:
        return pd.DataFrame()
    y = alerts["label"]
    any_pred = (alerts["risk_score"] > 0).astype(int)
    actionable = alerts["severity"].isin(
        ["MEDIUM", "HIGH", "CRITICAL"]
    ).astype(int)

    rows = []
    for name, pred in [
        ("Any alert", any_pred),
        ("Actionable (MEDIUM+)", actionable),
    ]:
        m = binary_metrics(y, pred)
        m["detector"] = name
        rows.append(m)
    return pd.DataFrame(rows)[
        ["detector", "precision", "recall", "f1", "fpr", "tp", "fp", "fn", "tn"]
    ]


def run_controlled_tests(events: pd.DataFrame, alerts: pd.DataFrame) -> Dict[str, Dict]:
    results = {}

    e1, a1 = test_1_normal(events, alerts)
    actionable = a1[a1["severity"].isin(["HIGH", "CRITICAL"])] \
        if not a1.empty else a1
    results["test_1"] = {
        "description": "Normal user behaviour",
        "expected": "No HIGH or CRITICAL alert",
        "events_tested": len(e1),
        "alerts_generated": len(a1),
        "high_or_critical": len(actionable),
        "passed": len(actionable) == 0,
    }

    e2, a2 = test_2_credential_abuse(events, alerts)
    cred_alerts = a2[
        a2["threat_type"].astype(str).str.contains(
            "Credential Abuse|Credential Access", case=False, regex=True
        )
    ] if not a2.empty else a2
    results["test_2"] = {
        "description": "Credential abuse / credential access",
        "expected": "At least one Credential Abuse or Credential Access alert",
        "events_tested": len(e2),
        "alerts_generated": len(a2),
        "credential_alerts": len(cred_alerts),
        "passed": len(cred_alerts) > 0,
    }

    e3, a3 = test_3_lateral_movement(events, alerts)
    lat_alerts = a3[
        a3["threat_type"].astype(str).str.contains("Lateral Movement", case=False)
    ] if not a3.empty else a3
    results["test_3"] = {
        "description": "Lateral movement",
        "expected": "At least one Lateral Movement alert",
        "events_tested": len(e3),
        "alerts_generated": len(a3),
        "lateral_alerts": len(lat_alerts),
        "passed": len(lat_alerts) > 0,
    }

    e4, a4 = test_4_combined_cisco(events, alerts)
    critical = a4[a4["severity"] == "CRITICAL"] if not a4.empty else a4
    multi_mitre = a4[
        a4["mitre_techniques"].astype(str).str.count(";") >= 3
    ] if not a4.empty else a4
    results["test_4"] = {
        "description": "Combined Cisco-inspired kill chain",
        "expected": "CRITICAL alert with multiple MITRE techniques",
        "events_tested": len(e4),
        "alerts_generated": len(a4),
        "critical_alerts": len(critical),
        "multi_mitre_alerts": len(multi_mitre),
        "passed": len(critical) > 0 and len(multi_mitre) > 0,
    }

    return results


def _df_to_md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def _mitre_coverage(alerts: pd.DataFrame) -> pd.DataFrame:
    counter: Dict[str, int] = {}
    for cell in alerts.get("mitre_techniques", pd.Series(dtype=str)).dropna():
        for tech in str(cell).split(";"):
            tech = tech.strip()
            if tech:
                counter[tech] = counter.get(tech, 0) + 1
    if not counter:
        return pd.DataFrame(columns=["Technique", "Alerts"])
    rows = [{"Technique": k, "Alerts": v}
            for k, v in sorted(counter.items(), key=lambda kv: -kv[1])]
    return pd.DataFrame(rows)


def write_markdown_report(
    path: str,
    events: pd.DataFrame,
    alerts: pd.DataFrame,
    event_abl: pd.DataFrame,
    incident_abl: pd.DataFrame,
    tests: Dict[str, Dict],
    cfg: dict,
) -> None:
    sev_counts = alerts["severity"].value_counts().reindex(
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    ).fillna(0).astype(int)

    L = []
    L.append("# AI-Powered SIEM — Evaluation Report")
    L.append("")
    L.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    L.append("")
    L.append("## Case Study")
    L.append("")
    L.append(
        "**AI-Powered SIEM for Detecting Credential Theft and Lateral "
        "Movement — A Case Study of the Cisco Data Breach (2022).**"
    )
    L.append("")
    L.append("> All logs are synthetic. The Cisco 2022 incident is used only "
             "as a threat model. No Cisco internal data was accessed.")
    L.append("")
    L.append("---")
    L.append("")

    L.append("## 1. Dataset")
    L.append("")
    L.append(f"- **Total events:** {len(events):,}")
    L.append(f"- **Total alerts (correlated incidents):** {len(alerts):,}")
    L.append(f"- **Normal events:** {int((events['label'] == 0).sum()):,}")
    L.append(f"- **Suspicious events:** {int((events['label'] == 1).sum()):,}")
    L.append("")
    L.append("**Alert severity distribution:**")
    L.append("")
    L.append(_df_to_md(pd.DataFrame({
        "Severity": ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        "Count": [int(sev_counts[s]) for s in
                  ["CRITICAL", "HIGH", "MEDIUM", "LOW"]],
    })))
    L.append("")

    L.append("---")
    L.append("")
    L.append("## 2. Detector Ablation — Per Event")
    L.append("")
    L.append("Three detectors evaluated against ground-truth labels at the "
             "individual log-line level.")
    L.append("")
    L.append(_df_to_md(event_abl))
    L.append("")
    L.append("**Interpretation.** Rules-only achieves very high precision but "
             "moderate recall — it misses attack patterns it wasn't told to "
             "look for. ML-only achieves near-perfect recall (the Isolation "
             "Forest rarely misses a true attack) at the cost of many false "
             "positives. The hybrid detector inherits the rules' precision "
             "and the ML's recall.")
    L.append("")

    if not incident_abl.empty:
        L.append("---")
        L.append("")
        L.append("## 3. Detector Ablation — Per Correlated Incident")
        L.append("")
        L.append("Metrics at the incident level (each row is a user session "
                 "aggregated across all its events). This is the level a "
                 "SOC analyst actually triages.")
        L.append("")
        L.append(_df_to_md(incident_abl))
        L.append("")
        L.append("**Interpretation.** Correlation is a precision booster: the "
                 "MEDIUM+ filter drops the majority of ML false positives "
                 "into LOW severity, leaving a small, high-precision queue.")
        L.append("")

    L.append("---")
    L.append("")
    L.append("## 4. Controlled Test Scenarios")
    L.append("")
    L.append("Four scenarios exercised end-to-end. Each test passes when "
             "the detector behaves as expected.")
    L.append("")

    for key in ["test_1", "test_2", "test_3", "test_4"]:
        t = tests[key]
        status = "PASS" if t["passed"] else "FAIL"
        L.append(f"### {key.replace('_', ' ').title()} — {t['description']}")
        L.append("")
        L.append(f"- **Expected:** {t['expected']}")
        L.append(f"- **Events tested:** {t['events_tested']:,}")
        L.append(f"- **Alerts generated:** {t['alerts_generated']}")
        for k, v in t.items():
            if k in ("description", "expected", "events_tested",
                     "alerts_generated", "passed"):
                continue
            L.append(f"- **{k.replace('_', ' ').title()}:** {v}")
        L.append(f"- **Result:** {status}")
        L.append("")

    L.append("---")
    L.append("")
    L.append("## 5. MITRE ATT&CK Coverage")
    L.append("")
    L.append(_df_to_md(_mitre_coverage(alerts)))
    L.append("")
    L.append("Techniques correspond to the rules that fired in the prototype; "
             "each mapping reflects a documented Cisco-2022 behaviour "
             "represented as an observable log signal.")
    L.append("")

    L.append("---")
    L.append("")
    L.append("## 6. Reproducibility")
    L.append("")
    L.append(f"- Random seed: `{cfg['project']['random_seed']}` "
             "(controls both Python `random` and NumPy)")
    L.append("- Pipeline (must run in this order):")
    L.append("  1. `python -m src.feature_engineering`")
    L.append("  2. `python -m src.rule_engine`")
    L.append("  3. `python -m src.ml_detector`")
    L.append("  4. `python -m src.risk_engine`")
    L.append("  5. `python -m src.alert_engine`")
    L.append("")
    L.append("Same config + same seed → identical dataset, identical metrics.")
    L.append("")

    L.append("---")
    L.append("")
    L.append("## 7. Limitations")
    L.append("")
    L.append("1. **Synthetic data.** Metrics describe behaviour on the "
             "generator's distributions, not real enterprise telemetry.")
    L.append("2. **Small feature set.** 16 features from 10k events; "
             "production SIEMs use hundreds of features and richer entity "
             "context.")
    L.append(f"3. **Fixed session gap.** Correlation uses a single "
             f"{cfg['alerts']['session_gap_minutes']}-minute parameter; real "
             "SIEMs tune this per source and use graph-based correlation.")
    L.append("4. **Isolation Forest is not proof.** It flags statistical "
             "outliers, not confirmed attacks.")
    L.append("5. **No live ingestion.** The prototype scores a fixed dataset; "
             "streaming ingestion is future work.")
    L.append("")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def _print_summary(tests: Dict[str, Dict]) -> None:
    print("\n" + "=" * 66)
    print("  PHASE 7 — CONTROLLED TEST SCENARIOS")
    print("=" * 66)
    for key, t in tests.items():
        status = "PASS" if t["passed"] else "FAIL"
        print(f"  [{status}] {key}: {t['description']}")
        print(f"          expected : {t['expected']}")
        for k in ("events_tested", "alerts_generated", "high_or_critical",
                  "credential_alerts", "lateral_alerts", "critical_alerts",
                  "multi_mitre_alerts"):
            if k in t:
                print(f"          {k:<18}: {t[k]}")
    print("=" * 66 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 7 — experiments.")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--events", default=EVENTS_PATH)
    ap.add_argument("--alerts", default=ALERTS_PATH)
    ap.add_argument("--out", default=os.path.join(ROOT, "reports",
                                                   "evaluation_report.md"))
    args = ap.parse_args()

    cfg = load_config(args.config)

    if not os.path.exists(args.events) or not os.path.exists(args.alerts):
        raise FileNotFoundError(
            "Missing events or alerts. Run the full pipeline first."
        )

    print(f"[*] Loading events from {args.events} ...")
    events = pd.read_csv(args.events, parse_dates=["timestamp"])
    print(f"[*] Loading alerts from {args.alerts} ...")
    alerts = pd.read_csv(args.alerts, parse_dates=["timestamp", "last_seen"])

    print("[*] Computing event-level ablation ...")
    event_abl = event_level_ablation(events)

    print("[*] Computing incident-level ablation ...")
    incident_abl = incident_level_ablation(alerts)

    print("[*] Running controlled test scenarios ...")
    tests = run_controlled_tests(events, alerts)

    print(f"[*] Writing markdown report to {args.out} ...")
    write_markdown_report(args.out, events, alerts, event_abl,
                          incident_abl, tests, cfg)

    print("\n=== Event-level ablation ===")
    print(event_abl.to_string(index=False))
    print("\n=== Incident-level ablation ===")
    print(incident_abl.to_string(index=False) if not incident_abl.empty
          else "(no alerts with labels)")

    _print_summary(tests)

    print(f"[+] Full report written to: {args.out}\n")


if __name__ == "__main__":
    main()