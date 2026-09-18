"""
Phase 5 — Elasticsearch Client
==============================

Thin wrapper around the official `elasticsearch` Python client.

Responsibilities:
  * create the `siem-logs` and `siem-alerts` indices with explicit mappings
  * bulk-index events and alerts from the CSVs produced by Phases 1.5-4b
  * expose simple search helpers used by the Streamlit dashboard
  * degrade gracefully when Elasticsearch is not reachable

Usage:
    python -m src.elasticsearch_client --recreate
    python -m src.elasticsearch_client --status
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import yaml

DEFAULT_CONFIG = os.path.join("config", "config.yaml")

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
    ES_IMPORT_OK = True
except Exception:
    Elasticsearch = None  # type: ignore
    bulk = None           # type: ignore
    ES_IMPORT_OK = False


def load_config(path: str = DEFAULT_CONFIG) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Mappings — aligned with the current event/alert schema
# ---------------------------------------------------------------------------
LOG_MAPPING: Dict[str, Any] = {
    "mappings": {
        "properties": {
            "event_id":             {"type": "keyword"},
            "timestamp":            {"type": "date"},
            "username":             {"type": "keyword"},
            "source_ip":            {"type": "ip"},
            "destination_host":     {"type": "keyword"},
            "event_type":           {"type": "keyword"},
            "login_status":         {"type": "keyword"},
            "device_id":            {"type": "keyword"},
            "login_hour":           {"type": "integer"},
            "privilege_level":      {"type": "keyword"},
            "new_device":           {"type": "integer"},
            "mfa_event":            {"type": "keyword"},
            "process_name":         {"type": "keyword"},
            "failed_login_count":   {"type": "integer"},
            "unique_host_count":    {"type": "integer"},
            "unique_ip_count":      {"type": "integer"},
            "login_frequency":      {"type": "integer"},
            "remote_login":         {"type": "integer"},
            "privilege_escalation": {"type": "integer"},
            "credential_access":    {"type": "integer"},
            "scenario":             {"type": "keyword"},
            "label":                {"type": "integer"},
            # --- Phase 1.5 context features ---
            "is_sensitive_host":    {"type": "integer"},
            "is_dc_access":         {"type": "integer"},
            "event_type_code":      {"type": "integer"},
            "user_avg_login_hour":  {"type": "float"},
            "hour_deviation":       {"type": "float"},
            "time_since_last_event_min": {"type": "float"},
            "failed_success_ratio": {"type": "float"},
            # --- Phase 2/3 rule + ML output ---
            "triggered_rules":      {"type": "keyword"},
            "rule_count":           {"type": "integer"},
            "rule_score":           {"type": "integer"},
            "anomaly_score":        {"type": "float"},
            "anomaly_prediction":   {"type": "integer"},
            "is_anomaly":           {"type": "integer"},
            "ml_tier":              {"type": "keyword"},
            "ml_bonus":             {"type": "integer"},
            # --- Phase 4a final risk ---
            "risk_score":           {"type": "integer"},
            "severity":             {"type": "keyword"},
        }
    }
}

ALERT_MAPPING: Dict[str, Any] = {
    "mappings": {
        "properties": {
            "alert_id":             {"type": "keyword"},
            "incident_id":          {"type": "keyword"},
            "timestamp":            {"type": "date"},
            "last_seen":            {"type": "date"},
            "duration_minutes":     {"type": "float"},
            "event_count":          {"type": "integer"},
            "username":             {"type": "keyword"},
            "source_ip":            {"type": "ip"},
            "affected_hosts":       {"type": "keyword"},
            "threat_type":          {"type": "keyword"},
            "triggered_indicators": {"type": "text"},
            "triggered_rule_ids":   {"type": "keyword"},
            "mitre_techniques":     {"type": "text"},
            "ml_anomaly_score":     {"type": "float"},
            "ml_tier":              {"type": "keyword"},
            "rule_score":           {"type": "integer"},
            "ml_bonus":             {"type": "integer"},
            "risk_score":           {"type": "integer"},
            "severity":             {"type": "keyword"},
            "status":               {"type": "keyword"},
            "recommended_response": {"type": "text"},
            "label":                {"type": "integer"},
            "scenarios":            {"type": "keyword"},
        }
    }
}


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class SIEMElasticClient:
    def __init__(self, host: str, log_index: str, alert_index: str):
        self.host = host
        self.log_index = log_index
        self.alert_index = alert_index
        self._client: Optional["Elasticsearch"] = None

    def connect(self, timeout: int = 3) -> bool:
        if not ES_IMPORT_OK:
            print("[!] elasticsearch package not installed.", file=sys.stderr)
            return False
        try:
            self._client = Elasticsearch(self.host, request_timeout=timeout)
            info = self._client.info()
            print(f"[+] Connected to Elasticsearch at {self.host} "
                  f"(cluster: {info.get('cluster_name')}, "
                  f"version: {info['version']['number']})")
            return True
        except Exception as exc:
            print(f"[!] Elasticsearch not reachable at {self.host}: {exc}",
                  file=sys.stderr)
            self._client = None
            return False

    def available(self) -> bool:
        return self._client is not None and bool(self._client.ping())

    def recreate_indices(self) -> None:
        self._require_client()
        for name, mapping in ((self.log_index, LOG_MAPPING),
                              (self.alert_index, ALERT_MAPPING)):
            if self._client.indices.exists(index=name):
                self._client.indices.delete(index=name)
                print(f"[*] Deleted existing index '{name}'.")
            self._client.indices.create(index=name, body=mapping)
            print(f"[+] Created index '{name}'.")

    def ensure_indices(self) -> None:
        self._require_client()
        for name, mapping in ((self.log_index, LOG_MAPPING),
                              (self.alert_index, ALERT_MAPPING)):
            if not self._client.indices.exists(index=name):
                self._client.indices.create(index=name, body=mapping)
                print(f"[+] Created index '{name}'.")

    def _df_to_actions(self, df: pd.DataFrame, index: str) -> Iterable[dict]:
        for _, row in df.iterrows():
            doc = row.where(pd.notnull(row), None).to_dict()
            for k, v in list(doc.items()):
                if isinstance(v, pd.Timestamp):
                    doc[k] = v.isoformat()
            yield {"_index": index, "_source": doc}

    def index_dataframe(self, df: pd.DataFrame, index: str,
                        chunk: int = 500) -> int:
        self._require_client()
        if df.empty:
            return 0
        actions = list(self._df_to_actions(df, index))
        success, errors = bulk(self._client, actions, chunk_size=chunk,
                               raise_on_error=False, stats_only=False)
        if errors:
            print(f"[!] {len(errors)} documents failed to index. "
                  f"First error:\n    {errors[0]}")
        self._client.indices.refresh(index=index)
        return success

    def count(self, index: str) -> int:
        self._require_client()
        return int(self._client.count(index=index)["count"])

    def search(self, index: str, query: Optional[dict] = None,
               size: int = 100, sort: Optional[list] = None) -> List[dict]:
        self._require_client()
        body: Dict[str, Any] = {"size": size}
        if query:
            body["query"] = query
        if sort:
            body["sort"] = sort
        res = self._client.search(index=index, body=body)
        return [hit["_source"] for hit in res["hits"]["hits"]]

    def _require_client(self) -> None:
        if self._client is None:
            raise RuntimeError("Elasticsearch client is not connected.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_csv(path: str, parse_dates: List[str]) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return pd.read_csv(path, parse_dates=parse_dates)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 5 — Elasticsearch client.")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--index", action="store_true")
    ap.add_argument("--events", default="data/processed/events_with_risk.csv")
    ap.add_argument("--alerts", default="data/alerts/alerts.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    es_cfg = cfg["elasticsearch"]

    client = SIEMElasticClient(
        es_cfg["host"], es_cfg["log_index"], es_cfg["alert_index"]
    )
    if not client.connect():
        print("\n[i] To start Elasticsearch:")
        print("      docker compose up -d")
        print("    Wait ~30 seconds, then retry.\n")
        sys.exit(1)

    if args.recreate:
        client.recreate_indices()
    else:
        client.ensure_indices()

    if args.index or args.recreate:
        print(f"[*] Loading events from {args.events} ...")
        events = _load_csv(args.events, ["timestamp"])
        n_events = client.index_dataframe(events, es_cfg["log_index"])
        print(f"[+] Indexed {n_events} events into '{es_cfg['log_index']}'.")

        print(f"[*] Loading alerts from {args.alerts} ...")
        alerts = _load_csv(args.alerts, ["timestamp", "last_seen"])
        n_alerts = client.index_dataframe(alerts, es_cfg["alert_index"])
        print(f"[+] Indexed {n_alerts} alerts into '{es_cfg['alert_index']}'.")

    n_logs = client.count(es_cfg["log_index"])
    n_alerts = client.count(es_cfg["alert_index"])
    print("\n" + "=" * 60)
    print("  ELASTICSEARCH STATUS")
    print("=" * 60)
    print(f"  Host         : {es_cfg['host']}")
    print(f"  Log index    : {es_cfg['log_index']:<20} {n_logs:>6} documents")
    print(f"  Alert index  : {es_cfg['alert_index']:<20} {n_alerts:>6} documents")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()