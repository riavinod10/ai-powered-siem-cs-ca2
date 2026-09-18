"""
Phase 2 helper — load and lightly prepare the raw log dataset.

This module intentionally does very little:
  - loads the CSV from Phase 1
  - parses the timestamp column
  - assigns a stable, human-readable event_id

Real feature engineering (rolling counts, encoding, scaling) happens later
in the ML pipeline (Phase 3). The rule engine can work directly off the
raw columns because Phase 1 already computed the windowed features.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd

REQUIRED_COLUMNS = {
    "timestamp", "username", "source_ip", "destination_host",
    "event_type", "login_status", "device_id", "login_hour",
    "privilege_level", "new_device", "mfa_event", "process_name",
    "failed_login_count", "unique_host_count", "unique_ip_count",
    "login_frequency", "remote_login", "privilege_escalation",
    "credential_access", "scenario", "label",
}


def load_logs(path: str) -> pd.DataFrame:
    """Load the raw security log CSV and add an event_id column."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Log file not found: '{path}'. "
            f"Run `python -m src.data_generator` first to create it."
        )

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Log file '{path}' is missing required columns: {sorted(missing)}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("Some timestamps could not be parsed.")

    # Stable, sortable event id. e.g. EVT-000001
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.insert(0, "event_id", [f"EVT-{i + 1:06d}" for i in range(len(df))])

    return df


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    """Columns that will be used as features by the ML detector (Phase 3).

    Explicitly excludes event_id, scenario and label to avoid leakage.
    """
    candidates = [
        "failed_login_count", "unique_host_count", "unique_ip_count",
        "login_frequency", "login_hour", "new_device", "remote_login",
        "privilege_escalation", "credential_access",
    ]
    return [c for c in candidates if c in df.columns]