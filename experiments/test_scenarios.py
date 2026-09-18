"""
Phase 7 — Controlled Test Scenarios
====================================

Helpers to isolate the four controlled tests required for the case study.

    TEST 1 — Normal user behaviour          → expect no CRITICAL/HIGH alert
    TEST 2 — Credential abuse               → expect a Credential Abuse alert
    TEST 3 — Lateral movement               → expect a Lateral Movement alert
    TEST 4 — Combined Cisco-inspired chain  → expect CRITICAL + multi-technique
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def test_1_normal(events: pd.DataFrame,
                  alerts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mask_e = events["scenario"].astype(str) == "normal"
    mask_a = alerts["scenarios"].astype(str).str.contains("normal", na=False)
    return events[mask_e], alerts[mask_a]


def test_2_credential_abuse(events: pd.DataFrame,
                            alerts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pattern = "credential_brute_force|credential_access"
    mask_e = events["scenario"].astype(str).str.contains(pattern, regex=True)
    mask_a = alerts["scenarios"].astype(str).str.contains(pattern, regex=True)
    return events[mask_e], alerts[mask_a]


def test_3_lateral_movement(events: pd.DataFrame,
                            alerts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mask_e = events["scenario"].astype(str).str.contains("lateral_movement")
    mask_a = alerts["scenarios"].astype(str).str.contains("lateral_movement")
    return events[mask_e], alerts[mask_a]


def test_4_combined_cisco(events: pd.DataFrame,
                          alerts: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    mask_e = events["scenario"].astype(str).str.contains("combined_cisco")
    mask_a = alerts["scenarios"].astype(str).str.contains("combined_cisco")
    return events[mask_e], alerts[mask_a]