"""
Phase 1 — Synthetic Security Log Generator
==========================================

Generates a synthetic SIEM-style security log dataset for the prototype:

    "AI-Powered SIEM for Detecting Credential Theft and Lateral Movement:
     A Case Study of the Cisco Data Breach (2022)"

IMPORTANT / ACADEMIC HONESTY
----------------------------
These are SYNTHETIC logs. We do NOT have access to Cisco's internal data.
The suspicious scenarios are *Cisco-2022-inspired simulations* that reproduce
publicly documented attack BEHAVIOURS as observable log events.

No malware, exploit code, credential theft, or offensive technique is
implemented. "credential_access" is a boolean column in a CSV, and
"sim_credential_access_tool" is a string value — nothing executable.

Usage
-----
    python -m src.data_generator
    python -m src.data_generator --records 10000 --out data/raw/security_logs.csv
"""

from __future__ import annotations

import argparse
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = os.path.join("config", "config.yaml")

# Final column order of the produced dataset
COLUMN_ORDER = [
    "timestamp",
    "username",
    "source_ip",
    "destination_host",
    "event_type",
    "login_status",
    "device_id",
    "login_hour",
    "privilege_level",
    "new_device",
    "mfa_event",
    "process_name",
    "failed_login_count",
    "unique_host_count",
    "unique_ip_count",
    "login_frequency",
    "remote_login",
    "privilege_escalation",
    "credential_access",
    "scenario",   # for evaluation / timeline only — NOT an ML feature
    "label",      # 0 = normal, 1 = suspicious — NOT an ML feature
]

# RFC 5737 documentation ranges — guaranteed never to be real hosts
EXTERNAL_IP_BLOCKS = ["192.0.2", "198.51.100", "203.0.113"]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def load_config(path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load the YAML config file with a clear error if it is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config file not found at '{path}'. "
            f"Run the script from the project root directory."
        )
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file '{path}' did not parse into a dictionary.")
    return cfg


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class SecurityLogGenerator:
    """Builds a synthetic security-log dataset with normal and attack activity."""

    # ---------------------------------------------------------------- setup
    def __init__(self, config: Dict):
        self.cfg = config

        seed = int(config["project"]["random_seed"])
        random.seed(seed)
        np.random.seed(seed)

        gen = config["data_generation"]
        self.total_records = int(gen["total_records"])
        self.normal_ratio = float(gen["normal_ratio"])
        self.start_date = datetime.fromisoformat(str(gen["start_date"]))
        self.days = int(gen["days"])
        self.feature_window = int(gen["feature_window_minutes"])

        self.work_start = int(config["working_hours"]["start"])
        self.work_end = int(config["working_hours"]["end"])

        self.internal_hosts = list(config["hosts"]["internal"])
        self.sensitive_hosts = list(config["hosts"]["sensitive"])

        self.users = self._build_users()

        self.normal_target = int(self.total_records * self.normal_ratio)
        self.suspicious_target = self.total_records - self.normal_target

    def _build_users(self) -> List[Dict]:
        """Create stable per-user behavioural profiles (home IP, device, hours)."""
        users: List[Dict] = []

        for i in range(1, int(self.cfg["users"]["normal_users"]) + 1):
            users.append({
                "username": f"employee{i:02d}",
                "privilege_level": "user",
                "home_ip": f"10.20.0.{i + 1}",
                "device_id": f"DEV-{1000 + i}",
                "login_hour_mean": 9.5,
                "login_hour_std": 1.2,
                "hosts": self.internal_hosts[:3],
            })

        for i in range(1, int(self.cfg["users"]["admin_users"]) + 1):
            users.append({
                "username": f"admin{i:02d}",
                "privilege_level": "admin",
                "home_ip": f"10.30.0.{i}",
                "device_id": f"DEV-{2000 + i}",
                "login_hour_mean": 10.0,
                "login_hour_std": 1.5,
                "hosts": self.internal_hosts + self.sensitive_hosts[:2],
            })

        return users

    # ------------------------------------------------------------ helpers
    def _base_row(self, ts: datetime, user: Dict, **overrides) -> dict:
        """A default normal-looking row, then apply scenario overrides."""
        row = {
            "timestamp": ts,
            "username": user["username"],
            "source_ip": user["home_ip"],
            "destination_host": "vpn-gateway",
            "event_type": "login",
            "login_status": "success",
            "device_id": user["device_id"],
            "login_hour": ts.hour,
            "privilege_level": user["privilege_level"],
            "new_device": 0,
            "mfa_event": "none",
            "process_name": "NA",
            "failed_login_count": 0,
            "unique_host_count": 0,
            "unique_ip_count": 0,
            "login_frequency": 0,
            "remote_login": 0,
            "privilege_escalation": 0,
            "credential_access": 0,
            "scenario": "normal",
            "label": 0,
        }
        row.update(overrides)
        row["login_hour"] = row["timestamp"].hour  # keep consistent
        return row

    @staticmethod
    def _random_external_ip() -> str:
        block = random.choice(EXTERNAL_IP_BLOCKS)
        return f"{block}.{random.randint(2, 254)}"

    def _random_workday_time(self, user: Dict) -> datetime:
        """A believable in-hours timestamp for a normal user."""
        day = self.start_date + timedelta(days=random.randint(0, self.days - 1))
        hour = float(np.clip(
            np.random.normal(user["login_hour_mean"], user["login_hour_std"]),
            self.work_start,
            self.work_end - 1,
        ))
        return day.replace(
            hour=int(hour),
            minute=random.randint(0, 59),
            second=random.randint(0, 59),
        )

    def _attack_time(self) -> datetime:
        """Attack activity tends to happen outside working hours."""
        day = self.start_date + timedelta(days=random.randint(0, self.days - 1))
        return day.replace(
            hour=random.randint(0, 5),
            minute=random.randint(0, 59),
            second=random.randint(0, 59),
        )

    # ----------------------------------------------------- normal activity
    def _normal_session(self, user: Dict) -> List[dict]:
        """One believable work session: VPN login -> activity -> logout."""
        ts = self._random_workday_time(user)
        events: List[dict] = []

        # 1) VPN login with MFA push
        events.append(self._base_row(
            ts, user,
            destination_host="vpn-gateway",
            event_type="login",
            login_status="success",
            remote_login=1,
            mfa_event="push",
        ))

        # 2) A few internal resource accesses
        for _ in range(random.randint(1, 3)):
            ts += timedelta(minutes=random.randint(3, 25))
            events.append(self._base_row(
                ts, user,
                destination_host=random.choice(user["hosts"]),
                event_type=random.choice(
                    ["file_access", "server_access", "email_access"]
                ),
                login_status="NA",
                remote_login=0,
            ))

        # 3) Logout
        ts += timedelta(minutes=random.randint(5, 30))
        events.append(self._base_row(
            ts, user,
            destination_host="vpn-gateway",
            event_type="logout",
            login_status="NA",
            remote_login=1,
        ))

        return events

    def _generate_normal_rows(self, target: int) -> List[dict]:
        rows: List[dict] = []
        # Sessions are added whole so we never truncate mid-session.
        while len(rows) < target:
            rows.extend(self._normal_session(random.choice(self.users)))
        return rows

    # ----------------------------------------------------- attack scenarios
    # Each scenario is a DISTINCT behavioural fingerprint so the detector
    # has genuinely different things to find.

    def _scenario_brute_force(self, user: Dict) -> List[dict]:
        """RULE 1 fingerprint: repeated failures then a success."""
        events: List[dict] = []
        ts = self._attack_time()
        ip = self._random_external_ip()
        dev = f"DEV-UNKNOWN-{random.randint(100, 999)}"

        for _ in range(random.randint(6, 12)):
            ts += timedelta(seconds=random.randint(4, 45))
            events.append(self._base_row(
                ts, user,
                source_ip=ip,
                destination_host="vpn-gateway",
                event_type="login",
                login_status="failed",
                device_id=dev,
                new_device=1,
                remote_login=1,
                scenario="credential_brute_force",
                label=1,
            ))

        ts += timedelta(minutes=random.randint(1, 3))
        events.append(self._base_row(
            ts, user,
            source_ip=ip,
            destination_host="vpn-gateway",
            event_type="login",
            login_status="success",
            device_id=dev,
            new_device=1,
            remote_login=1,
            scenario="credential_brute_force",
            label=1,
        ))
        return events

    def _scenario_unusual_login(self, user: Dict) -> List[dict]:
        """RULE 2/3/4 fingerprint: odd hour + unknown IP + new device."""
        events: List[dict] = []
        ts = self._attack_time()
        ip = self._random_external_ip()
        dev = f"DEV-UNKNOWN-{random.randint(100, 999)}"

        events.append(self._base_row(
            ts, user,
            source_ip=ip,
            destination_host="vpn-gateway",
            event_type="login",
            login_status="success",
            device_id=dev,
            new_device=1,
            remote_login=1,
            scenario="unusual_login",
            label=1,
        ))

        for _ in range(random.randint(1, 3)):
            ts += timedelta(minutes=random.randint(2, 8))
            events.append(self._base_row(
                ts, user,
                source_ip=ip,
                destination_host=random.choice(self.internal_hosts),
                event_type="server_access",
                login_status="NA",
                device_id=dev,
                new_device=1,
                scenario="unusual_login",
                label=1,
            ))
        return events

    def _scenario_mfa_fatigue(self, user: Dict) -> List[dict]:
        """RULE 4 fingerprint: MFA push spam, then a new MFA device registered."""
        events: List[dict] = []
        ts = self._attack_time()
        ip = self._random_external_ip()

        for _ in range(random.randint(4, 8)):
            ts += timedelta(minutes=random.randint(1, 4))
            events.append(self._base_row(
                ts, user,
                source_ip=ip,
                destination_host="mfa-service",
                event_type="mfa_event",
                login_status="NA",
                mfa_event="push_denied",
                scenario="mfa_fatigue",
                label=1,
            ))

        ts += timedelta(minutes=random.randint(1, 5))
        events.append(self._base_row(
            ts, user,
            source_ip=ip,
            destination_host="mfa-service",
            event_type="mfa_event",
            login_status="NA",
            mfa_event="new_device_registered",
            new_device=1,
            scenario="mfa_fatigue",
            label=1,
        ))

        ts += timedelta(minutes=1)
        events.append(self._base_row(
            ts, user,
            source_ip=ip,
            destination_host="vpn-gateway",
            event_type="login",
            login_status="success",
            new_device=1,
            remote_login=1,
            scenario="mfa_fatigue",
            label=1,
        ))
        return events

    def _scenario_privilege_escalation(self, user: Dict) -> List[dict]:
        """RULE 5 fingerprint: role change to admin + sensitive host access."""
        events: List[dict] = []
        ts = self._attack_time()

        events.append(self._base_row(
            ts, user,
            destination_host="vpn-gateway",
            event_type="login",
            login_status="success",
            remote_login=1,
            scenario="privilege_escalation",
            label=1,
        ))

        ts += timedelta(minutes=random.randint(2, 10))
        events.append(self._base_row(
            ts, user,
            destination_host="admin-portal",
            event_type="privilege_change",
            login_status="NA",
            privilege_level="admin",
            privilege_escalation=1,
            scenario="privilege_escalation",
            label=1,
        ))

        for _ in range(random.randint(1, 2)):
            ts += timedelta(minutes=random.randint(1, 5))
            events.append(self._base_row(
                ts, user,
                destination_host=random.choice(self.sensitive_hosts),
                event_type="server_access",
                login_status="NA",
                privilege_level="admin",
                scenario="privilege_escalation",
                label=1,
            ))
        return events

    def _scenario_credential_access(self, user: Dict) -> List[dict]:
        """RULE 8 fingerprint: simulated credential-access activity."""
        events: List[dict] = []
        ts = self._attack_time()

        events.append(self._base_row(
            ts, user,
            destination_host="vpn-gateway",
            event_type="login",
            login_status="success",
            remote_login=1,
            scenario="credential_access",
            label=1,
        ))

        for _ in range(random.randint(1, 3)):
            ts += timedelta(minutes=random.randint(1, 4))
            events.append(self._base_row(
                ts, user,
                destination_host=random.choice(self.internal_hosts),
                event_type="process_start",
                login_status="NA",
                process_name="sim_credential_access_tool",
                credential_access=1,
                scenario="credential_access",
                label=1,
            ))
        return events

    def _scenario_lateral_movement(self, user: Dict) -> List[dict]:
        """RULE 6 fingerprint: many distinct hosts in a very short window."""
        events: List[dict] = []
        ts = self._attack_time()

        events.append(self._base_row(
            ts, user,
            destination_host="vpn-gateway",
            event_type="login",
            login_status="success",
            remote_login=1,
            scenario="lateral_movement",
            label=1,
        ))

        targets = random.sample(self.internal_hosts, k=3)
        targets += random.sample(self.sensitive_hosts, k=2)
        random.shuffle(targets)

        for host in targets:
            ts += timedelta(seconds=random.randint(30, 120))
            events.append(self._base_row(
                ts, user,
                destination_host=host,
                event_type="server_access",
                login_status="NA",
                scenario="lateral_movement",
                label=1,
            ))
        return events

    def _scenario_new_privileged_account(self, user: Dict) -> List[dict]:
        """RULE 7 fingerprint: a brand-new privileged account appears."""
        events: List[dict] = []
        ts = self._attack_time()
        ip = self._random_external_ip()
        new_account = (
            f"svc_{random.choice(['backup', 'helpdesk', 'ops'])}{random.randint(10, 99)}"
        )

        # account creation event
        events.append(self._base_row(
            ts, user,
            source_ip=ip,
            destination_host="admin-portal",
            event_type="account_creation",
            login_status="NA",
            privilege_level="admin",
            privilege_escalation=1,
            scenario="new_privileged_account",
            label=1,
        ))

        # first login as the new account
        ts += timedelta(minutes=random.randint(1, 5))
        row = self._base_row(
            ts, user,
            source_ip=ip,
            destination_host="vpn-gateway",
            event_type="login",
            login_status="success",
            privilege_level="admin",
            remote_login=1,
            scenario="new_privileged_account",
            label=1,
        )
        row["username"] = new_account
        events.append(row)

        # sensitive host access as the new account
        ts += timedelta(minutes=random.randint(1, 5))
        row = self._base_row(
            ts, user,
            source_ip=ip,
            destination_host=random.choice(self.sensitive_hosts),
            event_type="server_access",
            login_status="NA",
            privilege_level="admin",
            scenario="new_privileged_account",
            label=1,
        )
        row["username"] = new_account
        events.append(row)

        return events

    def _scenario_combined_cisco(self, user: Dict) -> List[dict]:
        """
        Full Cisco-inspired chain, simulated end-to-end:
        failed auth -> VPN login -> MFA fatigue -> new MFA device
        -> privilege escalation -> credential access -> lateral movement
        -> domain controller access.
        """
        events: List[dict] = []
        ts = self._attack_time()
        ip = self._random_external_ip()
        dev = f"DEV-UNKNOWN-{random.randint(100, 999)}"

        # 1) repeated failed authentications
        for _ in range(random.randint(4, 8)):
            ts += timedelta(seconds=random.randint(20, 90))
            events.append(self._base_row(
                ts, user,
                source_ip=ip,
                destination_host="vpn-gateway",
                event_type="login",
                login_status="failed",
                device_id=dev,
                new_device=1,
                remote_login=1,
                scenario="combined_cisco",
                label=1,
            ))

        # 2) successful VPN login
        ts += timedelta(minutes=1)
        events.append(self._base_row(
            ts, user,
            source_ip=ip,
            destination_host="vpn-gateway",
            event_type="login",
            login_status="success",
            device_id=dev,
            new_device=1,
            remote_login=1,
            scenario="combined_cisco",
            label=1,
        ))

        # 3) MFA fatigue + new MFA device registration
        for _ in range(random.randint(3, 5)):
            ts += timedelta(minutes=random.randint(1, 3))
            events.append(self._base_row(
                ts, user,
                source_ip=ip,
                destination_host="mfa-service",
                event_type="mfa_event",
                login_status="NA",
                mfa_event="push_denied",
                device_id=dev,
                new_device=1,
                scenario="combined_cisco",
                label=1,
            ))

        ts += timedelta(minutes=1)
        events.append(self._base_row(
            ts, user,
            source_ip=ip,
            destination_host="mfa-service",
            event_type="mfa_event",
            login_status="NA",
            mfa_event="new_device_registered",
            new_device=1,
            scenario="combined_cisco",
            label=1,
        ))

        # 4) privilege escalation
        ts += timedelta(minutes=2)
        events.append(self._base_row(
            ts, user,
            source_ip=ip,
            destination_host="admin-portal",
            event_type="privilege_change",
            login_status="NA",
            privilege_level="admin",
            privilege_escalation=1,
            new_device=1,
            scenario="combined_cisco",
            label=1,
        ))

        # 5) simulated credential-access activity
        ts += timedelta(minutes=1)
        events.append(self._base_row(
            ts, user,
            source_ip=ip,
            destination_host=random.choice(self.internal_hosts),
            event_type="process_start",
            login_status="NA",
            process_name="sim_credential_access_tool",
            credential_access=1,
            privilege_level="admin",
            scenario="combined_cisco",
            label=1,
        ))

        # 6) lateral movement ending at the domain controller
        path = random.sample(self.internal_hosts, k=3)
        path += random.sample([h for h in self.sensitive_hosts if h != "dc-01"], k=1)
        path.append("dc-01")

        for host in path:
            ts += timedelta(seconds=random.randint(40, 90))
            events.append(self._base_row(
                ts, user,
                source_ip=ip,
                destination_host=host,
                event_type="server_access",
                login_status="NA",
                privilege_level="admin",
                scenario="combined_cisco",
                label=1,
            ))

        return events

    def _generate_attack_rows(self, target: int) -> List[dict]:
        """Mix scenarios with weights so the combined chain stays prominent."""
        weighted = [
            (self._scenario_brute_force, 3),
            (self._scenario_unusual_login, 2),
            (self._scenario_mfa_fatigue, 2),
            (self._scenario_privilege_escalation, 2),
            (self._scenario_credential_access, 2),
            (self._scenario_lateral_movement, 3),
            (self._scenario_new_privileged_account, 1),
            (self._scenario_combined_cisco, 3),
        ]
        pool = [fn for fn, weight in weighted for _ in range(weight)]

        rows: List[dict] = []
        while len(rows) < target:
            rows.extend(random.choice(pool)(random.choice(self.users)))
        return rows

    # ------------------------------------------------- windowed features
    def _add_window_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute backward-looking per-user aggregates over a rolling window.

        Only PAST data is used for each row, so there is no target leakage
        from later events back into earlier ones.
        """
        df = df.copy()
        win = self.feature_window

        for _, grp in df.groupby("username", sort=False):
            idx = grp.index.to_numpy()
            ts = grp["timestamp"].to_numpy()
            status = grp["login_status"].to_numpy()
            host = grp["destination_host"].to_numpy()
            ip = grp["source_ip"].to_numpy()
            etype = grp["event_type"].to_numpy()

            for i in range(len(idx)):
                lo = ts[i] - np.timedelta64(win, "m")
                mask = (ts >= lo) & (ts <= ts[i])

                df.at[idx[i], "failed_login_count"] = int(
                    np.sum(status[mask] == "failed")
                )
                df.at[idx[i], "unique_host_count"] = len(set(host[mask]))
                df.at[idx[i], "unique_ip_count"] = len(set(ip[mask]))

                lo1h = ts[i] - np.timedelta64(60, "m")
                mask1h = (ts >= lo1h) & (ts <= ts[i])
                df.at[idx[i], "login_frequency"] = int(
                    np.sum(etype[mask1h] == "login")
                )

        # Ensure integer dtypes for feature columns
        for col in ("failed_login_count", "unique_host_count",
                    "unique_ip_count", "login_frequency"):
            df[col] = df[col].astype(int)

        return df

    # ------------------------------------------------------------- public
    def generate(self) -> pd.DataFrame:
        """Produce the full dataset, sorted chronologically."""
        normal_rows = self._generate_normal_rows(self.normal_target)
        attack_rows = self._generate_attack_rows(self.suspicious_target)

        df = pd.DataFrame(normal_rows + attack_rows)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = self._add_window_features(df)
        return df[COLUMN_ORDER]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _print_summary(df: pd.DataFrame, out_path: str) -> None:
    total = len(df)
    suspicious = int(df["label"].sum())
    normal = total - suspicious

    print("\n" + "=" * 62)
    print("  PHASE 1 — SYNTHETIC SECURITY LOG DATASET")
    print("=" * 62)
    print(f"  Output file          : {out_path}")
    print(f"  Total records        : {total}")
    print(f"  Normal   (label=0)   : {normal}  ({normal / total:.1%})")
    print(f"  Suspicious (label=1) : {suspicious}  ({suspicious / total:.1%})")
    print(f"  Time span            : {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print("-" * 62)
    print("  Scenario distribution:")
    for name, count in df["scenario"].value_counts().items():
        marker = "  [normal]  " if name == "normal" else "  [attack]  "
        print(f"{marker}{name:<28} {count:>6}")
    print("=" * 62)
    print("\nNOTE: These are SYNTHETIC logs. Suspicious scenarios are")
    print("      Cisco-2022-inspired simulations of documented behaviours.\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic SIEM security log dataset (Phase 1)."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                        help="Path to config.yaml")
    parser.add_argument("--records", type=int, default=None,
                        help="Override total record count")
    parser.add_argument("--out", default=None,
                        help="Override output CSV path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.records is not None:
        cfg["data_generation"]["total_records"] = args.records

    out_path = args.out or cfg["data_generation"]["output_path"]
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"[*] Generating synthetic logs (seed={cfg['project']['random_seed']}) ...")
    generator = SecurityLogGenerator(cfg)
    df = generator.generate()
    df.to_csv(out_path, index=False)

    _print_summary(df, out_path)


if __name__ == "__main__":
    main()