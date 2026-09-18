# AI-Powered SIEM for Detecting Credential Theft and Lateral Movement
### A Case Study of the Cisco Data Breach (2022)

A prototype mini-SIEM that detects **credential abuse** and **lateral movement**
in security logs using a **hybrid rule-based + machine-learning** detection engine.

> **Academic honesty note:** This project does **not** use Cisco's internal data.
> The Cisco 2022 incident is used as a **threat model**. All logs are **synthetic**
> and the suspicious scenarios are *Cisco-inspired simulations* of publicly
> documented attack behaviours. No malware, exploits, or offensive techniques
> are implemented anywhere in this codebase.

---

## Current Status

| Phase | Description | Status |
|---|---|---|
| 1 | Synthetic security log dataset | ✅ Complete |
| 2 | Rule-based detection engine | ⬜ Pending |
| 3 | ML anomaly detection (Isolation Forest) | ⬜ Pending |
| 4 | Hybrid risk scoring + alert generation | ⬜ Pending |
| 5 | Elasticsearch integration | ⬜ Pending |
| 6 | Streamlit SOC dashboard | ⬜ Pending |
| 7 | Experiments and evaluation | ⬜ Pending |

---

## Requirements

- **Python 3.10 or newer**
- pip

Check your version:

```bash
python --version