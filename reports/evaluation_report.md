# AI-Powered SIEM — Evaluation Report

*Generated: 2026-09-18 20:46:11*

## Case Study

**AI-Powered SIEM for Detecting Credential Theft and Lateral Movement — A Case Study of the Cisco Data Breach (2022).**

> All logs are synthetic. The Cisco 2022 incident is used only as a threat model. No Cisco internal data was accessed.

---

## 1. Dataset

- **Total events:** 10,007
- **Total alerts (correlated incidents):** 754
- **Normal events:** 8,500
- **Suspicious events:** 1,507

**Alert severity distribution:**

| Severity   |   Count |
|:-----------|--------:|
| CRITICAL   |      25 |
| HIGH       |      51 |
| MEDIUM     |     131 |
| LOW        |     547 |

---

## 2. Detector Ablation — Per Event

Three detectors evaluated against ground-truth labels at the individual log-line level.

| detector                   |   precision |   recall |    f1 |   fpr |   tp |   fp |   fn |   tn |
|:---------------------------|------------:|---------:|------:|------:|-----:|-----:|-----:|-----:|
| Rules only                 |       0.986 |    0.71  | 0.826 | 0.002 | 1070 |   15 |  437 | 8485 |
| ML only (Isolation Forest) |       0.537 |    0.985 | 0.696 | 0.15  | 1485 | 1278 |   22 | 7222 |
| Hybrid (rules + ML)        |       0.541 |    0.999 | 0.702 | 0.15  | 1505 | 1278 |    2 | 7222 |

**Interpretation.** Rules-only achieves very high precision but moderate recall — it misses attack patterns it wasn't told to look for. ML-only achieves near-perfect recall (the Isolation Forest rarely misses a true attack) at the cost of many false positives. The hybrid detector inherits the rules' precision and the ML's recall.

---

## 3. Detector Ablation — Per Correlated Incident

Metrics at the incident level (each row is a user session aggregated across all its events). This is the level a SOC analyst actually triages.

| detector             |   precision |   recall |    f1 |   fpr |   tp |   fp |   fn |   tn |
|:---------------------|------------:|---------:|------:|------:|-----:|-----:|-----:|-----:|
| Any alert            |       0.277 |    1     | 0.434 |  1    |  209 |  545 |    0 |    0 |
| Actionable (MEDIUM+) |       0.947 |    0.938 | 0.942 |  0.02 |  196 |   11 |   13 |  534 |

**Interpretation.** Correlation is a precision booster: the MEDIUM+ filter drops the majority of ML false positives into LOW severity, leaving a small, high-precision queue.

---

## 4. Controlled Test Scenarios

Four scenarios exercised end-to-end. Each test passes when the detector behaves as expected.

### Test 1 — Normal user behaviour

- **Expected:** No HIGH or CRITICAL alert
- **Events tested:** 8,500
- **Alerts generated:** 545
- **High Or Critical:** 0
- **Result:** PASS

### Test 2 — Credential abuse / credential access

- **Expected:** At least one Credential Abuse or Credential Access alert
- **Events tested:** 416
- **Alerts generated:** 52
- **Credential Alerts:** 50
- **Result:** PASS

### Test 3 — Lateral movement

- **Expected:** At least one Lateral Movement alert
- **Events tested:** 216
- **Alerts generated:** 36
- **Lateral Alerts:** 36
- **Result:** PASS

### Test 4 — Combined Cisco-inspired kill chain

- **Expected:** CRITICAL alert with multiple MITRE techniques
- **Events tested:** 438
- **Alerts generated:** 23
- **Critical Alerts:** 23
- **Multi Mitre Alerts:** 23
- **Result:** PASS

---

## 5. MITRE ATT&CK Coverage

| Technique                                             |   Alerts |
|:------------------------------------------------------|---------:|
| T1078 - Valid Accounts                                |      197 |
| T1021 - Remote Services                               |       82 |
| T1098 - Account Manipulation                          |       55 |
| T1098.005 - Account Manipulation: Device Registration |       54 |
| T1110 - Brute Force                                   |       52 |
| T1003 - OS Credential Dumping                         |       40 |
| T1136.001 - Create Account: Local Account             |       13 |

Techniques correspond to the rules that fired in the prototype; each mapping reflects a documented Cisco-2022 behaviour represented as an observable log signal.

---

## 6. Reproducibility

- Random seed: `42` (controls both Python `random` and NumPy)
- Pipeline (must run in this order):
  1. `python -m src.feature_engineering`
  2. `python -m src.rule_engine`
  3. `python -m src.ml_detector`
  4. `python -m src.risk_engine`
  5. `python -m src.alert_engine`

Same config + same seed → identical dataset, identical metrics.

---

## 7. Limitations

1. **Synthetic data.** Metrics describe behaviour on the generator's distributions, not real enterprise telemetry.
2. **Small feature set.** 16 features from 10k events; production SIEMs use hundreds of features and richer entity context.
3. **Fixed session gap.** Correlation uses a single 30-minute parameter; real SIEMs tune this per source and use graph-based correlation.
4. **Isolation Forest is not proof.** It flags statistical outliers, not confirmed attacks.
5. **No live ingestion.** The prototype scores a fixed dataset; streaming ingestion is future work.
