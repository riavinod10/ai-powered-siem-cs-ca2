"""
Phase 6 — Streamlit SOC Dashboard
=================================

Six-page SOC-style dashboard on top of the hybrid SIEM pipeline:

  Page 1 — SOC Overview            (KPIs + event/severity charts)
  Page 2 — Authentication Monitor  (logins, failures, unusual times)
  Page 3 — Threat Alerts           (filterable alert table + detail view)
  Page 4 — Lateral Movement        (network graph per suspicious user)
  Page 5 — Incident Timeline       (chronological kill-chain replay)
  Page 6 — Analytics / Performance (confusion matrix, precision/recall/F1)

Data source
-----------
Reads from Elasticsearch (`siem-logs` and `siem-alerts`) when reachable.
Falls back to local CSVs automatically, so the dashboard always works.

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

# ---------------------------------------------------------------------------
# Config & paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config", "config.yaml")

EVENTS_CSV = os.path.join(ROOT, "data", "processed", "events_with_risk.csv")
ALERTS_CSV = os.path.join(ROOT, "data", "alerts", "alerts.csv")

SEVERITY_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
SEVERITY_COLORS = {
    "CRITICAL": "#d62728",
    "HIGH":     "#ff7f0e",
    "MEDIUM":   "#f1c40f",
    "LOW":      "#3498db",
}
THREAT_COLORS = {
    "Credential Access":                   "#d62728",
    "Lateral Movement":                    "#ff7f0e",
    "Privilege Escalation":                "#e67e22",
    "MFA Abuse":                           "#9b59b6",
    "Credential Abuse":                    "#c0392b",
    "Persistence: New Privileged Account": "#8e44ad",
    "Unrecognized Device":                 "#16a085",
    "Off-Hours Access":                    "#2980b9",
    "ML Anomaly (Unclassified)":           "#7f8c8d",
}

st.set_page_config(
    page_title="AI-Powered SIEM — SOC Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


CFG = load_config()


# ---------------------------------------------------------------------------
# Data loading — ES first, CSV fallback
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_from_es() -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Try to pull events and alerts from Elasticsearch. Return None on failure."""
    try:
        from elasticsearch import Elasticsearch
    except Exception:
        return None

    es_cfg = CFG["elasticsearch"]
    try:
        client = Elasticsearch(es_cfg["host"], request_timeout=2)
        if not client.ping():
            return None

        def _all(index: str, size: int = 20000) -> pd.DataFrame:
            res = client.search(
                index=index,
                body={"size": size, "query": {"match_all": {}}},
            )
            return pd.DataFrame(hit["_source"] for hit in res["hits"]["hits"])

        events = _all(es_cfg["log_index"])
        alerts = _all(es_cfg["alert_index"])
        if events.empty or alerts.empty:
            return None
        return events, alerts
    except Exception:
        return None


def _load_from_csv() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.exists(EVENTS_CSV) or not os.path.exists(ALERTS_CSV):
        st.error(
            "No data found. Run the pipeline first:\n\n"
            "```\npython -m src.feature_engineering\n"
            "python -m src.rule_engine\n"
            "python -m src.ml_detector\n"
            "python -m src.risk_engine\n"
            "python -m src.alert_engine\n```"
        )
        st.stop()
    events = pd.read_csv(EVENTS_CSV, parse_dates=["timestamp"])
    alerts = pd.read_csv(
        ALERTS_CSV, parse_dates=["timestamp", "last_seen"]
    )
    return events, alerts


@st.cache_data(show_spinner="Loading events and alerts ...")
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return (events, alerts, source_label)."""
    from_es = _load_from_es()
    if from_es is not None:
        events, alerts = from_es
        source = "Elasticsearch"
        for df in (events, alerts):
            for col in ("timestamp", "last_seen"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
    else:
        events, alerts = _load_from_csv()
        source = "Local CSV (Elasticsearch offline)"

    for df in (events, alerts):
        if "severity" in df.columns:
            df["severity"] = pd.Categorical(
                df["severity"], categories=SEVERITY_ORDER, ordered=True
            )
    return events, alerts, source


# ---------------------------------------------------------------------------
# Shared sidebar
# ---------------------------------------------------------------------------
def render_sidebar(events: pd.DataFrame, alerts: pd.DataFrame, source: str):
    st.sidebar.markdown(
        "## 🛡️ AI-Powered SIEM\n"
        "*Cisco 2022 Case Study — Prototype*"
    )
    st.sidebar.caption(
        "Hybrid rule + ML detection of credential theft and lateral movement."
    )
    st.sidebar.markdown("---")

    pages = [
        "1. SOC Overview",
        "2. Authentication Monitoring",
        "3. Threat Alerts",
        "4. Lateral Movement",
        "5. Incident Timeline",
        "6. Model Performance",
    ]
    page = st.sidebar.radio("Navigate", pages, index=0)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Data source:** {source}")
    st.sidebar.markdown(
        f"**Events:** {len(events):,}  \n"
        f"**Alerts:** {len(alerts):,}"
    )

    if alerts is not None and not alerts.empty:
        counts = alerts["severity"].value_counts().reindex(
            SEVERITY_ORDER
        ).fillna(0).astype(int)
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Alert queue**")
        for sev in SEVERITY_ORDER:
            st.sidebar.markdown(
                f"<span style='color:{SEVERITY_COLORS[sev]};font-weight:bold'>"
                f"● {sev}</span>&nbsp;&nbsp;{counts[sev]}",
                unsafe_allow_html=True,
            )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "⚠️ Synthetic data. Suspicious events simulate behaviours "
        "documented publicly in the Cisco 2022 incident. "
        "No Cisco data was used."
    )
    return page


# ---------------------------------------------------------------------------
# Page 1 — SOC Overview
# ---------------------------------------------------------------------------
def page_overview(events: pd.DataFrame, alerts: pd.DataFrame):
    st.title("SOC Overview")
    st.caption(
        "Executive view of security events, alerts, and severity distribution."
    )

    total_events = len(events)
    suspicious_events = int((events["risk_score"] > 0).sum()) \
        if "risk_score" in events.columns else 0
    total_alerts = len(alerts)

    sev_counts = alerts["severity"].value_counts().reindex(
        SEVERITY_ORDER
    ).fillna(0).astype(int)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total events", f"{total_events:,}")
    c2.metric("Suspicious events", f"{suspicious_events:,}")
    c3.metric("Total alerts", f"{total_alerts:,}")
    c4.metric("Critical alerts", int(sev_counts["CRITICAL"]))
    c5.metric(
        "Actionable (MEDIUM+)",
        int(sev_counts["MEDIUM"] + sev_counts["HIGH"] + sev_counts["CRITICAL"]),
    )

    st.markdown("---")

    left, right = st.columns([1, 2])

    with left:
        st.subheader("Severity distribution")
        sev_df = pd.DataFrame({
            "severity": SEVERITY_ORDER,
            "count": [int(sev_counts[s]) for s in SEVERITY_ORDER],
        })
        fig = px.bar(
            sev_df, x="severity", y="count", color="severity",
            color_discrete_map=SEVERITY_COLORS,
            text="count",
        )
        fig.update_layout(showlegend=False, height=340,
                          margin=dict(l=0, r=0, t=10, b=0))
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Threat type distribution")
        tt = alerts["threat_type"].value_counts().reset_index()
        tt.columns = ["threat_type", "count"]
        fig = px.bar(
            tt.head(10), x="count", y="threat_type", orientation="h",
            color="threat_type", color_discrete_map=THREAT_COLORS,
            text="count",
        )
        fig.update_layout(showlegend=False, height=340,
                          margin=dict(l=0, r=0, t=10, b=0),
                          yaxis_title="", xaxis_title="")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Activity over time")
    ev = events.copy()
    ev["hour"] = ev["timestamp"].dt.floor("h")
    ev["is_susp"] = (ev.get("risk_score", 0) > 0).astype(int)

    timeline = ev.groupby("hour").agg(
        total=("event_id", "count"),
        suspicious=("is_susp", "sum"),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timeline["hour"], y=timeline["total"],
        name="All events", mode="lines",
        line=dict(color="#3498db", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=timeline["hour"], y=timeline["suspicious"],
        name="Suspicious events", mode="lines",
        line=dict(color="#d62728", width=2),
    ))
    fig.update_layout(
        height=350, margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        xaxis_title="", yaxis_title="Events per hour",
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 2 — Authentication Monitoring
# ---------------------------------------------------------------------------
def page_auth(events: pd.DataFrame, alerts: pd.DataFrame):
    st.title("Authentication Monitoring")
    st.caption(
        "Login activity, failed authentications, unusual times, "
        "new devices and suspicious source IPs."
    )

    ev = events.copy()
    logins = ev[ev["event_type"].astype(str) == "login"]
    failed = logins[logins["login_status"].astype(str) == "failed"]
    success = logins[logins["login_status"].astype(str) == "success"]
    new_dev = logins[logins["new_device"] == 1]

    start = int(CFG["rules"]["unusual_hour_start"])
    end = int(CFG["rules"]["unusual_hour_end"])
    unusual = success[
        (success["login_hour"] >= start) & (success["login_hour"] <= end)
    ]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total logins", f"{len(logins):,}")
    c2.metric("Successful", f"{len(success):,}")
    c3.metric("Failed", f"{len(failed):,}")
    c4.metric("Off-hours logins", f"{len(unusual):,}")
    c5.metric("New devices", f"{len(new_dev):,}")

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.subheader("Logins by hour of day")
        hourly = logins.groupby("login_hour").size().reset_index(name="count")
        fig = px.bar(hourly, x="login_hour", y="count")
        fig.update_traces(marker_color="#3498db")
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title="Hour", yaxis_title="Logins")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Failed logins per user (top 10)")
        top_failed = failed["username"].value_counts().head(10).reset_index()
        top_failed.columns = ["username", "failed_count"]
        fig = px.bar(top_failed, x="failed_count", y="username",
                     orientation="h")
        fig.update_traces(marker_color="#d62728")
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                          yaxis_title="", xaxis_title="Failed logins")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Suspicious source IPs")
    sus_ips = (
        ev[ev.get("risk_score", 0) > 0]
        .groupby("source_ip")
        .agg(events=("event_id", "count"),
             max_risk=("risk_score", "max"))
        .reset_index()
        .sort_values("max_risk", ascending=False)
        .head(15)
    )
    if sus_ips.empty:
        st.info("No suspicious source IPs.")
    else:
        st.dataframe(sus_ips, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page 3 — Threat Alerts
# ---------------------------------------------------------------------------
def page_alerts(events: pd.DataFrame, alerts: pd.DataFrame):
    st.title("Threat Alerts")
    st.caption(
        "All alerts, ranked by risk score. Select an alert to see full detail."
    )

    fc1, fc2, fc3 = st.columns([1, 1, 2])
    with fc1:
        sev_filter = st.multiselect(
            "Severity", SEVERITY_ORDER, default=SEVERITY_ORDER
        )
    with fc2:
        threat_filter = st.multiselect(
            "Threat type",
            sorted(alerts["threat_type"].unique().tolist()),
            default=sorted(alerts["threat_type"].unique().tolist()),
        )
    with fc3:
        search = st.text_input("Search username / IP / host", "")

    view = alerts[
        alerts["severity"].isin(sev_filter)
        & alerts["threat_type"].isin(threat_filter)
    ].copy()
    if search:
        mask = (
            view["username"].astype(str).str.contains(search, case=False)
            | view["source_ip"].astype(str).str.contains(search, case=False)
            | view["affected_hosts"].astype(str).str.contains(search, case=False)
        )
        view = view[mask]

    st.caption(f"Showing **{len(view)}** of {len(alerts)} alerts")

    cols = ["alert_id", "timestamp", "username", "threat_type",
            "risk_score", "severity", "status"]
    st.dataframe(
        view[cols].sort_values("risk_score", ascending=False),
        use_container_width=True, hide_index=True, height=380,
    )

    st.markdown("---")
    st.subheader("Alert detail")
    if view.empty:
        st.info("No alerts match the current filters.")
        return

    chosen = st.selectbox(
        "Select alert", view["alert_id"].tolist(),
        format_func=lambda aid: (
            f"{aid} — {view[view['alert_id'] == aid]['username'].iloc[0]} "
            f"({view[view['alert_id'] == aid]['risk_score'].iloc[0]}/100)"
        ),
    )
    row = view[view["alert_id"] == chosen].iloc[0]

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"### {row['alert_id']}")
        st.markdown(f"**Threat** : {row['threat_type']}")
        st.markdown(f"**User** : `{row['username']}`")
        st.markdown(f"**Source IP** : `{row['source_ip']}`")
        st.markdown(f"**Affected hosts** : `{row['affected_hosts']}`")
        st.markdown(f"**First seen** : {row['timestamp']}")
        st.markdown(f"**Last seen** : {row['last_seen']}")
        st.markdown(f"**Duration** : {row['duration_minutes']} min")
        st.markdown(f"**Events in incident** : {row['event_count']}")
    with c2:
        st.markdown("### Risk")
        st.markdown(
            f"<h1 style='color:{SEVERITY_COLORS[row['severity']]};"
            f"margin-top:0'>{row['risk_score']}/100 &nbsp; {row['severity']}</h1>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**Rule score** : {row['rule_score']}")
        st.markdown(f"**ML bonus** : {row['ml_bonus']} (tier: {row['ml_tier']})")
        st.markdown(f"**ML anomaly score** : {row['ml_anomaly_score']}")
        st.markdown("**Recommended response:**")
        st.info(row["recommended_response"])

    st.markdown("### Triggered indicators")
    st.write(row["triggered_indicators"])

    st.markdown("### MITRE ATT&CK techniques")
    if row["mitre_techniques"]:
        for tech in str(row["mitre_techniques"]).split(";"):
            tech = tech.strip()
            if tech:
                st.markdown(f"- `{tech}`")
    else:
        st.caption("No MITRE technique mapping (pure ML anomaly alert).")


# ---------------------------------------------------------------------------
# Page 4 — Lateral Movement
# ---------------------------------------------------------------------------
def page_lateral(events: pd.DataFrame, alerts: pd.DataFrame):
    st.title("Lateral Movement")
    st.caption(
        "Sessions where a single user accessed multiple hosts — "
        "the classic lateral-movement pattern."
    )

    ev = events.copy()

    lat_alerts = alerts[alerts["threat_type"] == "Lateral Movement"]
    candidates = set(lat_alerts["username"].unique().tolist())

    if not candidates:
        st.info("No lateral-movement sessions detected.")
        return

    st.markdown(f"**{len(candidates)}** users flagged with lateral movement.")

    user = st.selectbox("Select user", sorted(candidates))

    user_alerts = lat_alerts[lat_alerts["username"] == user].sort_values(
        ["risk_score", "timestamp"], ascending=[False, False]
    )
    if user_alerts.empty:
        st.info("No lateral-movement alert for this user.")
        return

    alert_row = user_alerts.iloc[0]
    inc_start = alert_row["timestamp"]
    inc_end = alert_row["last_seen"]

    st.markdown(
        f"**Incident window:** {inc_start} → {inc_end} "
        f"({alert_row['duration_minutes']} min)"
    )

    inc = ev[
        (ev["username"] == user)
        & (ev["timestamp"] >= inc_start)
        & (ev["timestamp"] <= inc_end)
    ].sort_values("timestamp")

    if inc.empty:
        st.info("No events in the incident window.")
        return

    host_events = inc[inc["event_type"].astype(str) == "server_access"] \
        .sort_values("timestamp")
    host_sequence = host_events["destination_host"].tolist()

    G = nx.DiGraph()
    user_node = f"👤 {user}"
    G.add_node(user_node)

    prev = user_node
    ordered_nodes = [user_node]
    for host in host_sequence:
        G.add_edge(prev, host)
        ordered_nodes.append(host)
        prev = host

    seen = set()
    ordered_nodes = [
        n for n in ordered_nodes
        if not (n in seen or seen.add(n))
    ]

    n = len(ordered_nodes)
    pos = {}
    for i, node in enumerate(ordered_nodes):
        x = i / max(n - 1, 1)
        y = 0.5 + 0.08 * (1 if i % 2 == 0 else -1)
        pos[node] = (x, y)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=2, color="#666"),
        hoverinfo="none",
    )

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for n in G.nodes():
        x, y = pos[n]
        node_x.append(x); node_y.append(y)
        node_text.append(str(n))
        if str(n).startswith("👤"):
            node_color.append("#f39c12")
            node_size.append(38)
        elif "dc-01" in str(n):
            node_color.append("#d62728")
            node_size.append(34)
        elif any(s in str(n) for s in ("backup", "admin-jump")):
            node_color.append("#e67e22")
            node_size.append(30)
        else:
            node_color.append("#3498db")
            node_size.append(26)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=node_text, textposition="top center",
        marker=dict(size=node_size, color=node_color,
                    line=dict(width=1.5, color="white")),
        hoverinfo="text",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        height=360, showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-0.1, 1.1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[0.2, 0.8]),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Hosts accessed in this incident")
    host_summary = (
        host_events
        .groupby("destination_host", sort=False)
        .agg(first_seen=("timestamp", "min"),
             accesses=("event_id", "count"))
        .reset_index()
    )
    st.dataframe(host_summary, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page 5 — Incident Timeline
# ---------------------------------------------------------------------------
def page_timeline(events: pd.DataFrame, alerts: pd.DataFrame):
    st.title("Incident Timeline")
    st.caption(
        "Chronological replay of a user's suspicious session — "
        "the Cisco-2022-inspired kill chain."
    )

    suspicious_users = sorted(
        alerts[alerts["severity"].isin(["CRITICAL", "HIGH"])]["username"]
        .unique().tolist()
    )
    if not suspicious_users:
        suspicious_users = sorted(alerts["username"].unique().tolist())

    user = st.selectbox("Select user", suspicious_users)

    user_alerts = alerts[alerts["username"] == user].sort_values(
        ["risk_score", "timestamp"], ascending=[False, False]
    )
    if user_alerts.empty:
        st.info("No alerts for this user.")
        return

    top_alert = user_alerts.iloc[0]

    ev = events.copy()
    inc = ev[
        (ev["username"] == user)
        & (ev["timestamp"] >= top_alert["timestamp"])
        & (ev["timestamp"] <= top_alert["last_seen"])
    ].sort_values("timestamp")

    if inc.empty:
        st.info("No events in the incident window.")
        return

    st.markdown(
        f"**Alert:** `{top_alert['alert_id']}` — "
        f"**{top_alert['threat_type']}** — "
        f"**{top_alert['risk_score']}/100 {top_alert['severity']}**"
    )

    st.markdown("### Event-by-event replay")
    for _, row in inc.iterrows():
        t = row["timestamp"]
        etype = row.get("event_type", "")
        icon = {
            "login": "🔑",
            "logout": "🚪",
            "mfa_event": "📱",
            "privilege_change": "🔺",
            "process_start": "⚙️",
            "server_access": "🖧",
            "file_access": "📄",
            "email_access": "✉️",
            "account_creation": "🆕",
        }.get(etype, "•")

        detail_parts = [f"**{etype}**"]
        if pd.notna(row.get("destination_host")):
            detail_parts.append(f"on `{row['destination_host']}`")
        if row.get("login_status") not in (None, "NA", float("nan")):
            detail_parts.append(f"— status: {row['login_status']}")
        if row.get("mfa_event") not in (None, "none", float("nan")):
            detail_parts.append(f"— mfa: {row['mfa_event']}")
        if row.get("privilege_escalation") == 1:
            detail_parts.append("— ⚠️ privilege escalation")
        if row.get("credential_access") == 1:
            detail_parts.append("— ⚠️ credential-access activity")
        if int(row.get("rule_score", 0)) > 0:
            detail_parts.append(f"— rules: {row.get('triggered_rules', '')}")

        st.markdown(
            f"`{t.strftime('%H:%M:%S')}` &nbsp; {icon} &nbsp; "
            + " ".join(str(p) for p in detail_parts)
        )

    st.markdown("---")
    st.markdown("### Per-event rule + risk score progression")
    ts_df = inc[["timestamp", "rule_score", "risk_score", "ml_tier"]].copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_df["timestamp"], y=ts_df["rule_score"],
        name="Per-event rule score", mode="lines+markers",
        line=dict(color="#f39c12"),
    ))
    fig.add_trace(go.Scatter(
        x=ts_df["timestamp"], y=ts_df["risk_score"],
        name="Per-event risk score (rule + ML bonus)",
        mode="lines+markers",
        line=dict(color="#d62728"),
    ))
    fig.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="", yaxis_title="Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 6 — Model Performance
# ---------------------------------------------------------------------------
def page_performance(events: pd.DataFrame, alerts: pd.DataFrame):
    st.title("Analytics / Model Performance")
    st.caption(
        "Hybrid detection compared against ground-truth labels. "
        "Labels are used for evaluation only — never for training."
    )

    ev = events.copy()
    if "label" not in ev.columns:
        st.warning("No ground-truth labels available in the event data.")
        return

    def _metrics(pred_col: str) -> dict:
        y = ev["label"].astype(int)
        p = ev[pred_col].astype(int)
        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        tn = int(((p == 0) & (y == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) \
            if (precision + recall) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                    precision=precision, recall=recall,
                    f1=f1, fpr=fpr)

    ev["_rule_pred"] = (ev.get("rule_score", 0) > 0).astype(int)
    ev["_ml_pred"] = (ev.get("ml_tier", "none").astype(str) != "none").astype(int)
    ev["_hybrid_pred"] = (ev.get("risk_score", 0) > 0).astype(int)

    rule_m = _metrics("_rule_pred")
    ml_m = _metrics("_ml_pred")
    hy_m = _metrics("_hybrid_pred")

    st.subheader("Per-event detection performance")
    perf = pd.DataFrame([
        {"Detector": "Rules only",
         "Precision": rule_m["precision"], "Recall": rule_m["recall"],
         "F1": rule_m["f1"], "FPR": rule_m["fpr"]},
        {"Detector": "ML only (Isolation Forest)",
         "Precision": ml_m["precision"], "Recall": ml_m["recall"],
         "F1": ml_m["f1"], "FPR": ml_m["fpr"]},
        {"Detector": "Hybrid (rules + ML)",
         "Precision": hy_m["precision"], "Recall": hy_m["recall"],
         "F1": hy_m["f1"], "FPR": hy_m["fpr"]},
    ]).set_index("Detector")

    st.dataframe(
        perf.style.format("{:.3f}").background_gradient(cmap="RdYlGn_r"),
        use_container_width=True,
    )

    st.markdown("---")
    st.subheader("Confusion matrices (per event)")

    def _cm_fig(m: dict, title: str):
        fig = px.imshow(
            [[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
            text_auto=True, aspect="auto",
            labels=dict(x="Predicted", y="Actual"),
            x=["Normal", "Suspicious"], y=["Normal", "Suspicious"],
            color_continuous_scale="Blues",
        )
        fig.update_layout(title=title, height=320,
                          margin=dict(l=0, r=0, t=40, b=0),
                          coloraxis_showscale=False)
        return fig

    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(_cm_fig(rule_m, "Rules only"), use_container_width=True)
    c2.plotly_chart(_cm_fig(ml_m, "ML only"), use_container_width=True)
    c3.plotly_chart(_cm_fig(hy_m, "Hybrid"), use_container_width=True)

    st.markdown("---")
    st.subheader("Incident-level performance (the actionable metric)")

    actionable = alerts[
        alerts["severity"].isin(["MEDIUM", "HIGH", "CRITICAL"])
    ]
    if len(actionable):
        tp = int((actionable["label"] == 1).sum())
        fp = int((actionable["label"] == 0).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        c1, c2, c3 = st.columns(3)
        c1.metric("Actionable alerts", len(actionable))
        c2.metric("True positives", tp)
        c3.metric("Precision @ MEDIUM+", f"{precision:.3f}")

    st.markdown("---")
    st.subheader("Anomaly score distribution (normal vs suspicious)")

    if "anomaly_score" in ev.columns:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=ev[ev["label"] == 0]["anomaly_score"],
            name="Normal", opacity=0.7, nbinsx=60,
            marker_color="#3498db",
        ))
        fig.add_trace(go.Histogram(
            x=ev[ev["label"] == 1]["anomaly_score"],
            name="Suspicious", opacity=0.7, nbinsx=60,
            marker_color="#d62728",
        ))
        fig.update_layout(
            barmode="overlay", height=340,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Isolation Forest anomaly score",
            yaxis_title="Event count",
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def main():
    events, alerts, source = load_data()
    page = render_sidebar(events, alerts, source)

    try:
        if page.startswith("1"):
            page_overview(events, alerts)
        elif page.startswith("2"):
            page_auth(events, alerts)
        elif page.startswith("3"):
            page_alerts(events, alerts)
        elif page.startswith("4"):
            page_lateral(events, alerts)
        elif page.startswith("5"):
            page_timeline(events, alerts)
        elif page.startswith("6"):
            page_performance(events, alerts)
    except Exception as exc:
        st.error(f"Page error: {exc}")
        st.exception(exc)


if __name__ == "__main__":
    main()