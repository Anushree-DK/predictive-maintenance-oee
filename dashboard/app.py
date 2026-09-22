"""Unified Command Center: machine health / RUL / risk / OEE in one view, with
agentic actions that fire and visibly update the dashboard + outcome log.

Run with: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from src import data_access, oee
from src.decision_layer import explain
from src.feature_engineering import get_feature_table
from src.ml.predict import predict_for_features
from src.outcomes import log_action, record_failure_result

st.set_page_config(page_title="Unified Command Center", layout="wide")


@st.cache_data(ttl=60)
def load_dashboard_data():
    machines = data_access.load_machine_data()
    sensors = data_access.load_raw_sensor_data()
    production = data_access.load_production_data()

    features = get_feature_table(group_col="MACHINE_ID")
    predictions = predict_for_features(features, id_col="MACHINE_ID")
    oee_latest = oee.latest_oee_by_machine(production)

    summary = (
        machines.merge(predictions, on="MACHINE_ID", how="left")
        .merge(oee_latest, on="MACHINE_ID", how="left")
    )
    return summary, features, sensors


RISK_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
RISK_COLOR = {"CRITICAL": "#d62728", "HIGH": "#ff7f0e", "MEDIUM": "#f2c744", "LOW": "#2ca02c"}

st.title("Unified Command Center")
st.caption(f"Mode: `{config.SNOWFLAKE_MODE}` — machine health, RUL, risk, and OEE in one place.")

try:
    summary, features, sensors = load_dashboard_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

summary = summary.sort_values(by="RISK_CLASS", key=lambda s: s.map(RISK_ORDER))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Machines tracked", len(summary))
col2.metric("Critical / High risk", int((summary["RISK_CLASS"].isin(["CRITICAL", "HIGH"])).sum()))
col3.metric("Avg OEE", f"{summary['OEE'].mean():.0%}")
col4.metric("Avg model confidence", f"{summary['CONFIDENCE_SCORE'].mean():.0%}")

st.divider()

left, right = st.columns([2, 3])

with left:
    st.subheader("Fleet")
    display_cols = ["MACHINE_ID", "MACHINE_NAME", "RISK_CLASS", "RUL_PREDICTION", "FAILURE_PROBABILITY", "OEE", "CONFIDENCE_SCORE"]
    st.dataframe(
        summary[display_cols].style.apply(
            lambda row: [f"background-color: {RISK_COLOR[row['RISK_CLASS']]}33"] * len(row), axis=1
        ),
        hide_index=True,
        use_container_width=True,
    )
    selected_machine = st.selectbox("Inspect machine", summary["MACHINE_ID"])

with right:
    machine_row = summary[summary["MACHINE_ID"] == selected_machine].iloc[0]
    feature_row = features[features["MACHINE_ID"] == selected_machine].iloc[0]

    st.subheader(f"{machine_row['MACHINE_NAME']} ({selected_machine})")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RUL (cycles)", f"{machine_row['RUL_PREDICTION']:.0f}")
    m2.metric("Risk", machine_row["RISK_CLASS"])
    m3.metric("Failure prob. (30cy)", f"{machine_row['FAILURE_PROBABILITY']:.0%}")
    m4.metric("Confidence", f"{machine_row['CONFIDENCE_SCORE']:.0%}")

    sensor_history = sensors[sensors["MACHINE_ID"] == selected_machine]
    top_sensor = feature_row.filter(like="_SLOPE").abs().idxmax().replace("_SLOPE", "")
    fig = px.line(
        sensor_history, x="TIME_CYCLE", y=top_sensor,
        title=f"Most-degrading sensor: {top_sensor}",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**AI / Decision layer**")
    reasoning = explain(selected_machine, machine_row.to_dict(), feature_row)
    st.write(f"- **Why it will fail:** {reasoning['why']}")
    if reasoning["root_cause"]:
        st.write(f"- **Root cause:** {reasoning['root_cause']}")
        st.write(f"- **Recommended action:** {reasoning['recommended_action']}")
        st.write(f"- **Confidence:** {reasoning['confidence_note']}")

    st.markdown("**Agentic actions**")
    a1, a2, a3 = st.columns(3)
    action_map = {
        "Create Work Order": a1,
        "Schedule Repair": a2,
        "Recommend Parts": a3,
    }
    for action_type, button_col in action_map.items():
        if button_col.button(action_type, key=f"{action_type}-{selected_machine}"):
            action_id = log_action(selected_machine, action_type, machine_row.to_dict())
            st.success(f"{action_type} logged for {selected_machine} (action `{action_id[:8]}`).")
            st.cache_data.clear()

st.divider()
st.subheader("Outcome log (feedback loop)")
outcomes = data_access.load_action_outcomes()
if outcomes.empty:
    st.caption("No actions logged yet — trigger one above to see it here.")
else:
    st.dataframe(outcomes.sort_values("RECOMMENDED_AT", ascending=False), hide_index=True, use_container_width=True)

    open_outcomes = outcomes[outcomes["FAILURE_OCCURRED_FLAG"].isna()]
    if not open_outcomes.empty:
        st.caption("Close the loop on a pending action:")
        pending_id = st.selectbox("Action", open_outcomes["ACTION_ID"])
        oc1, oc2 = st.columns(2)
        if oc1.button("Mark: failure occurred"):
            record_failure_result(pending_id, True, pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
            st.cache_data.clear()
            st.rerun()
        if oc2.button("Mark: no failure"):
            record_failure_result(pending_id, False)
            st.cache_data.clear()
            st.rerun()
