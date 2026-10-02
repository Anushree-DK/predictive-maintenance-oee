"""Unified Command Center: engine health / RUL / risk / OEE in one view, with
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
from src.business_impact import add_expected_cost_column, expected_cost_avoided, fleet_risk_exposure
from src.decision_layer import explain
from src.feature_engineering import get_feature_table
from src.ml.predict import load_model, predict_with_bundle
from src.outcomes import log_action, record_failure_result

st.set_page_config(page_title="Unified Command Center", layout="wide")


@st.cache_data(ttl=60)
def load_dashboard_data():
    machines = data_access.load_machine_data()
    periods = data_access.load_operating_periods()

    bundle = load_model()
    features = get_feature_table(group_col="MACHINE_ID")
    predictions = add_expected_cost_column(predict_with_bundle(bundle, features, id_col="MACHINE_ID"))

    summary = (
        machines[machines["STATUS"] == "IN_SERVICE"]
        .merge(predictions, on="MACHINE_ID", how="inner")
        .merge(oee.latest_oee_by_machine(periods), on="MACHINE_ID", how="left")
    )
    return summary, features, oee.oee_by_fleet(periods, machines), bundle["metrics"]


RISK_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
RISK_COLOR = {"CRITICAL": "#d62728", "HIGH": "#ff7f0e", "MEDIUM": "#f2c744", "LOW": "#2ca02c"}

st.title("Unified Command Center")
st.caption(
    f"Backend: `{config.SNOWFLAKE_MODE}` · NASA C-MAPSS turbofan fleets — real sensor data, "
    "every KPI derived from it (see DATA_SOURCES.md)."
)

try:
    summary, features, fleet_oee, metrics = load_dashboard_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

fleets = st.multiselect("Fleets", sorted(summary["FLEET"].unique()), default=sorted(summary["FLEET"].unique()))
summary = summary[summary["FLEET"].isin(fleets)]
if summary.empty:
    st.info("Select at least one fleet.")
    st.stop()
summary = summary.sort_values(
    by=["RISK_CLASS", "RUL_LOWER"], key=lambda s: s.map(RISK_ORDER) if s.name == "RISK_CLASS" else s
)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Engines in service", len(summary))
col2.metric("Critical / High risk", int((summary["RISK_CLASS"].isin(["CRITICAL", "HIGH"])).sum()))
col3.metric("Avg OEE (recent)", f"{summary['OEE'].mean():.0%}")
col4.metric("RUL error (RMSE, NASA test set)", f"{metrics['rul']['overall']['rmse']:.1f} cycles")
col5.metric("Est. fleet risk exposure", f"${fleet_risk_exposure(summary):,.0f}")
st.caption(
    "Risk exposure = expected downtime cost if nothing is done (calibrated P(failure) × cost gap "
    "between an emergency and a scheduled repair). Cost rates are assumptions in config.py."
)

with st.expander("How accurate is the model? (evaluated on NASA's held-out test engines)"):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RUL RMSE", f"{metrics['rul']['overall']['rmse']:.1f} cycles")
    m2.metric(f"{config.PREDICTION_INTERVAL:.0%} interval coverage", f"{metrics['rul_interval']['empirical_coverage']:.0%}")
    m3.metric("Failure-in-30 AUC", f"{metrics['failure_probability']['roc_auc']:.3f}")
    m4.metric("Recall at P ≥ 0.5", f"{metrics['failure_probability']['recall_at_0.5']:.0%}")
    st.dataframe(
        pd.DataFrame(metrics["rul"]["per_fleet"]).T.rename_axis("Fleet").reset_index(),
        hide_index=True, width="stretch",
    )
    st.caption(
        f"Splitting by engine, not by row: a random row split reports RMSE "
        f"{metrics['baselines']['leaky_random_row_split_fd001_rmse']:.1f} on FD001 because it tests on engines "
        f"the model already saw. Predicting the mean scores {metrics['baselines']['constant_mean_rul_rmse']:.1f}."
    )

st.divider()

left, right = st.columns([2, 3])

with left:
    st.subheader("Fleet")
    table = summary.assign(
        RUL_INTERVAL=summary["RUL_LOWER"].map("{:.0f}".format) + "–" + summary["RUL_UPPER"].map("{:.0f}".format)
    )
    display_cols = ["MACHINE_ID", "FLEET", "RISK_CLASS", "RUL_PREDICTION", "RUL_INTERVAL", "FAILURE_PROBABILITY", "OEE"]
    st.dataframe(
        table[display_cols].style.apply(
            lambda row: [f"background-color: {RISK_COLOR[row['RISK_CLASS']]}33"] * len(row), axis=1
        ).format({"RUL_PREDICTION": "{:.0f}", "FAILURE_PROBABILITY": "{:.0%}", "OEE": "{:.0%}"}),
        hide_index=True,
        width="stretch",
        height=420,
    )
    selected_machine = st.selectbox("Inspect engine", summary["MACHINE_ID"])

with right:
    machine_row = summary[summary["MACHINE_ID"] == selected_machine].iloc[0]
    feature_row = features[features["MACHINE_ID"] == selected_machine].iloc[0]

    st.subheader(f"{selected_machine} · fleet {machine_row['FLEET']}")
    st.caption(
        f"{machine_row['OPERATING_CONDITIONS']} operating condition(s) · fault modes: {machine_row['FAULT_MODES']} · "
        f"{machine_row['CYCLES_OBSERVED']} cycles flown"
    )
    m1, m2, m3, m4, m5 = st.columns(5)
    rul_label = f"{machine_row['RUL_PREDICTION']:.0f}" + ("+" if machine_row["RUL_PREDICTION"] >= config.RUL_CAP else "")
    m1.metric("RUL (cycles)", rul_label)
    m2.metric(f"{config.PREDICTION_INTERVAL:.0%} interval", f"{machine_row['RUL_LOWER']:.0f}–{machine_row['RUL_UPPER']:.0f}")
    m3.metric("Risk", machine_row["RISK_CLASS"])
    m4.metric(f"P(fail ≤ {config.FAILURE_HORIZON_CYCLES} cy)", f"{machine_row['FAILURE_PROBABILITY']:.0%}")
    m5.metric("Value of acting now", f"${machine_row['EXPECTED_COST_AVOIDED_USD']:,.0f}")

    sensor_history = data_access.load_sensor_history(selected_machine)
    top_sensor = feature_row.filter(like="_SLOPE").abs().idxmax().replace("_SLOPE", "")
    fig = px.line(
        sensor_history, x="TIME_CYCLE", y=top_sensor,
        title=f"Most-degrading sensor: {top_sensor} (raw reading)",
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("**AI / Decision layer**")
    reasoning = explain(selected_machine, machine_row.to_dict(), feature_row)
    st.write(f"- **Why it will fail:** {reasoning['why']}")
    if reasoning["root_cause"]:
        st.write(f"- **Root cause:** {reasoning['root_cause']}")
        st.write(f"- **Recommended action:** {reasoning['recommended_action']}")
        st.write(f"- **Certainty:** {reasoning['confidence_note']}")

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
            cost_avoided = expected_cost_avoided(machine_row["FAILURE_PROBABILITY"])
            st.success(
                f"{action_type} logged for {selected_machine} (action `{action_id[:8]}`) — "
                f"est. ${cost_avoided:,.0f} in avoided downtime cost."
            )
            st.cache_data.clear()

st.divider()
st.subheader("OEE by fleet (lifetime, all engines incl. failed)")
st.dataframe(
    fleet_oee.style.format({c: "{:.1%}" for c in oee.OEE_COLUMNS}), hide_index=True, width="stretch"
)
st.caption(
    "Availability = flight hours vs. unplanned repair downtime · Performance = health index learned from "
    "run-to-failure trajectories · Quality = share of cycles with every sensor in spec."
)

if config.SNOWFLAKE_MODE == "snowflake":
    st.divider()
    st.subheader("Ask the fleet a question (Cortex Analyst)")
    st.caption(
        "Natural language over MACHINE_DATA / MAINTENANCE_HISTORY / OPERATING_PERIODS / "
        "ACTION_OUTCOMES, via the Semantic View in sql/003_semantic_model.yaml."
    )
    question = st.text_input("Ask a question", placeholder="Which fleet has the worst OEE?")
    if question:
        from src.cortex_analyst import ask as ask_analyst

        try:
            result = ask_analyst(question)
            st.write(result["answer"])
            if result["sql"]:
                with st.expander("Generated SQL"):
                    st.code(result["sql"], language="sql")
        except Exception as e:
            st.warning(f"Cortex Analyst unavailable: {e}")

st.divider()
st.subheader("Outcome log (feedback loop)")
outcomes = data_access.load_action_outcomes()
if outcomes.empty:
    st.caption("No actions logged yet — trigger one above to see it here.")
else:
    st.dataframe(outcomes.sort_values("RECOMMENDED_AT", ascending=False), hide_index=True, width="stretch")

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
