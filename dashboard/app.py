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
from src.decision_layer import describe_sensor, explain, top_drifting_sensors
from src.feature_engineering import get_feature_table
from src.ml.predict import load_model, predict_with_bundle

PREDICTION_COLUMNS = ["MACHINE_ID", "RUL_PREDICTION", "RUL_LOWER", "RUL_UPPER", "FAILURE_PROBABILITY", "RISK_CLASS"]
from src.outcomes import log_action, record_failure_result

st.set_page_config(page_title="Unified Command Center", layout="wide")


@st.cache_data(ttl=60)
def load_dashboard_data():
    machines = data_access.load_machine_data()
    periods = data_access.load_operating_periods()

    if config.SNOWFLAKE_MODE == "snowflake":
        features, predictions, metrics, model_label = data_access.load_snowflake_scoring()
    else:
        bundle = load_model()
        features = get_feature_table(group_col="MACHINE_ID")
        predictions = predict_with_bundle(bundle, features, id_col="MACHINE_ID")
        metrics, model_label = bundle["metrics"], f"local model trained {bundle['trained_at']}"
    predictions = add_expected_cost_column(predictions[PREDICTION_COLUMNS])

    summary = (
        machines[machines["STATUS"] == "IN_SERVICE"]
        .merge(predictions, on="MACHINE_ID", how="inner")
        .merge(oee.latest_oee_by_machine(periods), on="MACHINE_ID", how="left")
    )
    return summary, features, oee.oee_by_fleet(periods, machines), metrics, model_label


RISK_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
RISK_COLOR = {"CRITICAL": "#d62728", "HIGH": "#ff7f0e", "MEDIUM": "#f2c744", "LOW": "#2ca02c"}

st.title("Unified Command Center")
st.caption(
    f"Backend: `{config.SNOWFLAKE_MODE}` · NASA C-MAPSS turbofan fleets — real sensor data, "
    "every KPI derived from it (see DATA_SOURCES.md)."
)

try:
    summary, features, fleet_oee, metrics, model_label = load_dashboard_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

fleets = st.multiselect("Fleets", sorted(summary["FLEET"].unique()), default=sorted(summary["FLEET"].unique()))
summary = summary[summary["FLEET"].isin(fleets)]
if summary.empty:
    st.info("Select at least one fleet.")
    st.stop()
st.caption(f"Predictions: {model_label}")
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
    top_sensor = top_drifting_sensors(feature_row, n=1)[0][0]
    fig = px.line(
        sensor_history, x="TIME_CYCLE", y=top_sensor,
        title=f"Fastest-drifting sensor: {describe_sensor(top_sensor)}",
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("**AI / Decision layer**")
    reasoning = explain(selected_machine, machine_row.to_dict(), feature_row)
    st.write(f"- **Why it will fail:** {reasoning['why']}")
    st.write(f"- **Root cause:** {reasoning['root_cause']}")
    st.write(f"- **Recommended action:** {reasoning['recommended_action']}")
    st.write(f"- **Certainty:** {reasoning['confidence_note']}")
    if reasoning["citations"]:
        with st.expander(f"Sources — NASA C-MAPSS documentation ({len(reasoning['citations'])})"):
            for citation in reasoning["citations"]:
                st.caption(citation)
    st.caption(f"Reasoning: {reasoning['source']}")

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
    st.subheader("Maintenance agent")
    st.caption(
        "A Cortex Agent: it queries the fleet (Cortex Analyst), reads NASA's C-MAPSS documentation "
        "(Cortex Search), and drafts work orders for a planner to approve below."
    )
    agent_tab, analyst_tab = st.tabs(["Ask the agent", "Cortex Analyst (SQL only)"])

    with agent_tab:
        from src.maintenance_agent import ask as ask_agent

        history = st.session_state.setdefault("agent_history", [])
        for turn in history:
            with st.chat_message(turn["role"]):
                st.markdown(turn["text"])
        prompt = st.chat_input("e.g. Draft work orders for the critical engines in fleet FD003")
        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("Agent is working…"):
                    try:
                        result = ask_agent(prompt, history=history)
                    except Exception as e:
                        result = None
                        st.warning(f"Agent unavailable: {e}")
                if result:
                    st.markdown(result["answer"])
                    for table in result["tables"]:
                        st.dataframe(pd.DataFrame(table["rows"], columns=table["columns"]), hide_index=True)
                    with st.expander(f"How the agent got this ({len(result['steps'])} tool calls)"):
                        for step in result["steps"]:
                            st.markdown(f"**{step['tool']}**")
                            if step["sql"]:
                                st.code(step["sql"].strip(), language="sql")
                            elif step["query"]:
                                st.caption(f"search: {step['query']}")
                            else:
                                st.json(step["input"])
                        if result["citations"]:
                            st.caption("Sources: " + "; ".join(result["citations"]))
                    history += [{"role": "user", "text": prompt}, {"role": "assistant", "text": result["answer"]}]
                    if any(step["tool"] == "draft_work_order" for step in result["steps"]):
                        st.cache_data.clear()

    with analyst_tab:
        question = st.text_input("Ask a question", placeholder="Which fleet has the worst OEE?")
        if question:
            from src.connection import get_session
            from src.cortex_analyst import ask as ask_analyst

            try:
                result = ask_analyst(question)
                st.write(result["answer"])
                if result["sql"]:
                    st.dataframe(get_session().sql(result["sql"]).to_pandas(), hide_index=True)
                    with st.expander("Generated SQL"):
                        st.code(result["sql"], language="sql")
            except Exception as e:
                st.warning(f"Cortex Analyst unavailable: {e}")

    st.divider()
    st.subheader("Work orders awaiting approval")
    from src import work_orders

    orders = work_orders.load_work_orders()
    pending = orders[orders["STATUS"] == "PENDING_APPROVAL"]
    if pending.empty:
        st.caption("No drafts waiting. Ask the agent to draft work orders for at-risk engines.")
    for _, order in pending.iterrows():
        with st.container(border=True):
            c1, c2, c3 = st.columns([5, 1, 1])
            c1.markdown(
                f"**{order['PRIORITY']} {order['WORK_TYPE']}** · {order['MACHINE_ID']} · "
                f"{order['RISK_CLASS_AT_DRAFT']}, RUL {order['RUL_PREDICTION_AT_DRAFT']:.0f} cycles at draft"
            )
            c1.caption(order["JUSTIFICATION"]
            )
            if c2.button("Approve", key=f"approve-{order['WORK_ORDER_ID']}", type="primary"):
                work_orders.approve(order["WORK_ORDER_ID"])
                st.cache_data.clear()
                st.rerun()
            if c3.button("Reject", key=f"reject-{order['WORK_ORDER_ID']}"):
                work_orders.reject(order["WORK_ORDER_ID"])
                st.rerun()
    reviewed = orders[orders["STATUS"] != "PENDING_APPROVAL"]
    if not reviewed.empty:
        with st.expander(f"Reviewed work orders ({len(reviewed)})"):
            st.dataframe(
                reviewed[["MACHINE_ID", "WORK_TYPE", "PRIORITY", "STATUS", "REVIEWED_BY", "REVIEWED_AT", "CREATED_BY"]],
                hide_index=True,
            )

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
