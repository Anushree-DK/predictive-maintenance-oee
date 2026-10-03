"""Fleet command center. Run with: streamlit run dashboard/app.py"""

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

st.set_page_config(page_title="PrognoSys · Fleet Command Center", layout="wide")


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
        trained = pd.Timestamp(bundle["trained_at"]).strftime("%d %b %Y")
        metrics, model_label = bundle["metrics"], f"RUL model trained {trained} on NASA's run-to-failure engines"
    predictions = add_expected_cost_column(predictions[PREDICTION_COLUMNS])

    summary = (
        machines[machines["STATUS"] == "IN_SERVICE"]
        .merge(predictions, on="MACHINE_ID", how="inner")
        .merge(oee.latest_oee_by_machine(periods), on="MACHINE_ID", how="left")
    )
    return summary, features, oee.oee_by_fleet(periods, machines), metrics, model_label


@st.cache_data(ttl=60)
def load_rul_drivers(machine_id: str) -> pd.DataFrame:
    """Per-sensor SHAP contributions to this engine's predicted RUL."""
    if config.SNOWFLAKE_MODE == "snowflake":
        from src.connection import get_session

        return get_session().sql(
            "SELECT * FROM ENGINE_RUL_DRIVERS WHERE MACHINE_ID = ?", params=[machine_id]
        ).to_pandas()
    from src.ml.explain import sensor_contributions, to_long

    row = features[features["MACHINE_ID"] == machine_id]
    return to_long(row["MACHINE_ID"], sensor_contributions(load_model(), row))


@st.cache_data
def load_backtest():
    import json

    return json.loads((config.REPORTS_DIR / "backtest.json").read_text()), pd.read_csv(config.REPORTS_DIR / "backtest_curve.csv")


# chart colors: blue/orange for series, blue/red for SHAP direction
SERIES_1, SERIES_2 = "#2a78d6", "#eb6834"
RAISES_RUL, LOWERS_RUL = "#2a78d6", "#e34948"

RISK_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
RISK_COLOR = {"CRITICAL": "#d62728", "HIGH": "#ff7f0e", "MEDIUM": "#f2c744", "LOW": "#2ca02c"}

st.title("PrognoSys · Fleet Command Center")
st.caption(
    "NASA C-MAPSS turbofan fleets: real sensor readings, with every KPI derived from them"
    + (" · running on Snowflake" if config.SNOWFLAKE_MODE == "snowflake" else "")
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
    "between an emergency and a scheduled repair). Cost rates are planning assumptions."
)

with st.expander("How accurate is the model? (evaluated on NASA's held-out test engines)"):
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RUL RMSE", f"{metrics['rul']['overall']['rmse']:.1f} cycles")
    m2.metric(f"{config.PREDICTION_INTERVAL:.0%} interval coverage", f"{metrics['rul_interval']['empirical_coverage']:.0%}")
    m3.metric("Failure prediction AUC", f"{metrics['failure_probability']['roc_auc']:.3f}")
    m4.metric("Failures detected", f"{metrics['failure_probability']['recall_at_0.5']:.0%}")
    st.caption("AUC and detection rate are for failure within the next 30 cycles.")
    per_fleet = pd.DataFrame(metrics["rul"]["per_fleet"]).T.rename_axis("Fleet").reset_index().rename(columns={
        "rmse": "RMSE (cycles)", "mae": "Mean error (cycles)", "nasa_score": "NASA score", "n_engines": "Engines",
    })
    st.dataframe(
        per_fleet.style.format({"RMSE (cycles)": "{:.1f}", "Mean error (cycles)": "{:.1f}",
                                "NASA score": "{:,.0f}", "Engines": "{:.0f}"}),
        hide_index=True, width="stretch",
    )
    st.caption(
        f"Splitting by engine, not by row: a random row split reports RMSE "
        f"{metrics['baselines']['leaky_random_row_split_fd001_rmse']:.1f} on FD001 because it tests on engines "
        f"the model already saw. Predicting the mean scores {metrics['baselines']['constant_mean_rul_rmse']:.1f}."
    )

if config.SNOWFLAKE_MODE == "snowflake":
    from src.connection import get_session

    @st.fragment(run_every=15)
    def live_replay_panel():
        """Replay controls; refreshes the page when a new scoring run lands."""
        session = get_session()
        queued = session.sql("SELECT COUNT(*) FROM REPLAY_QUEUE").collect()[0][0]
        last_run = session.sql(
            "SELECT RUN_AT, NEW_SENSOR_ROWS FROM SCORING_RUNS ORDER BY RUN_AT DESC LIMIT 1"
        ).collect()[0]
        if st.session_state.get("last_scoring_run") not in (None, last_run["RUN_AT"]):
            st.session_state["last_scoring_run"] = last_run["RUN_AT"]
            st.cache_data.clear()
            st.rerun(scope="app")
        st.session_state["last_scoring_run"] = last_run["RUN_AT"]

        with st.expander("Live replay — stream real sensor readings through the pipeline", expanded=queued > 0):
            st.caption(
                "Rewind holds back each engine's last 20 real NASA cycles. Streaming sends them back in, "
                "and Snowflake rescores the fleet automatically in about a minute. This page refreshes itself."
            )
            r1, r2, r3, r4, r5 = st.columns([1, 1, 1, 1, 2])
            if r1.button("Rewind 20 cycles", disabled=queued > 0):
                with st.spinner("Rewinding and rescoring…"):
                    st.toast(session.sql("CALL REPLAY_PREPARE(20)").collect()[0][0])
                st.cache_data.clear()
                st.rerun(scope="app")
            if r2.button("Stream 1 cycle", disabled=queued == 0):
                st.toast(session.sql("CALL REPLAY_STEP(1)").collect()[0][0])
            if r3.button("Stream 5 cycles", disabled=queued == 0):
                st.toast(session.sql("CALL REPLAY_STEP(5)").collect()[0][0])
            if r4.button("Restore", disabled=queued == 0):
                st.toast(session.sql("CALL REPLAY_RESTORE()").collect()[0][0])
            r5.caption(
                f"{queued:,} readings queued · last scoring run {last_run['RUN_AT']:%H:%M:%S} UTC "
                f"({last_run['NEW_SENSOR_ROWS']:,} new readings)"
            )

    live_replay_panel()

try:
    backtest, backtest_curve = load_backtest()
except FileNotFoundError:
    backtest = None
if backtest:
    st.subheader("Business impact — backtest on all 709 real engine failures")
    predictive, fixed = backtest["predictive"], backtest["fixed_interval_same_catch_rate"]
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Failures caught in time", f"{predictive['catch_rate']:.1%}")
    b1.caption(f"{predictive['failures_caught']} of {predictive['engines']} engines")
    b2.metric("Shop visits vs. fixed interval", f"−{backtest['shop_visit_reduction']:.0%}")
    b2.caption(f"{predictive['shop_visits_per_100k_cycles']:.0f} vs {fixed['shop_visits_per_100k_cycles']:.0f} per 100k flight cycles")
    b3.metric("Engine life used", f"{predictive['mean_share_of_life_used']:.1%}")
    b3.caption(f"vs {fixed['mean_share_of_life_used']:.1%} with fixed intervals")
    b4.metric("Median warning", f"{predictive['median_warning_cycles']:.0f} cycles")
    b4.caption("before failure")
    with st.expander("How this was measured, and the trade-off curve"):
        st.caption(
            backtest["method"] + " Policy: " + backtest["headline_policy"] + ". The fixed-interval baseline overhauls "
            "each fleet at the single age that catches the same share of failures, chosen with hindsight from that "
            "fleet's own failure ages — so it is a generous baseline."
        )
        curve = backtest_curve.melt(
            id_vars=["RUL_LOWER_THRESHOLD", "CATCH_RATE"],
            value_vars=["PREDICTIVE_SHARE_OF_LIFE_USED", "FIXED_INTERVAL_SHARE_OF_LIFE_USED"],
            var_name="POLICY", value_name="SHARE_OF_LIFE_USED",
        ).replace({"PREDICTIVE_SHARE_OF_LIFE_USED": "Predictive (this model)",
                   "FIXED_INTERVAL_SHARE_OF_LIFE_USED": "Fixed interval, same catch rate"})
        fig = px.line(
            curve, x="RUL_LOWER_THRESHOLD", y="SHARE_OF_LIFE_USED", color="POLICY", markers=True,
            hover_data={"CATCH_RATE": ":.1%", "SHARE_OF_LIFE_USED": ":.1%"},
            color_discrete_sequence=[SERIES_1, SERIES_2],
            labels={"RUL_LOWER_THRESHOLD": "Pull engine when RUL lower bound ≤ (cycles)",
                    "SHARE_OF_LIFE_USED": "Share of engine life used", "POLICY": "", "CATCH_RATE": "Failures caught"},
            title="Engine life used before maintenance, at each trigger threshold",
        )
        fig.update_traces(line_width=2, marker_size=8)
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(legend=dict(orientation="h", y=-0.25), hovermode="x unified")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(
            backtest_curve[["RUL_LOWER_THRESHOLD", "CATCH_RATE", "PREDICTIVE_SHARE_OF_LIFE_USED",
                            "FIXED_INTERVAL_SHARE_OF_LIFE_USED", "PREDICTIVE_SHOP_VISITS_PER_100K",
                            "FIXED_INTERVAL_SHOP_VISITS_PER_100K"]].rename(columns={
                "RUL_LOWER_THRESHOLD": "Trigger (cycles)", "CATCH_RATE": "Failures caught",
                "PREDICTIVE_SHARE_OF_LIFE_USED": "Life used (predictive)",
                "FIXED_INTERVAL_SHARE_OF_LIFE_USED": "Life used (fixed)",
                "PREDICTIVE_SHOP_VISITS_PER_100K": "Shop visits / 100k (predictive)",
                "FIXED_INTERVAL_SHOP_VISITS_PER_100K": "Shop visits / 100k (fixed)",
            }).style.format({"Failures caught": "{:.1%}", "Life used (predictive)": "{:.1%}", "Life used (fixed)": "{:.1%}",
                             "Shop visits / 100k (predictive)": "{:.0f}", "Shop visits / 100k (fixed)": "{:.0f}"}),
            hide_index=True,
        )

st.divider()

left, right = st.columns([2, 3])

with left:
    st.subheader("Fleet")
    table = summary.assign(
        RUL_INTERVAL=summary["RUL_LOWER"].map("{:.0f}".format) + "–" + summary["RUL_UPPER"].map("{:.0f}".format)
    )
    display_cols = {"MACHINE_ID": "Engine", "FLEET": "Fleet", "RISK_CLASS": "Risk", "RUL_PREDICTION": "RUL (cycles)",
                    "RUL_INTERVAL": "90% interval", "FAILURE_PROBABILITY": "P(fail ≤ 30)", "OEE": "OEE"}
    st.dataframe(
        table[list(display_cols)].rename(columns=display_cols).style.apply(
            lambda row: [f"background-color: {RISK_COLOR[row['Risk']]}33"] * len(row), axis=1
        ).format({"RUL (cycles)": "{:.0f}", "P(fail ≤ 30)": "{:.0%}", "OEE": "{:.0%}"}),
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
        f"{int(feature_row['TIME_CYCLE'])} cycles flown"
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

    drivers = load_rul_drivers(selected_machine)
    if not drivers.empty:
        top = drivers.reindex(drivers["CONTRIBUTION_CYCLES"].abs().sort_values(ascending=False).index).head(6)
        top = top.assign(
            LABEL=top["SOURCE"] + " · " + top["DESCRIPTION"].str.split(" — ").str[0],
            EFFECT=top["CONTRIBUTION_CYCLES"].map(lambda v: "Lowers predicted RUL" if v < 0 else "Raises predicted RUL"),
        ).iloc[::-1]
        fig = px.bar(
            top, x="CONTRIBUTION_CYCLES", y="LABEL", orientation="h", color="EFFECT",
            color_discrete_map={"Lowers predicted RUL": LOWERS_RUL, "Raises predicted RUL": RAISES_RUL},
            hover_data={"DESCRIPTION": True, "CONTRIBUTION_CYCLES": ":.1f", "LABEL": False, "EFFECT": False},
            labels={"CONTRIBUTION_CYCLES": "Contribution to predicted RUL (cycles)", "LABEL": "", "EFFECT": ""},
            title=f"What drives this prediction (SHAP) — fleet baseline {drivers['BASE_RUL'].iloc[0]:.0f} cycles",
        )
        fig.update_traces(marker_cornerradius=4)
        fig.update_layout(legend=dict(orientation="h", y=-0.3), bargap=0.35)
        st.plotly_chart(fig, width="stretch")

    st.markdown("**Diagnosis**")
    reasoning = explain(selected_machine, machine_row.to_dict(), feature_row)
    st.write(f"- **Why it will fail:** {reasoning['why']}")
    st.write(f"- **Root cause:** {reasoning['root_cause']}")
    st.write(f"- **Recommended action:** {reasoning['recommended_action']}")
    st.write(f"- **Certainty:** {reasoning['confidence_note']}")
    if reasoning["citations"]:
        with st.expander(f"Sources — NASA C-MAPSS documentation ({len(reasoning['citations'])})"):
            for citation in reasoning["citations"]:
                st.caption(citation)
    st.caption(
        "Written by Snowflake Cortex from this engine's data and NASA's documentation"
        if reasoning["source"].startswith("Cortex") else "Rule-based summary (Cortex not available)"
    )

    st.markdown("**Actions**")
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
st.subheader("Shop plan — which engines to service, and when")
from src.scheduler import optimize_schedule

p1, p2 = st.columns(2)
capacity = p1.slider("Shop capacity (engines per slot)", 1, 40, config.SHOP_CAPACITY_PER_SLOT)
n_slots = p2.slider(f"Slots to plan (one every {config.SHOP_SLOT_CYCLES} cycles)", 1, 12, config.PLANNING_HORIZON_SLOTS)
plan = optimize_schedule(summary, capacity=capacity, n_slots=n_slots)
s1, s2, s3, s4 = st.columns(4)
s1.metric("Ground now", len(plan["ground_now"]))
s1.caption("likely to fail before the first slot")
s2.metric("Engines scheduled", len(plan["schedule"]))
s2.caption(f"of {plan['capacity_total']} slots available")
s3.metric("Expected downtime cost (optimized)", f"${plan['expected_cost_optimized'] / 1e6:,.1f}M")
s3.caption(f"${(plan['expected_cost_worst_first'] - plan['expected_cost_optimized']) / 1e6:,.1f}M less than servicing the worst engines first")
s4.metric("If nothing is scheduled", f"${plan['expected_cost_do_nothing'] / 1e6:,.1f}M")
st.caption(
    "A shop slot goes to an engine only when servicing it then is worth more than the risk of leaving it. "
    "Engines likely to fail before any slot opens are flagged to ground instead. Cost rates are planning assumptions."
)
g, sch = st.columns([1, 2])
g.markdown("**Ground now**")
g.dataframe(
    plan["ground_now"].rename(columns={"MACHINE_ID": "Engine", "RUL_PREDICTION": "RUL (cycles)",
                                       "RUL_LOWER": "RUL lower bound", "FAILURE_PROBABILITY": "P(fail ≤ 30)"}),
    hide_index=True, height=300,
)
sch.markdown("**Shop schedule**")
sch.dataframe(
    plan["schedule"].rename(columns={"SLOT": "Slot", "SERVICE_BY_CYCLE": "Service by cycle", "MACHINE_ID": "Engine",
                                     "P_FAIL_BEFORE_SERVICE": "Risk before service", "EXPECTED_SAVING_USD": "Expected saving"})
    .style.format({"Risk before service": "{:.0%}", "Expected saving": "${:,.0f}"}),
    hide_index=True, height=300,
)

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
