"""Fleet Command Center — Streamlit in Snowflake edition."""

import json
import uuid

import plotly.express as px
import streamlit as st
from snowflake.snowpark.context import get_active_session

st.set_page_config(page_title="Fleet Command Center", layout="wide")

session = get_active_session()

RISK_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
RISK_COLOR = {"CRITICAL": "#d62728", "HIGH": "#ff7f0e", "MEDIUM": "#f2c744", "LOW": "#2ca02c"}
RAISES_RUL, LOWERS_RUL = "#2a78d6", "#e34948"


def run_query(sql):
    return session.sql(sql).to_pandas()


@st.cache_data(ttl=60)
def load_fleet():
    return run_query("""
        SELECT p.*, m.FLEET, m.OPERATING_CONDITIONS, m.FAULT_MODES
        FROM FLEET_PREDICTIONS p
        JOIN MACHINE_DATA m ON p.MACHINE_ID = m.MACHINE_ID
        WHERE m.STATUS = 'IN_SERVICE'
        ORDER BY
            CASE p.RISK_CLASS
                WHEN 'CRITICAL' THEN 0 WHEN 'HIGH' THEN 1
                WHEN 'MEDIUM' THEN 2 ELSE 3 END,
            p.RUL_LOWER
    """)


@st.cache_data(ttl=60)
def load_model_metrics():
    return run_query("""
        SELECT METRICS, MODEL_VERSION, TRAINED_AT
        FROM MODEL_METRICS ORDER BY TRAINED_AT DESC LIMIT 1
    """)


@st.cache_data(ttl=60)
def load_drivers(machine_id):
    return session.sql(
        "SELECT * FROM ENGINE_RUL_DRIVERS"
        " WHERE MACHINE_ID = ?"
        " ORDER BY ABS(CONTRIBUTION_CYCLES) DESC"
        " LIMIT 8",
        params=[machine_id],
    ).to_pandas()


VALID_SENSORS = {f"SENSOR_{i}" for i in range(1, 22)}


@st.cache_data(ttl=60)
def load_sensor_history(machine_id, sensor):
    if sensor not in VALID_SENSORS:
        return session.create_dataframe([], schema=["TIME_CYCLE", "VALUE"]).to_pandas()
    return session.sql(
        f"SELECT TIME_CYCLE, {sensor} FROM RAW_SENSOR_DATA"
        " WHERE MACHINE_ID = ?"
        " ORDER BY TIME_CYCLE",
        params=[machine_id],
    ).to_pandas()


@st.cache_data(ttl=60)
def load_work_orders():
    return run_query("SELECT * FROM WORK_ORDERS ORDER BY CREATED_AT DESC")


@st.cache_data(ttl=60)
def load_scoring_runs():
    return run_query("SELECT * FROM SCORING_RUNS ORDER BY RUN_AT DESC LIMIT 1")


# 1. Fleet KPIs
st.title("Fleet Command Center")

fleet_df = load_fleet()
metrics_row = load_model_metrics()

metrics = json.loads(metrics_row["METRICS"].iloc[0]) if len(metrics_row) > 0 else {}
model_version = metrics_row["MODEL_VERSION"].iloc[0] if len(metrics_row) > 0 else "?"
scored_at = metrics_row["TRAINED_AT"].iloc[0] if len(metrics_row) > 0 else "?"
rmse = metrics.get("rul", {}).get("overall", {}).get("rmse", "?")

fleets = st.multiselect("Fleets", sorted(fleet_df["FLEET"].unique()),
                         default=sorted(fleet_df["FLEET"].unique()))
view = fleet_df[fleet_df["FLEET"].isin(fleets)]
if view.empty:
    st.info("Select at least one fleet.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Engines in service", len(view))
c2.metric("Critical / High", int(view["RISK_CLASS"].isin(["CRITICAL", "HIGH"]).sum()))
c3.metric("Model version", model_version)
c4.metric("RUL RMSE", f"{rmse:.1f} cycles" if isinstance(rmse, (int, float)) else rmse)
st.caption(f"Last scoring: {scored_at}")

# 2. Fleet table
st.subheader("Fleet overview")
display = view[["MACHINE_ID", "FLEET", "RISK_CLASS", "RUL_PREDICTION",
                "RUL_LOWER", "RUL_UPPER", "FAILURE_PROBABILITY"]].copy()
display["RUL_INTERVAL"] = display["RUL_LOWER"].map("{:.0f}".format) + "\u2013" + display["RUL_UPPER"].map("{:.0f}".format)
st.dataframe(
    display[["MACHINE_ID", "FLEET", "RISK_CLASS", "RUL_PREDICTION",
             "RUL_INTERVAL", "FAILURE_PROBABILITY"]]
    .style
    .apply(lambda row: [f"background-color: {RISK_COLOR.get(row['RISK_CLASS'], '#ffffff')}33"] * len(row), axis=1)
    .format({"RUL_PREDICTION": "{:.0f}", "FAILURE_PROBABILITY": "{:.0%}"}),
    hide_index=True, use_container_width=True, height=420,
)

# 3. Engine detail
st.divider()
selected = st.selectbox("Inspect engine", view["MACHINE_ID"])
row = view[view["MACHINE_ID"] == selected].iloc[0]

st.subheader(f"{selected} \u00b7 fleet {row['FLEET']}")
st.caption(f"{row['OPERATING_CONDITIONS']} operating condition(s) \u00b7 fault modes: {row['FAULT_MODES']}")

m1, m2, m3, m4 = st.columns(4)
m1.metric("RUL (cycles)", f"{row['RUL_PREDICTION']:.0f}")
m2.metric("90% interval", f"{row['RUL_LOWER']:.0f}\u2013{row['RUL_UPPER']:.0f}")
m3.metric("Risk", row["RISK_CLASS"])
m4.metric("P(fail \u2264 30 cy)", f"{row['FAILURE_PROBABILITY']:.0%}")

drivers = load_drivers(selected)
if not drivers.empty:
    top = drivers.head(6).copy()
    top["LABEL"] = top["SOURCE"] + " \u00b7 " + top["DESCRIPTION"].str.split(" \u2014 ").str[0]
    top["EFFECT"] = top["CONTRIBUTION_CYCLES"].apply(
        lambda v: "Lowers predicted RUL" if v < 0 else "Raises predicted RUL"
    )
    top = top.iloc[::-1]
    fig = px.bar(
        top, x="CONTRIBUTION_CYCLES", y="LABEL", orientation="h", color="EFFECT",
        color_discrete_map={"Lowers predicted RUL": LOWERS_RUL, "Raises predicted RUL": RAISES_RUL},
        labels={"CONTRIBUTION_CYCLES": "Contribution (cycles)", "LABEL": "", "EFFECT": ""},
        title=f"SHAP drivers \u2014 baseline {top['BASE_RUL'].iloc[0]:.0f} cycles",
    )
    fig.update_traces(marker_cornerradius=4)
    fig.update_layout(legend=dict(orientation="h", y=-0.3), bargap=0.35)
    st.plotly_chart(fig, use_container_width=True)

# fastest-drifting sensor from top negative SHAP driver
drift_sensor = None
if not drivers.empty:
    neg = drivers[drivers["CONTRIBUTION_CYCLES"] < 0]
    if not neg.empty:
        drift_sensor = neg.iloc[0]["SOURCE"]
if drift_sensor and drift_sensor.startswith("SENSOR_"):
    sensor_hist = load_sensor_history(selected, drift_sensor)
    if not sensor_hist.empty:
        fig2 = px.line(sensor_hist, x="TIME_CYCLE", y=drift_sensor,
                       title=f"Sensor history: {drift_sensor}")
        st.plotly_chart(fig2, use_container_width=True)

# 4. Work orders
st.divider()
st.subheader("Work orders awaiting approval")

orders = load_work_orders()
pending = orders[orders["STATUS"] == "PENDING_APPROVAL"]

if pending.empty:
    st.caption("No drafts waiting. Ask the maintenance agent to draft work orders.")

for idx, wo in pending.iterrows():
    with st.container(border=True):
        col_info, col_approve, col_reject = st.columns([5, 1, 1])
        col_info.markdown(
            f"**{wo['PRIORITY']} {wo['WORK_TYPE']}** \u00b7 {wo['MACHINE_ID']} \u00b7 "
            f"{wo['RISK_CLASS_AT_DRAFT']}, RUL {wo['RUL_PREDICTION_AT_DRAFT']:.0f} at draft"
        )
        col_info.caption(wo["JUSTIFICATION"])

        if col_approve.button("Approve", key=f"approve-{wo['WORK_ORDER_ID']}", type="primary"):
            action_id = str(uuid.uuid4())
            session.sql(
                "INSERT INTO ACTION_OUTCOMES "
                "(ACTION_ID, MACHINE_ID, ACTION_TYPE, RECOMMENDED_AT, "
                "RUL_PREDICTION_AT_ACTION, RUL_LOWER_AT_ACTION, "
                "FAILURE_PROBABILITY_AT_ACTION, RISK_CLASS_AT_ACTION, "
                "TAKEN_FLAG, TAKEN_AT) "
                "SELECT ?, MACHINE_ID, "
                "'Work order: ' || WORK_TYPE || ' (' || PRIORITY || ')', "
                "SYSDATE(), "
                "RUL_PREDICTION_AT_DRAFT, RUL_LOWER_AT_DRAFT, "
                "FAILURE_PROBABILITY_AT_DRAFT, RISK_CLASS_AT_DRAFT, "
                "TRUE, SYSDATE() "
                "FROM WORK_ORDERS WHERE WORK_ORDER_ID = ?",
                params=[action_id, wo["WORK_ORDER_ID"]],
            ).collect()
            session.sql(
                "UPDATE WORK_ORDERS "
                "SET STATUS = 'APPROVED', REVIEWED_AT = SYSDATE(), "
                "REVIEWED_BY = CURRENT_USER(), ACTION_ID = ? "
                "WHERE WORK_ORDER_ID = ?",
                params=[action_id, wo["WORK_ORDER_ID"]],
            ).collect()
            st.cache_data.clear()
            st.rerun()

        if col_reject.button("Reject", key=f"reject-{wo['WORK_ORDER_ID']}"):
            session.sql(
                "UPDATE WORK_ORDERS "
                "SET STATUS = 'REJECTED', REVIEWED_AT = SYSDATE(), "
                "REVIEWED_BY = CURRENT_USER() "
                "WHERE WORK_ORDER_ID = ? AND STATUS = 'PENDING_APPROVAL'",
                params=[wo["WORK_ORDER_ID"]],
            ).collect()
            st.cache_data.clear()
            st.rerun()

reviewed = orders[orders["STATUS"] != "PENDING_APPROVAL"]
if not reviewed.empty:
    with st.expander(f"Reviewed work orders ({len(reviewed)})"):
        st.dataframe(
            reviewed[["MACHINE_ID", "WORK_TYPE", "PRIORITY", "STATUS",
                       "REVIEWED_BY", "REVIEWED_AT", "CREATED_BY"]],
            hide_index=True, use_container_width=True,
        )

# 5. Live replay
st.divider()
st.subheader("Live replay")
st.caption(
    "Rewind holds back each engine's last 20 real NASA cycles. Streaming sends them back in, "
    "and Snowflake rescores the fleet automatically in about a minute."
)

queued = session.sql("SELECT COUNT(*) AS N FROM REPLAY_QUEUE").to_pandas()["N"].iloc[0]
scoring = load_scoring_runs()

r1, r2, r3, r4, r5 = st.columns([1, 1, 1, 1, 2])
if r1.button("Rewind 20 cycles", disabled=int(queued) > 0):
    session.sql("CALL REPLAY_PREPARE(20)").collect()
    st.cache_data.clear()
    st.rerun()
if r2.button("Stream 1 cycle", disabled=int(queued) == 0):
    session.sql("CALL REPLAY_STEP(1)").collect()
if r3.button("Stream 5 cycles", disabled=int(queued) == 0):
    session.sql("CALL REPLAY_STEP(5)").collect()
if r4.button("Restore", disabled=int(queued) == 0):
    session.sql("CALL REPLAY_RESTORE()").collect()
    st.cache_data.clear()
    st.rerun()

if len(scoring) > 0:
    last = scoring.iloc[0]
    r5.caption(f"{int(queued):,} readings queued \u00b7 last scoring {last['RUN_AT']} ({last['NEW_SENSOR_ROWS']:,} new rows)")
else:
    r5.caption(f"{int(queued):,} readings queued")
