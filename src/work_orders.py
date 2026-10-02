"""Approve or reject work orders drafted by the agent."""

import pandas as pd


def _session():
    from src.connection import get_session

    return get_session()


def load_work_orders() -> pd.DataFrame:
    return _session().sql("SELECT * FROM WORK_ORDERS ORDER BY CREATED_AT DESC").to_pandas()


def approve(work_order_id: str) -> str:
    session = _session()
    order = session.sql(
        "SELECT * FROM WORK_ORDERS WHERE WORK_ORDER_ID = ? AND STATUS = 'PENDING_APPROVAL'", params=[work_order_id]
    ).collect()
    if not order:
        raise ValueError(f"Work order {work_order_id} is not pending approval")
    order = order[0]

    from src.outcomes import log_action

    action_id = log_action(
        order["MACHINE_ID"],
        f"Work order: {order['WORK_TYPE']} ({order['PRIORITY']})",
        {
            "RUL_PREDICTION": order["RUL_PREDICTION_AT_DRAFT"],
            "RUL_LOWER": order["RUL_LOWER_AT_DRAFT"],
            "FAILURE_PROBABILITY": order["FAILURE_PROBABILITY_AT_DRAFT"],
            "RISK_CLASS": order["RISK_CLASS_AT_DRAFT"],
        },
    )
    session.sql(
        "UPDATE WORK_ORDERS SET STATUS = 'APPROVED', REVIEWED_AT = SYSDATE(), REVIEWED_BY = CURRENT_USER(), "
        "ACTION_ID = ? WHERE WORK_ORDER_ID = ?",
        params=[action_id, work_order_id],
    ).collect()
    return action_id


def reject(work_order_id: str) -> None:
    _session().sql(
        "UPDATE WORK_ORDERS SET STATUS = 'REJECTED', REVIEWED_AT = SYSDATE(), REVIEWED_BY = CURRENT_USER() "
        "WHERE WORK_ORDER_ID = ? AND STATUS = 'PENDING_APPROVAL'",
        params=[work_order_id],
    ).collect()
