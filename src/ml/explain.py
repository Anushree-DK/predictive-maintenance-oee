"""Why the model predicts what it does, per engine: exact SHAP values (TreeExplainer)
of the point-RUL model, summed per sensor over its four features (mean, std, slope,
latest), in cycles of RUL. Negative = this sensor is pulling the prediction down.

Contributions plus BASE_RUL add up exactly to the point prediction (before capping).
"""

import numpy as np
import pandas as pd

import config

SOURCES = config.SENSOR_COLUMNS + ["TIME_CYCLE"]


def sensor_contributions(bundle: dict, feature_table: pd.DataFrame) -> pd.DataFrame:
    """One row per input row: BASE_RUL plus one column per sensor (and TIME_CYCLE)."""
    import shap

    cols = bundle["feature_columns"]
    explainer = shap.TreeExplainer(bundle["point"])
    values = explainer.shap_values(feature_table[cols])

    out = pd.DataFrame({"BASE_RUL": np.full(len(feature_table), float(np.ravel(explainer.expected_value)[0]))})
    for source in SOURCES:
        idx = [i for i, c in enumerate(cols) if c == source or c.startswith(f"{source}_")]
        out[f"SHAP_{source}"] = values[:, idx].sum(axis=1)
    return out.round(3)


def to_long(machine_ids, contributions: pd.DataFrame) -> pd.DataFrame:
    """Wide contributions -> one row per (engine, sensor), ranked by impact."""
    long = contributions.assign(MACHINE_ID=list(machine_ids)).melt(
        id_vars=["MACHINE_ID", "BASE_RUL"], var_name="SOURCE", value_name="CONTRIBUTION_CYCLES"
    )
    long["SOURCE"] = long["SOURCE"].str.removeprefix("SHAP_")
    long["DESCRIPTION"] = long["SOURCE"].map(config.SENSOR_DESCRIPTIONS).fillna("engine age (cycles flown)")
    long["IMPACT_RANK"] = long.groupby("MACHINE_ID")["CONTRIBUTION_CYCLES"].rank(method="first", ascending=True).astype(int)
    return long.sort_values(["MACHINE_ID", "IMPACT_RANK"]).reset_index(drop=True)
