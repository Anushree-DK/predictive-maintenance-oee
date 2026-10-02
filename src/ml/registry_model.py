"""Snowflake Model Registry wrapper around the trained bundle (src/ml/train.py), so
the fleet is scored inside Snowflake by the registered model rather than by a local
joblib file. Logged by src/snowflake_pipeline.py::train_procedure.
"""

import joblib
import pandas as pd
from snowflake.ml.model import custom_model

from src.ml.predict import predict_with_bundle


class RulModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)
        self.bundle = joblib.load(self.context.path("bundle"))

    @custom_model.inference_api
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return predict_with_bundle(self.bundle, X, id_col=None)
