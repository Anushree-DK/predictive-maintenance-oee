"""Model Registry wrapper exposing predict and explain."""

import joblib
import pandas as pd
from snowflake.ml.model import custom_model

from src.ml.explain import sensor_contributions
from src.ml.predict import predict_with_bundle


class RulModel(custom_model.CustomModel):
    def __init__(self, context: custom_model.ModelContext) -> None:
        super().__init__(context)
        self.bundle = joblib.load(self.context.path("bundle"))

    @custom_model.inference_api
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return predict_with_bundle(self.bundle, X, id_col=None)

    @custom_model.inference_api
    def explain(self, X: pd.DataFrame) -> pd.DataFrame:
        return sensor_contributions(self.bundle, X)
