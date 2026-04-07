from __future__ import annotations

import pandas as pd

from app.agents.base_agent import BaseAgent


FEATURES_PATH = "data/processed/electronics_labeled.csv"


class DataAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="DataAgent")
        self.features_df = pd.read_csv(FEATURES_PATH)

    def run(self, product_id: str) -> dict:
        matches = self.features_df[self.features_df["product_id"].astype(str) == str(product_id)]
        if matches.empty:
            raise ValueError(f"{self.name}: product not found: {product_id}")

        return matches.iloc[0].to_dict()