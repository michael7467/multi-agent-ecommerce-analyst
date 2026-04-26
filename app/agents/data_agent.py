from __future__ import annotations

import pandas as pd
from app.agents.base_agent import BaseAgent
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent
from app.config.paths import FEATURES_PATH

logger = get_logger("agents.data_agent")

class DataAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="DataAgent")
        self.features_df = pd.read_csv(FEATURES_PATH)

        # Validate schema
        if "product_id" not in self.features_df.columns:
            raise ValueError("DataAgent: 'product_id' column missing in dataset")

    @traced_agent("DataAgent.run")
    def run(self, product_id: str) -> dict:
        matches = self.features_df[
            self.features_df["product_id"].astype(str) == str(product_id)
        ]

        if matches.empty:
            logger.warning(f"{self.name}: product not found: {product_id}")
            raise ValueError(f"{self.name}: product not found: {product_id}")

        return matches.iloc[0].to_dict()
