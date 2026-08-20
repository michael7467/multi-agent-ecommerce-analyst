from __future__ import annotations

import pandas as pd
from app.agents.base_agent import BaseAgent
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent
from app.config.paths import FEATURES_PATH

logger = get_logger("agents.data_agent")

_FEATURES_DF: pd.DataFrame | None = None  # shared across every DataAgent instance


def _load_features_df() -> pd.DataFrame:

    global _FEATURES_DF
    if _FEATURES_DF is None:
        df = pd.read_csv(FEATURES_PATH)

        if "product_id" not in df.columns:
            raise ValueError("DataAgent: 'product_id' column missing in dataset")

        df["product_id"] = df["product_id"].astype(str)
        _FEATURES_DF = df

    return _FEATURES_DF


class DataAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="DataAgent")
   
        self.features_df = _load_features_df().set_index("product_id", drop=False)

    @traced_agent("DataAgent.run")
    def run(self, product_id: str) -> dict:
        product_id = str(product_id)

        try:
            row = self.features_df.loc[product_id]
        except KeyError:
            logger.warning(f"{self.name}: product not found: {product_id}")
            raise ValueError(f"{self.name}: product not found: {product_id}")

        if isinstance(row, pd.DataFrame):
      
            logger.warning(
                f"{self.name}: multiple rows found for product_id={product_id}, "
                f"using the first"
            )
            row = row.iloc[0]

        return row.to_dict()