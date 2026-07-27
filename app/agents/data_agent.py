from __future__ import annotations

import pandas as pd
from app.agents.base_agent import BaseAgent
from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent
from app.config.paths import FEATURES_PATH

logger = get_logger("agents.data_agent")

_FEATURES_DF: pd.DataFrame | None = None  # shared across every DataAgent instance


def _load_features_df() -> pd.DataFrame:
    """Loads the features table once per process, shared across every
    consumer -- not just every DataAgent instance.

    Several services (e.g. CompetitiveService, TrendDetectionService)
    construct their own DataAgent or independently need this same table.
    Cached at module level so it's read and parsed from disk exactly once,
    however many places end up needing it.

    Kept as a plain DataFrame (product_id as a normal column, not the
    index) specifically because it's shared: DataAgent wants an indexed
    view for fast lookups, but TrendDetectionService merges on product_id
    as an ordinary column, and pandas' merge() raises "'product_id' is
    both an index level and a column label" if the shared copy has it as
    the index too. Found this the hard way. Each consumer that wants an
    index builds its own local one from this shared, already-loaded data
    -- no extra disk I/O, just an in-memory .set_index() call.

    Not thread-safe against the very first concurrent constructions (a
    narrow race could load the CSV twice during startup) -- same tradeoff
    already accepted elsewhere in this codebase for the _MODEL singleton
    pattern in the retriever modules. Not a correctness issue, just an
    occasional redundant load during warm-up.
    """
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
        # Shared, plain DataFrame -- do not mutate in place, see
        # _load_features_df. The indexed view below is local to this
        # instance, built cheaply in-memory from data that's already
        # loaded, not re-read from disk.
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
            # Non-unique product_id in the source data -- .loc returns all
            # matching rows instead of one. Previously silently took
            # iloc[0] with no record that a duplicate existed at all.
            logger.warning(
                f"{self.name}: multiple rows found for product_id={product_id}, "
                f"using the first"
            )
            row = row.iloc[0]

        return row.to_dict()