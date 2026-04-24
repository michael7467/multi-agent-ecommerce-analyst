from __future__ import annotations

import faiss
import numpy as np
import pandas as pd

from app.logging.logger import get_logger
from app.observability.agent_tracing import traced_agent
from app.config.paths import IMAGE_FAISS_INDEX_PATH, IMAGE_METADATA_PATH

logger = get_logger("retriever.image")


class ImageRetriever:
    def __init__(
        self,
        index_path: str = str(IMAGE_FAISS_INDEX_PATH),
        metadata_path: str = str(IMAGE_METADATA_PATH),
    ) -> None:
        try:
            self.index = faiss.read_index(index_path)
        except Exception:
            logger.error("Failed to load FAISS index", exc_info=True)
            raise

        try:
            self.metadata = pd.read_csv(metadata_path)
        except Exception:
            logger.error("Failed to load image metadata CSV", exc_info=True)
            raise