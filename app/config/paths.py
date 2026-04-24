import os
from pathlib import Path

def env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default))

# Data
FEATURES_PATH = env_path(
    "FEATURES_PATH",
    "data/processed/electronics_labeled.csv"
)

SENTIMENT_FEATURES_PATH = env_path(
    "SENTIMENT_FEATURES_PATH",
    "data/processed/electronics_sentiment_features.csv"
)

REVIEWS_PATH = env_path(
    "REVIEWS_PATH",
    "data/interim/reviews_electronics_clean.csv"
)

# Review embeddings
REVIEW_EMBEDDINGS_PATH = env_path(
    "REVIEW_EMBEDDINGS_PATH",
    "artifacts/embeddings/review_embeddings.npy"
)

REVIEW_METADATA_PATH = env_path(
    "REVIEW_METADATA_PATH",
    "artifacts/embeddings/review_embedding_metadata.csv"
)

REVIEW_FAISS_INDEX_PATH = env_path(
    "REVIEW_FAISS_INDEX_PATH",
    "artifacts/indexes/review_faiss.index"
)

# Image embeddings
IMAGE_FAISS_INDEX_PATH = env_path(
    "IMAGE_FAISS_INDEX_PATH",
    "artifacts/indexes/image_faiss.index"
)

IMAGE_METADATA_PATH = env_path(
    "IMAGE_METADATA_PATH",
    "artifacts/embeddings/image_embedding_metadata.csv"
)

# Topic modeling
TOPIC_MODELING_OUTPUT_DIR = env_path(
    "TOPIC_MODELING_OUTPUT_DIR",
    "artifacts/topic_modeling"
)

EMBEDDINGS_PATH = env_path(
    "EMBEDDINGS_PATH",
    "artifacts/embeddings/review_embeddings.npy"
)
METADATA_PATH = env_path(
    "METADATA_PATH",        
    "artifacts/embeddings/review_embedding_metadata.csv"
)

TOPIC_KEYWORDS_PATH = env_path(
    "TOPIC_KEYWORDS_PATH",
    "artifacts/topic_modeling/topic_keywords_global.csv"
)

OUTPUT_DIR = env_path(
    "OUTPUT_DIR",
    "artifacts/topic_modeling"
)