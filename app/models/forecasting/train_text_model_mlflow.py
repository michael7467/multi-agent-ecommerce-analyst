from __future__ import annotations

from pathlib import Path
import numpy as np
import joblib
import sklearn
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


NUMERIC_FEATURES = [
    "review_count",
    "avg_rating",
    "rating_std",
    "verified_purchase_ratio",
    "avg_review_length",
    "review_time_span",
]

TARGET_COLUMN = "price_class"

# Same embeddings the RAG pipeline already computes and uses for
# retrieval -- reused here rather than building a second, separate text
# pipeline. Paths match the confirmed, real config values from earlier
# this session (app/config/paths.py's EMBEDDINGS_PATH/METADATA_PATH).
REVIEW_EMBEDDINGS_PATH = "artifacts/embeddings/review_embeddings.npy"
REVIEW_METADATA_PATH = "artifacts/embeddings/review_embedding_metadata.csv"
EMBEDDING_DIM = 384
EMBEDDING_COLUMNS = [f"review_emb_{i}" for i in range(EMBEDDING_DIM)]

# Hyperparameters pulled out to named constants rather than left as
# magic numbers inline in build_pipeline() -- so they can be logged to
# MLflow by reference instead of duplicated as separate literals that
# could silently drift out of sync with what's actually used.
N_ESTIMATORS = 200
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_product_level_embeddings() -> pd.DataFrame:
    """Mean-pool per-review embeddings into one vector per product_id.

    Same embeddings the RAG pipeline already computes for retrieval --
    loaded here, not recomputed. Averaging is the simplest, standard
    approach for collapsing many reviews into one product-level signal;
    a real design choice, not a given, and worth revisiting if a
    weighted or attention-based pooling ever turns out to matter more.
    """
    embeddings = np.load(REVIEW_EMBEDDINGS_PATH)
    metadata = pd.read_csv(REVIEW_METADATA_PATH)

    if "product_id" not in metadata.columns:
        raise RuntimeError(
            f"{REVIEW_METADATA_PATH} has no 'product_id' column -- "
            "cannot aggregate review embeddings to product level without one."
        )
    if len(embeddings) != len(metadata):
        raise RuntimeError(
            f"Embeddings count ({len(embeddings)}) does not match "
            f"metadata row count ({len(metadata)})"
        )

    emb_df = pd.DataFrame(embeddings, columns=EMBEDDING_COLUMNS)
    emb_df["product_id"] = metadata["product_id"].values

    return emb_df.groupby("product_id", as_index=False)[EMBEDDING_COLUMNS].mean()


class TextForecastTrainer:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self.label_encoder = LabelEncoder()

    def prepare_data(self):
        df = self.df.copy()

        # Load-bearing assumption, not verified against the real file --
        # flagged clearly rather than silently assumed. Fails loudly and
        # immediately if wrong, instead of producing a silently broken
        # join.
        if "product_id" not in df.columns:
            raise RuntimeError(
                "electronics_labeled.csv has no 'product_id' column -- "
                "required to join review embeddings to training rows. "
                "This was an assumption, not a confirmed fact about this "
                "file -- check the real column names if this fires."
            )

        product_embeddings = load_product_level_embeddings()

        needed_cols = NUMERIC_FEATURES + ["product_id"] + [TARGET_COLUMN]
        df = df[needed_cols].copy()

        for col in NUMERIC_FEATURES:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=[TARGET_COLUMN])

        before = len(df)
        df = df.merge(product_embeddings, on="product_id", how="inner")
        dropped = before - len(df)
        if dropped:
            print(
                f"Dropped {dropped} of {before} rows with no matching "
                f"review embeddings for their product_id"
            )

        X = df[NUMERIC_FEATURES + EMBEDDING_COLUMNS]
        y = self.label_encoder.fit_transform(df[TARGET_COLUMN])

        return train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

    def build_pipeline(self) -> Pipeline:
        # Embeddings are pre-computed, numeric, dense vectors -- not raw
        # text the pipeline needs to vectorize itself, unlike TF-IDF.
        # They join the same imputer as the six numeric features rather
        # than getting their own ColumnTransformer branch.
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, NUMERIC_FEATURES + EMBEDDING_COLUMNS),
            ],
            remainder="drop",
        )

        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        return pipeline

    def train(self):
        X_train, X_test, y_train, y_test = self.prepare_data()
        pipeline = self.build_pipeline()

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "classification_report": classification_report(
                y_test,
                y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=False,
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }

        return pipeline, metrics


def save_model_artifacts(
    input_path: str = "data/processed/electronics_labeled.csv",
    model_output_path: str = "artifacts/models/price_class_model_with_text.joblib",
    encoder_output_path: str = "artifacts/models/price_class_label_encoder_with_text.joblib",
) -> None:
    df = pd.read_csv(input_path)

    trainer = TextForecastTrainer(df)

    with mlflow.start_run(run_name="price_class_text_model"):
        # Logged before training runs, not after -- so a run that fails
        # partway through training still leaves a record of what was
        # attempted, rather than only successful runs ever appearing.
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("embedding_dim", EMBEDDING_DIM)
        mlflow.log_param("embedding_source", REVIEW_EMBEDDINGS_PATH)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("input_rows", len(df))

        # This is the specific thing that would have caught the sklearn
        # version-skew issue hit earlier this session (model pickled with
        # 1.7.0, served on 1.9.0, discovered via a runtime warning, not
        # anything that had recorded which version trained it). Logging
        # the exact library versions at train time means a future mismatch
        # is immediately traceable instead of discovered by accident.
        mlflow.set_tag("sklearn_version", sklearn.__version__)
        mlflow.set_tag("joblib_version", joblib.__version__)

        pipeline, metrics = trainer.train()

        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_text(metrics["classification_report"], "classification_report.txt")

        # File paths below are unchanged -- predict.py loads the same
        # names. But what the model actually expects as input has
        # fundamentally changed, from raw title/categories text to
        # numeric embedding vectors -- predict.py would need updating to
        # compute or fetch a review embedding for any product it's
        # asked to score, not just to keep loading the same filenames.
        # Never seen predict.py's real source this session; flagging
        # this dependency clearly rather than leaving it hidden.
        Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(pipeline, model_output_path)
        joblib.dump(trainer.label_encoder, encoder_output_path)

        # Also logs the model into MLflow's own model store -- separate
        # from, not a replacement for, the joblib files above.
        # skops_trusted_types explicitly trusts numpy.dtype, which shows
        # up inside this pipeline's internal state (TfidfVectorizer /
        # ColumnTransformer) and MLflow's skops-based serializer otherwise
        # rejects by default -- a genuine, benign type here, not a reason
        # to fall back to less-secure raw pickle serialization instead.
        mlflow.sklearn.log_model(pipeline, name="model", skops_trusted_types=["numpy.dtype"])

        print(f"Saved model to: {model_output_path}")
        print(f"Saved label encoder to: {encoder_output_path}")
        print(f"MLflow run ID: {mlflow.active_run().info.run_id}")

        print("\nAccuracy:")
        print(metrics["accuracy"])

        print("\nClassification Report:")
        print(metrics["classification_report"])

        print("\nConfusion Matrix:")
        print(metrics["confusion_matrix"])


if __name__ == "__main__":
    save_model_artifacts()