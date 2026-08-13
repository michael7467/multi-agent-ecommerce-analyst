from __future__ import annotations

from pathlib import Path
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
from sklearn.feature_extraction.text import TfidfVectorizer


NUMERIC_FEATURES = [
    "review_count",
    "avg_rating",
    "rating_std",
    "verified_purchase_ratio",
    "avg_review_length",
    "review_time_span",
]

TEXT_FEATURES = [
    "title",
    "categories",
]

TARGET_COLUMN = "price_class"

# Hyperparameters pulled out to named constants rather than left as
# magic numbers inline in build_pipeline() -- so they can be logged to
# MLflow by reference instead of duplicated as separate literals that
# could silently drift out of sync with what's actually used.
N_ESTIMATORS = 200
TITLE_TFIDF_MAX_FEATURES = 3000
CATEGORIES_TFIDF_MAX_FEATURES = 2000
TFIDF_NGRAM_RANGE = (1, 2)
TEST_SIZE = 0.2
RANDOM_STATE = 42


class TextForecastTrainer:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self.label_encoder = LabelEncoder()

    def prepare_data(self):
        df = self.df.copy()

        needed_cols = NUMERIC_FEATURES + TEXT_FEATURES + [TARGET_COLUMN]
        df = df[needed_cols].copy()

        for col in TEXT_FEATURES:
            df[col] = df[col].fillna("").astype(str)

        for col in NUMERIC_FEATURES:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=[TARGET_COLUMN])

        X = df[NUMERIC_FEATURES + TEXT_FEATURES]
        y = self.label_encoder.fit_transform(df[TARGET_COLUMN])

        return train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

    def build_pipeline(self) -> Pipeline:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, NUMERIC_FEATURES),
                ("title_tfidf", TfidfVectorizer(max_features=TITLE_TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE), "title"),
                ("cat_tfidf", TfidfVectorizer(max_features=CATEGORIES_TFIDF_MAX_FEATURES, ngram_range=TFIDF_NGRAM_RANGE), "categories"),
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
        mlflow.log_param("title_tfidf_max_features", TITLE_TFIDF_MAX_FEATURES)
        mlflow.log_param("categories_tfidf_max_features", CATEGORIES_TFIDF_MAX_FEATURES)
        mlflow.log_param("tfidf_ngram_range", str(TFIDF_NGRAM_RANGE))
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

        # Existing artifact-saving behavior is completely unchanged below --
        # predict.py depends on these exact files at these exact paths, so
        # this isn't touched. Everything MLflow-related is additive.
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