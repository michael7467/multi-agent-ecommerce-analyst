from __future__ import annotations

from pathlib import Path
import joblib
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
            test_size=0.2,
            random_state=42,
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
                ("title_tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2)), "title"),
                ("cat_tfidf", TfidfVectorizer(max_features=2000, ngram_range=(1, 2)), "categories"),
            ],
            remainder="drop",
        )

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
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
    pipeline, metrics = trainer.train()

    Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_output_path)
    joblib.dump(trainer.label_encoder, encoder_output_path)

    print(f"Saved model to: {model_output_path}")
    print(f"Saved label encoder to: {encoder_output_path}")

    print("\nAccuracy:")
    print(metrics["accuracy"])

    print("\nClassification Report:")
    print(metrics["classification_report"])

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])


if __name__ == "__main__":
    save_model_artifacts()