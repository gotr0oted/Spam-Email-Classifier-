from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "spam_dataset.csv"


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    data = pd.read_csv(dataset_path)
    required_columns = {"label", "message"}
    if not required_columns.issubset(data.columns):
        raise ValueError(
            f"Dataset must contain columns {required_columns}, found {set(data.columns)}"
        )
    return data


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("classifier", MultinomialNB()),
        ]
    )


def train_model(dataset_path: Path) -> Pipeline:
    data = load_dataset(dataset_path)
    x_train, x_test, y_train, y_test = train_test_split(
        data["message"],
        data["label"],
        test_size=0.25,
        random_state=42,
        stratify=data["label"],
    )

    model = build_model()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Model accuracy: {accuracy:.2f}")
    print("Classification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    return model


def predict_message(model: Pipeline, message: str) -> str:
    return str(model.predict([message])[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a spam email/message classifier and predict new input."
    )
    parser.add_argument(
        "--message",
        type=str,
        help="New message text to classify after training.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to a CSV dataset with 'label' and 'message' columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = train_model(args.dataset)

    if args.message:
        prediction = predict_message(model, args.message)
        print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
