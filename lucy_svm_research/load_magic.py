
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ucimlrepo import fetch_ucirepo


RANDOM_STATE = 42
SAMPLE_SIZE = 1200

FEATURE_NAMES = [
    "fLength",
    "fWidth",
    "fSize",
    "fConc",
    "fConc1",
    "fAsym",
    "fM3Long",
    "fM3Trans",
    "fAlpha",
    "fDist",
]


def main():
    dataset = fetch_ucirepo(id=159)

    X = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:, 0].copy()

    assert X.shape == (19020, 10), X.shape
    assert not X.isna().any().any()
    assert not y.isna().any()
    assert set(y.unique()) == {"g", "h"}

    X.columns = FEATURE_NAMES
    y = y.map({"g": 1, "h": -1})

    indices = np.arange(len(y))

    sample_idx, _ = train_test_split(
        indices,
        train_size=SAMPLE_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_sample = X.iloc[sample_idx].reset_index(drop=True)
    y_sample = y.iloc[sample_idx].reset_index(drop=True)

    print("\n--- MAGIC Dataset ---")
    print("Full observations:", len(X))
    print("Features:", X.shape[1])
    print("Missing values:", X.isna().sum().sum())
    print("\nFull class distribution:")
    print(y.value_counts().sort_index())

    print("\n--- Reproducible Research Sample ---")
    print("Observations:", len(X_sample))
    print("Features:", X_sample.shape[1])
    print("Class distribution:")
    print(y_sample.value_counts().sort_index())
    print("Feature names:", list(X_sample.columns))

    output_dir = Path(__file__).resolve().parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = X_sample.copy()
    sample["target"] = y_sample
    output_path = output_dir / "magic_sample.csv"
    sample.to_csv(output_path, index=False)

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
