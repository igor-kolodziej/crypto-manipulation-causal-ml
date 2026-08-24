"""Group-aware predictive evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

from .schema import FEATURE_COLUMNS, validate_events


@dataclass(frozen=True)
class GroupedSplit:
    train_index: np.ndarray
    test_index: np.ndarray


def make_grouped_holdout(
    events: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> GroupedSplit:
    """Split by channel so no channel occurs in both train and test data."""

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    validated = validate_events(events)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_index, test_index = next(
        splitter.split(validated, validated["success"], groups=validated["channel_id"])
    )
    return GroupedSplit(train_index=train_index, test_index=test_index)


def run_grouped_benchmark(
    events: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> dict[str, float | int]:
    """Train a compact gradient-boosting baseline and return holdout metrics."""

    validated = validate_events(events)
    split = make_grouped_holdout(validated, test_size=test_size, seed=seed)
    train = validated.iloc[split.train_index]
    test = validated.iloc[split.test_index]

    classifier = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=140,
        max_leaf_nodes=15,
        l2_regularization=0.2,
        random_state=seed,
    )
    classifier.fit(train[FEATURE_COLUMNS], train["success"])
    prediction = classifier.predict(test[FEATURE_COLUMNS])

    train_channels = set(train["channel_id"])
    test_channels = set(test["channel_id"])
    if train_channels & test_channels:
        raise RuntimeError("Grouped holdout leaked channels across train and test")

    return {
        "train_events": int(len(train)),
        "test_events": int(len(test)),
        "train_channels": int(len(train_channels)),
        "test_channels": int(len(test_channels)),
        "accuracy": float(accuracy_score(test["success"], prediction)),
        "macro_f1": float(f1_score(test["success"], prediction, average="macro")),
        "success_precision": float(precision_score(test["success"], prediction, zero_division=0)),
        "success_recall": float(recall_score(test["success"], prediction, zero_division=0)),
    }

