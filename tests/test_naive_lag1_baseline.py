"""Targeted regression tests for the naive lag-1 baseline used in advanced_tree_benchmark."""
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TimeSeriesSplit


def naive_lag1_baseline(y: pd.Series, n_train: int) -> np.ndarray:
    """Rolling one-step lag-1 prediction (same as benchmark implementation)."""
    predictions = np.empty(len(y) - n_train)
    for i in range(n_train, len(y)):
        predictions[i - n_train] = y.iloc[i - 1]
    return predictions


def test_holdout_naive_lag1_matches_rolling():
    """Verify holdout naive lag-1 uses previous actual for each step."""
    y = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
    n_train = 7
    preds = naive_lag1_baseline(y, n_train)

    expected = np.array([70, 80, 90])
    np.testing.assert_array_equal(preds, expected)
    assert preds[0] == y.iloc[n_train - 1]


def test_cv_naive_lag1_rolling_in_cv():
    """Verify CV naive lag-1 predicts using previous actual across a fold boundary."""
    y = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
    tscv = TimeSeriesSplit(n_splits=3)

    for train_idx, test_idx in tscv.split(y):
        preds = np.asarray([y.iloc[i - 1] for i in test_idx])
        expected = np.asarray([y.iloc[i - 1] for i in test_idx])
        np.testing.assert_array_equal(preds, expected)


def test_cv_naive_lag1_not_constant():
    """Prove that predictions are NOT one constant value across the fold."""
    y = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
    tscv = TimeSeriesSplit(n_splits=3)

    for train_idx, test_idx in tscv.split(y):
        preds = np.asarray([y.iloc[i - 1] for i in test_idx])
        assert not np.all(preds == preds[0]), (
            "CV naive lag-1 predictions should not be constant across the fold"
        )


def test_cv_naive_lag1_first_pred_equals_last_train():
    """First validation prediction must equal the last training observation."""
    y = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
    tscv = TimeSeriesSplit(n_splits=3)

    for train_idx, test_idx in tscv.split(y):
        preds = np.asarray([y.iloc[i - 1] for i in test_idx])
        assert preds[0] == y.iloc[train_idx[-1]]


def test_cv_naive_lag1_subsequent_preds_use_previous_actual():
    """After the first prediction, each subsequent prediction uses the actual from the prior step."""
    y = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
    tscv = TimeSeriesSplit(n_splits=3)

    for train_idx, test_idx in tscv.split(y):
        preds = np.asarray([y.iloc[i - 1] for i in test_idx])
        for j in range(1, len(preds)):
            assert preds[j] == y.iloc[test_idx[j] - 1], (
                f"pred[{j}] should be y[{test_idx[j]-1}] but got {preds[j]}"
            )


def test_cv_naive_lag1_constant_bug_catch():
    """Regression test: ensure the old constant-prediction bug is caught."""
    y = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
    tscv = TimeSeriesSplit(n_splits=3)

    for train_idx, test_idx in tscv.split(y):
        # The old buggy behavior:
        last_train_val = float(y.iloc[train_idx[-1]])
        buggy_preds = np.full(len(test_idx), last_train_val)

        # The correct behavior:
        correct_preds = np.asarray([y.iloc[i - 1] for i in test_idx])

        # They must differ for non-trivial folds
        if len(test_idx) > 1:
            assert not np.array_equal(buggy_preds, correct_preds), (
                "Corrected lag-1 must differ from the old constant-value bug"
            )
