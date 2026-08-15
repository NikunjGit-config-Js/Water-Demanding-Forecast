import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from experiments.phase0_baseline import make_windows, run_phase0_baseline


def test_make_windows_accesses_only_prior_rows() -> None:
    values = np.arange(6, dtype=float).reshape(-1, 1)
    features, targets = make_windows(values, 3)
    np.testing.assert_array_equal(features, [[0, 1, 2], [1, 2, 3], [2, 3, 4]])
    np.testing.assert_array_equal(targets.ravel(), [3, 4, 5])


def test_phase0_runner_preserves_complete_artifact_set(tmp_path: Path) -> None:
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    dataset = tmp_path / "dataset.csv"
    pd.DataFrame({"Date": dates, "Consumption": np.arange(1, 31, dtype=float)}).to_csv(
        dataset, index=False
    )

    output = run_phase0_baseline(
        dataset, tmp_path / "artifacts", "test_run", test_fraction=0.2, window_size=5
    )
    expected = {
        "actual_vs_predicted.png",
        "config.json",
        "execution.log",
        "linear_regression.joblib",
        "metrics.json",
        "predictions.csv",
        "split_manifest.json",
        "standard_scaler.joblib",
    }
    assert {path.name for path in output.iterdir()} == expected
    manifest = json.loads((output / "split_manifest.json").read_text())
    assert manifest["train"]["row_count"] == 24
    assert manifest["test"]["row_count"] == 6
    config = json.loads((output / "config.json").read_text())
    assert config["determinism"]["random_seed"] is None
    assert config["feature_definition"]["past_only"] is True
    assert joblib.load(output / "standard_scaler.joblib").mean_[0] == 12.5
    predictions = pd.read_csv(output / "predictions.csv")
    assert list(predictions["Date"]) == list(dates[-6:].strftime("%Y-%m-%d"))
    assert len(predictions) == 6
