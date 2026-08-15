import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.phase10_patchtst import Phase10Config, load_development_prefix, make_one_step_windows


def test_windows_are_strictly_past_only() -> None:
    x, y = make_one_step_windows(np.arange(10), 4)
    np.testing.assert_array_equal(x[0, :, 0], [0, 1, 2, 3])
    assert y[0] == 4
    np.testing.assert_array_equal(x[-1, :, 0], [5, 6, 7, 8])
    assert y[-1] == 9


def test_loader_never_reads_locked_test(monkeypatch, tmp_path: Path) -> None:
    config = Phase10Config(expected_total_rows=20, train_fraction=.7, validation_fraction=.15,
                           context_length=4, patch_length=2, patch_stride=2)
    dataset = tmp_path / "series.csv"
    pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=20),
                  "Consumption": np.arange(20)}).to_csv(dataset, index=False)
    real_read_csv = pd.read_csv
    calls = []

    def guarded(*args, **kwargs):
        calls.append(kwargs.copy())
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded)
    frame = load_development_prefix(dataset, config)
    assert len(frame) == 17
    assert calls == [{"usecols": ["Date", "Consumption"], "nrows": 17}]
    assert frame["Consumption"].max() == 16


def test_split_and_config_are_reproducible() -> None:
    config = Phase10Config()
    assert config.split_rows() == (2660, 3230, 3800)
    assert config.random_seed == 42
    assert config.context_length == 56
    assert config.patch_length == config.patch_stride == 7
