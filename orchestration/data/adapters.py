from __future__ import annotations

import json
import zipfile
from pathlib import Path, PurePosixPath

import pandas as pd

from .registry import CitySource


def _read_zip(path: Path, source: CitySource) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if any(PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts for name in names):
            raise ValueError("unsafe ZIP member path")
        member = source.zip_member
        if member is None:
            candidates = [name for name in names if Path(name).suffix.lower() in {".csv", ".json", ".xls", ".xlsx"}]
            if len(candidates) != 1:
                raise ValueError("ZIP must contain one supported data file or declare zip_member")
            member = candidates[0]
        if member not in names:
            raise ValueError("configured ZIP member is absent")
        with archive.open(member) as stream:
            suffix = Path(member).suffix.lower()
            if suffix == ".csv": return pd.read_csv(stream)
            if suffix == ".json": return pd.DataFrame(json.load(stream))
            return pd.read_excel(stream)


def read_source(path: Path, source: CitySource) -> pd.DataFrame:
    kind = (source.format or path.suffix.lstrip(".")).lower()
    if kind == "csv": return pd.read_csv(path)
    if kind == "json": return pd.read_json(path)
    if kind in {"xls", "xlsx", "excel"}: return pd.read_excel(path)
    if kind == "zip": return _read_zip(path, source)
    raise ValueError(f"unsupported source format: {kind}")
