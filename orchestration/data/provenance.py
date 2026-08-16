from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ProvenanceManifest:
    city: str
    source_name: str
    source_url: str | None
    official: bool | None
    acquisition_method: str
    downloaded_at_utc: str | None
    raw_format: str
    raw_path: str
    raw_sha256: str
    canonical_path: str
    canonical_sha256: str
    source_unit: str | None
    canonical_unit: str | None
    adapter_type: str
    expected_frequency: str
    transformations: tuple[str, ...] = field(default_factory=tuple)
    wards_aggregated: bool = False
    row_count: int = 0
    date_start: str = ""
    date_end: str = ""
    schema_version: int = 1

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def read(cls, path: Path) -> "ProvenanceManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload.get("transformations"), list):
            payload["transformations"] = tuple(payload["transformations"])
        return cls(**payload)
