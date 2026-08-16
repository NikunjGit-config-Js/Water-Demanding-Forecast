from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_CITIES = frozenset(
    {"london", "bengaluru", "delhi", "gurgaon", "hyderabad", "pune"}
)


def validate_city_slug(city: str) -> str:
    """Return a canonical registered city slug, rejecting path-like input."""
    if not isinstance(city, str) or city != city.strip().lower():
        raise ValueError("City must be a lowercase canonical slug.")
    if city not in SUPPORTED_CITIES:
        raise ValueError(
            f"Unsupported city {city!r}; choose from {sorted(SUPPORTED_CITIES)}."
        )
    return city


@dataclass(frozen=True, slots=True)
class RunContext:
    city: str
    dataset_path: Path
    artifact_root: Path
    report_root: Path
    state_root: Path
    checkpoint_root: Path
    legacy_london: bool

    @classmethod
    def for_city(
        cls, city: str = "london", *, project_root: Path = PROJECT_ROOT
    ) -> "RunContext":
        slug = validate_city_slug(city)
        if slug == "london":
            state = project_root / "orchestration" / "state"
            return cls(
                city=slug,
                dataset_path=project_root / "data/preprocessed/all/preprocessed_data.csv",
                artifact_root=project_root / "artifacts",
                report_root=project_root / "reports",
                state_root=state,
                checkpoint_root=state / "checkpoints",
                legacy_london=True,
            )
        state = project_root / "orchestration" / "state" / "cities" / slug
        return cls(
            city=slug,
            dataset_path=project_root / "data" / "cities" / slug / "canonical" / "water_demand.csv",
            artifact_root=project_root / "artifacts" / "cities" / slug,
            report_root=project_root / "reports" / "cities" / slug,
            state_root=state,
            checkpoint_root=state / "checkpoints",
            legacy_london=False,
        )

    def phase_artifact_root(self, number: int) -> Path:
        return self.artifact_root / f"phase{number}"

    def phase_report_root(self, number: int) -> Path:
        return self.report_root / f"phase{number}"

    def phase_dependencies(self, number: int) -> tuple[Path, ...]:
        if number == 3:
            phases = (2,)
        elif number in {4, 5, 6, 7, 8}:
            phases = (2, 3)
        elif number == 11:
            phases = tuple(range(5, 11))
        elif number in {12, 13}:
            phases = tuple(range(number))
        else:
            phases = ()
        return tuple(self.phase_artifact_root(item) for item in phases)

    def prompt_block(self, phase_number: int | None = None) -> str:
        dependencies = (
            ", ".join(str(path) for path in self.phase_dependencies(phase_number))
            if phase_number is not None
            else ""
        )
        return (
            "RUN CONTEXT (authoritative; never fall back to another city's files):\n"
            f"city: {self.city}\n"
            f"dataset_path: {self.dataset_path}\n"
            f"artifact_root: {self.artifact_root}\n"
            f"report_root: {self.report_root}\n"
            f"checkpoint_root: {self.checkpoint_root}\n"
            f"phase_dependency_locations: {dependencies or 'none'}"
        )
