from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType


CITY_SLUG_RE = re.compile(r"^[a-z][a-z0-9-]*$")


def validate_city_slug(city: str) -> str:
    if not isinstance(city, str) or not CITY_SLUG_RE.fullmatch(city):
        raise ValueError(f"invalid city slug: {city!r}")
    return city


@dataclass(frozen=True)
class CitySource:
    city: str
    configured: bool
    source_name: str | None = None
    source_url: str | None = None
    official: bool | None = None
    format: str | None = None
    expected_frequency: str | None = None
    date_column: str | None = None
    consumption_column: str | None = None
    unit: str | None = None
    canonical_unit: str | None = None
    unit_multiplier: float = 1.0
    aggregation: str | None = None
    adapter_type: str | None = None
    zone_column: str | None = None
    zip_member: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        validate_city_slug(self.city)
        if self.configured and not self.source_name:
            raise ValueError("configured sources require a source_name")
        if self.unit_multiplier <= 0:
            raise ValueError("unit_multiplier must be positive")


_REGISTRY = MappingProxyType({
    "london": CitySource(
        city="london", configured=True, source_name="Validated bundled London canonical dataset",
        format="csv", expected_frequency="daily", date_column="Date",
        consumption_column="Consumption", adapter_type="existing_canonical",
        notes="Existing validated artifact; external source URL and unit are not evidenced here.",
    ),
    "bengaluru": CitySource(city="bengaluru", configured=False, notes="Verified compatible source required."),
    "delhi": CitySource(
        city="delhi", configured=True,
        source_name="Delhi Jal Board - Daily Water Production Report Archive",
        source_url="https://delhijalboard.delhi.gov.in/daily-water-production-report",
        official=True, format="pdf", expected_frequency="daily",
        date_column="Date", consumption_column="Consumption",
        unit="MGD", canonical_unit="MGD",
        adapter_type="djb_archive",
        notes=(
            "Official DJB daily water production reports. Multi-document paginated "
            "archive. Source-specific adapter required. Values represent verified "
            "Delhi municipal daily water production/supply proxy in MGD."
        ),
    ),
    "gurgaon": CitySource(city="gurgaon", configured=False, notes="Verified compatible source required."),
    "hyderabad": CitySource(city="hyderabad", configured=False, notes="Verified compatible source required."),
    "pune": CitySource(city="pune", configured=False, notes="Verified compatible source required."),
})


def supported_cities() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def get_city_source(city: str) -> CitySource:
    validate_city_slug(city)
    try:
        return _REGISTRY[city]
    except KeyError as exc:
        raise KeyError(f"unknown city: {city}") from exc
