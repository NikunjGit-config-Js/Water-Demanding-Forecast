from __future__ import annotations

import shutil
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Protocol
from urllib.parse import urlparse

from .provenance import sha256_file
from .registry import CitySource
from .status import DataStatus


class DownloadResponse(Protocol):
    def read(self, size: int = -1) -> bytes: ...
    def __enter__(self) -> "DownloadResponse": ...
    def __exit__(self, *args: object) -> None: ...


Opener = Callable[[str], DownloadResponse]
SeleniumFetcher = Callable[[str, Path], Path]


@dataclass(frozen=True)
class AcquisitionResult:
    status: DataStatus
    raw_path: Path | None = None
    source_url: str | None = None
    downloaded_at_utc: str | None = None
    sha256: str | None = None
    method: str | None = None
    reason: str | None = None


def _default_opener(url: str) -> DownloadResponse:
    request = urllib.request.Request(url, headers={"User-Agent": "water-forecast-data/1"})
    return urllib.request.urlopen(request, timeout=60)  # type: ignore[return-value]


def selenium_download(url: str, raw_dir: Path, *, timeout_seconds: float = 60,
                      driver_factory: Callable[[object], object] | None = None,
                      waiter_factory: Callable[[object, float], object] | None = None) -> Path:
    """Download a public URL with locked-down headless Chrome and an explicit wait.

    This function only navigates to the supplied public URL. It does not solve
    CAPTCHAs, authenticate, click consent controls, or evade access restrictions.
    Factories are injectable so tests never require a browser or live network.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    destination = raw_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    try:
        from selenium import webdriver
        from selenium.webdriver.support.ui import WebDriverWait
    except ImportError as exc:
        raise RuntimeError("Selenium dependency is unavailable") from exc

    options = webdriver.ChromeOptions()
    for argument in ("--headless=new", "--disable-gpu", "--no-sandbox",
                     "--disable-dev-shm-usage"):
        options.add_argument(argument)
    options.add_experimental_option("prefs", {
        "download.default_directory": str(destination),
        "download.prompt_for_download": False,
        "download.directory_upgrade": False,
        "safebrowsing.enabled": True,
    })
    create_driver = driver_factory or (lambda configured: webdriver.Chrome(options=configured))
    create_waiter = waiter_factory or WebDriverWait
    before = {path.resolve() for path in destination.iterdir() if path.is_file()}
    driver = None
    try:
        driver = create_driver(options)
        driver.get(url)  # type: ignore[attr-defined]

        def completed_download(_: object) -> Path | bool:
            candidates = [path.resolve() for path in destination.iterdir()
                          if path.is_file() and path.resolve() not in before
                          and not path.name.endswith((".crdownload", ".part", ".tmp"))]
            return candidates[0] if len(candidates) == 1 else False

        result = create_waiter(driver, timeout_seconds).until(completed_download)  # type: ignore[attr-defined]
        downloaded = Path(result).resolve()
        if destination not in downloaded.parents or not downloaded.is_file():
            raise RuntimeError("browser download escaped the configured raw directory")
        return downloaded
    except Exception as exc:
        raise RuntimeError(f"browser download failed: {type(exc).__name__}") from exc
    finally:
        if driver is not None:
            driver.quit()  # type: ignore[attr-defined]


def acquire(source: CitySource, raw_dir: Path, *, opener: Opener = _default_opener,
            selenium_fetcher: SeleniumFetcher | None = None,
            allow_selenium: bool = False) -> AcquisitionResult:
    if not source.configured or not source.source_url:
        return AcquisitionResult(DataStatus.DATA_SOURCE_REQUIRED, reason="source is not configured")
    raw_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(urlparse(source.source_url).path).suffix or f".{source.format or 'bin'}"
    target = raw_dir / f"source{suffix}"
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with opener(source.source_url) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        temporary.replace(target)
        method = "official_api" if source.adapter_type in {"api", "ckan", "ogd"} else "direct_download"
    except Exception as direct_error:
        temporary.unlink(missing_ok=True)
        fallback = selenium_fetcher or (selenium_download if allow_selenium else None)
        if fallback is None:
            return AcquisitionResult(DataStatus.ACQUISITION_FAILED, source_url=source.source_url,
                                     reason=f"download failed: {type(direct_error).__name__}")
        try:
            fetched = fallback(source.source_url, raw_dir)
            if not fetched.is_file() or raw_dir.resolve() not in fetched.resolve().parents:
                raise ValueError("Selenium result must be a file inside the raw directory")
            target = fetched
            method = "selenium_fallback"
        except Exception as selenium_error:
            return AcquisitionResult(DataStatus.ACQUISITION_FAILED, source_url=source.source_url,
                                     reason=f"Selenium fallback failed: {type(selenium_error).__name__}")
    return AcquisitionResult(DataStatus.READY, target, source.source_url,
                             datetime.now(UTC).isoformat(), sha256_file(target), method)
