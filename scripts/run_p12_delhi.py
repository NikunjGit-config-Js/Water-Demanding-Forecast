"""Delhi Phase 12-compatible validation (adapted for non-London dates)."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.delhi_pipeline import ARTIFACT_ROOT, DATASET, REPORT_ROOT, write_checkpoint


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_latest(subdir: Path) -> Path:
    if not subdir.exists():
        return subdir
    parts = sorted(d for d in subdir.iterdir() if d.is_dir())
    return parts[-1] if parts else subdir


def load_metrics(path: Path) -> dict:
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        return df.to_dict(orient="records")[0] if len(df) > 0 else {}
    return json.loads(path.read_text())


def main():
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("delhi_validation")

    log.info("=" * 60)
    log.info("PHASE 12: Delhi Validation Audit")
    log.info("=" * 60)

    source = pd.read_csv(DATASET)
    source["Date"] = pd.to_datetime(source["Date"])
    dataset_sha = sha256_file(DATASET)

    log.info(f"Dataset SHA256: {dataset_sha}")
    log.info(f"Rows: {len(source)}")
    log.info(f"Date range: {source['Date'].iloc[0].date()} to {source['Date'].iloc[-1].date()}")
    log.info(f"Consumption range: {source['Consumption'].min():.2f} to {source['Consumption'].max():.2f} MGD")

    results = {}
    errors = {}

    # Phase 5
    p5 = find_latest(ARTIFACT_ROOT / "phase5")
    try:
        preds = pd.read_csv(p5 / "predictions.csv")
        metrics = load_metrics(p5 / "metrics.csv")
        assert len(preds) > 0, "No predictions"
        assert "actual" in preds, "Missing 'actual' column"
        assert preds["Date"].is_monotonic_increasing, "Dates not chronological"
        results["phase5"] = {"status": "PASS", "rows": len(preds), "metrics": metrics}
    except Exception as e:
        results["phase5"] = {"status": "FAIL", "error": str(e)}
        errors["phase5"] = str(e)

    # Phase 6
    p6 = find_latest(ARTIFACT_ROOT / "phase6")
    try:
        preds = pd.read_csv(p6 / "predictions.csv")
        metrics = load_metrics(p6 / "metrics_summary.csv")
        assert len(preds) > 0, "No predictions"
        assert "fold" in preds, "Missing fold column"
        results["phase6"] = {"status": "PASS", "rows": len(preds), "folds": int(preds["fold"].nunique()), "metrics": metrics}
    except Exception as e:
        results["phase6"] = {"status": "FAIL", "error": str(e)}
        errors["phase6"] = str(e)

    # Phase 7
    p7 = find_latest(ARTIFACT_ROOT / "phase7")
    try:
        preds = pd.read_csv(p7 / "locked_test_predictions.csv")
        metrics = load_metrics(p7 / "locked_test_metrics.csv")
        assert len(preds) > 0, "No predictions"
        results["phase7"] = {"status": "PASS", "rows": len(preds), "metrics": metrics}
    except Exception as e:
        results["phase7"] = {"status": "FAIL", "error": str(e)}
        errors["phase7"] = str(e)

    # Phase 8
    p8 = find_latest(ARTIFACT_ROOT / "phase8")
    try:
        preds = pd.read_csv(p8 / "predictions.csv")
        metrics = load_metrics(p8 / "metrics_summary.csv")
        assert len(preds) > 0, "No predictions"
        results["phase8"] = {"status": "PASS", "rows": len(preds), "folds": int(preds["fold"].nunique()), "metrics": metrics}
    except Exception as e:
        results["phase8"] = {"status": "FAIL", "error": str(e)}
        errors["phase8"] = str(e)

    # Phase 9
    p9 = find_latest(ARTIFACT_ROOT / "phase9")
    try:
        preds = pd.read_csv(p9 / "predictions.csv")
        metrics = load_metrics(p9 / "metrics.csv")
        report = json.loads((p9 / "forecast_report.json").read_text())
        assert len(preds) > 0, "No predictions"
        assert preds["Date"].is_monotonic_increasing, "Dates not chronological"
        assert report["dataset_sha256"] == dataset_sha, "Dataset hash mismatch"
        results["phase9"] = {"status": "PASS", "rows": len(preds), "models": len(report.get("model_names", [])), "metrics": metrics}
    except Exception as e:
        results["phase9"] = {"status": "FAIL", "error": str(e)}
        errors["phase9"] = str(e)

    # Phase 10
    p10 = find_latest(ARTIFACT_ROOT / "phase10")
    try:
        preds = pd.read_csv(p10 / "predictions.csv")
        metrics = load_metrics(p10 / "metrics.csv")
        report = json.loads((p10 / "forecast_report.json").read_text())
        assert len(preds) > 0, "No predictions"
        results["phase10"] = {"status": "PASS", "rows": len(preds), "metrics": metrics}
    except Exception as e:
        results["phase10"] = {"status": "FAIL", "error": str(e)}
        errors["phase10"] = str(e)

    # Summary
    log.info("\n" + "=" * 60)
    log.info("VALIDATION SUMMARY")
    log.info("=" * 60)
    for phase_name, result in results.items():
        status = result["status"]
        detail = result.get("metrics", result.get("error", ""))
        log.info(f"  {phase_name}: {status} - {detail}")

    report = {
        "city": "delhi",
        "dataset": str(DATASET),
        "dataset_sha256": dataset_sha,
        "total_rows": len(source),
        "date_range": [str(source["Date"].iloc[0].date()), str(source["Date"].iloc[0].date())],
        "phase_results": results,
        "errors": errors,
        "overall": "PASS" if not errors else "FAIL",
    }

    out_dir = REPORT_ROOT / "phase12_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "delhi_validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    write_checkpoint(12, f"Delhi validation audit complete. Overall: {report['overall']}")
    log.info(f"\nPhase 12 artifacts: {out_dir}")
    return report


if __name__ == "__main__":
    main()
