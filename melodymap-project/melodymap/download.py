from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from .config import DATASETS, LEGACY_API, MELBOURNE_API, RAW

RESOURCE_IDS = {
    "venues": "f36bd25f-d2e2-4f5d-aeb3-ac49221de7f7",
    "sensors": "677b5ce2-6352-473a-be2c-cbc350af3f7d",
    "cafes": "cc7750fc-3c57-43c3-a5f5-c2af85db4c74",
    "bars": "c5a0f0ef-520c-44c1-975c-4ff6dcce7274",
}
CKAN_BASE = "https://discover.data.vic.gov.au/api/3/action/datastore_search"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _stream_get(url: str, params: dict, target: Path, timeout: int = 90) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with requests.get(url, params=params, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        ctype = (r.headers.get("content-type") or "").lower()
        if "text/html" in ctype:
            raise RuntimeError(f"Expected a data file, received HTML from {r.url}")
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        final_url = r.url
    if tmp.stat().st_size < 20:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is unexpectedly small: {final_url}")
    tmp.replace(target)
    # Make sure a CSV can at least be parsed and has a header.
    probe = pd.read_csv(target, nrows=3)
    if probe.shape[1] < 2:
        raise RuntimeError(f"CSV parse check failed for {target}")
    return {
        "url": final_url,
        "bytes": target.stat().st_size,
        "sha256": _sha256(target),
        "columns": list(probe.columns),
    }


def _download_ckan_resource(resource_id: str, target: Path, page_size: int = 5000) -> dict:
    """Fallback downloader for DataVic mirrors whose datastore is complete."""
    rows: list[dict] = []
    offset = 0
    total = None
    while total is None or offset < total:
        params = {"resource_id": resource_id, "limit": page_size, "offset": offset}
        r = requests.get(CKAN_BASE, params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()
        if not payload.get("success"):
            raise RuntimeError(f"DataVic datastore error for {resource_id}")
        result = payload["result"]
        total = int(result["total"])
        batch = result["records"]
        rows.extend(batch)
        if not batch:
            break
        offset += len(batch)
    if not rows:
        raise RuntimeError(f"No records returned for DataVic resource {resource_id}")
    df = pd.DataFrame(rows)
    if "_id" in df.columns:
        df = df.drop(columns="_id")
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)
    return {
        "url": f"{CKAN_BASE}?resource_id={resource_id}",
        "bytes": target.stat().st_size,
        "sha256": _sha256(target),
        "columns": list(df.columns),
        "rows": len(df),
        "fallback": "DataVic CKAN datastore",
    }


def download_small_dataset(name: str, force: bool = False) -> tuple[Path, dict]:
    if name not in ("venues", "sensors", "cafes", "bars"):
        raise ValueError(f"Unsupported small dataset: {name}")
    target = RAW / f"{name}.csv"
    if target.exists() and not force:
        return target, {"skipped": True, "bytes": target.stat().st_size, "sha256": _sha256(target)}
    dataset_id = DATASETS[name]
    url = f"{MELBOURNE_API}/{dataset_id}/exports/csv"
    params = {
        "lang": "en",
        "timezone": "Australia/Melbourne",
        "use_labels": "false",
        "delimiter": ",",
    }
    try:
        meta = _stream_get(url, params, target)
    except Exception as first_error:
        # DataVic mirrors are useful if the primary portal is temporarily unavailable.
        resource_id = RESOURCE_IDS[name]
        try:
            meta = _download_ckan_resource(resource_id, target)
            meta["primary_error"] = repr(first_error)
        except Exception:
            # Retain the legacy export URL as a final compatibility attempt.
            legacy = f"{LEGACY_API}/{dataset_id}/exports/csv"
            meta = _stream_get(legacy, {"delimiter": ","}, target)
            meta["primary_error"] = repr(first_error)
    return target, meta


def download_pedestrian_year(year: int, months: Iterable[int] = range(1, 13), force: bool = False) -> list[tuple[Path, dict]]:
    dataset_id = DATASETS["pedestrians"]
    url = f"{MELBOURNE_API}/{dataset_id}/exports/csv"
    results = []
    for month in months:
        target = RAW / f"pedestrians_{year}_{month:02d}.csv"
        if target.exists() and not force:
            results.append((target, {"skipped": True, "bytes": target.stat().st_size, "sha256": _sha256(target)}))
            continue
        params = {
            "lang": "en",
            # City of Melbourne supports this monthly facet refinement. It keeps each file manageable.
            "refine": f'sensing_date:"{year}/{month:02d}"',
            "timezone": "Australia/Melbourne",
            "use_labels": "false",
            "delimiter": ",",
        }
        meta = _stream_get(url, params, target, timeout=180)
        results.append((target, meta))
    return results


def download_all(year: int = 2024, force: bool = False) -> dict:
    manifest: dict[str, object] = {"year": year, "datasets": {}}
    for name in ("venues", "sensors", "cafes", "bars"):
        path, meta = download_small_dataset(name, force=force)
        manifest["datasets"][name] = {"path": str(path), **meta}
    ped = download_pedestrian_year(year, force=force)
    manifest["datasets"]["pedestrians"] = [
        {"path": str(path), **meta} for path, meta in ped
    ]
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
