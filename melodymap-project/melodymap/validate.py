from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class SchemaSpec:
    required_any: tuple[tuple[str, ...], ...]
    min_rows: int = 1


SCHEMAS: dict[str, SchemaSpec] = {
    "venues": SchemaSpec(
        required_any=(("venue_name",), ("space_type",), ("lat", "latitude"), ("lon", "longitude")),
        min_rows=10,
    ),
    "sensors": SchemaSpec(
        required_any=(("location_id", "sensor_id"), ("latitude", "lat"), ("longitude", "lon")),
        min_rows=5,
    ),
    "pedestrians": SchemaSpec(
        required_any=(("location_id", "sensor_id"), ("sensing_date", "date"), ("hourday", "hour"),
                      ("pedestriancount", "pedestrian_count", "total_of_directions")),
        min_rows=100,
    ),
    "cafes": SchemaSpec(
        required_any=(("census_year",), ("number_of_seats",), ("latitude", "lat"), ("longitude", "lon")),
        min_rows=10,
    ),
    "bars": SchemaSpec(
        required_any=(("census_year",), ("number_of_patrons",), ("latitude", "lat"), ("longitude", "lon")),
        min_rows=10,
    ),
}


def snake(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def normalized_columns(df: pd.DataFrame) -> list[str]:
    return [snake(c) for c in df.columns]


def validate_frame(name: str, df: pd.DataFrame) -> dict:
    if name not in SCHEMAS:
        raise ValueError(f"Unknown dataset name: {name}")
    spec = SCHEMAS[name]
    cols = set(normalized_columns(df))
    missing_groups = []
    for alternatives in spec.required_any:
        if not any(alt in cols for alt in alternatives):
            missing_groups.append(alternatives)
    if missing_groups:
        raise ValueError(f"{name}: missing required column groups: {missing_groups}; columns={sorted(cols)}")
    if len(df) < spec.min_rows:
        raise ValueError(f"{name}: expected at least {spec.min_rows} rows, found {len(df)}")
    return {
        "dataset": name,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "required_columns_ok": True,
        "duplicate_rows": int(df.duplicated().sum()),
    }


def validate_csv(name: str, path: Path, nrows: int | None = None) -> dict:
    df = pd.read_csv(path, nrows=nrows)
    report = validate_frame(name, df)
    report["path"] = str(path)
    report["bytes"] = int(path.stat().st_size)
    return report
