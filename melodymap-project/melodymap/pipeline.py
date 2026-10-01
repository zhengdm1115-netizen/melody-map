from __future__ import annotations

import json
from pathlib import Path

from .analysis import (
    build_sensor_context,
    build_sensor_metrics,
    build_venue_features,
    district_summary,
    hourly_profile,
)
from .config import DEFAULT_RADIUS_M, DEFAULT_YEAR, FIGURES, OUTPUTS, PROCESSED
from .demo import generate_demo_data
from .evaluation import evaluate, save_metrics
from .preprocess import preprocess, read_raw
from .visualize import make_all_figures


def run_pipeline(mode: str = "demo", year: int = DEFAULT_YEAR, radius_m: int = DEFAULT_RADIUS_M) -> dict:
    if mode == "demo":
        generate_demo_data(year=year)

    raw = read_raw(mode=mode, year=year)
    data = preprocess(raw, year=year)

    sensors_metrics = build_sensor_metrics(data["pedestrians"], data["sensors"])
    profile = hourly_profile(data["pedestrians"])
    venue_features = build_venue_features(
        data["venues"], sensors_metrics, data["cafes"], data["bars"], radius_m=radius_m
    )
    sensor_context = build_sensor_context(
        sensors_metrics, data["venues"], data["cafes"], data["bars"], radius_m=radius_m
    )
    districts = district_summary(venue_features)
    metrics, sensitivity = evaluate(
        data["venues"], data["sensors"], sensors_metrics, data["cafes"], data["bars"], sensor_context
    )

    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    venue_features.to_csv(PROCESSED / f"venue_features_{mode}.csv", index=False)
    sensors_metrics.to_csv(PROCESSED / f"sensor_metrics_{mode}.csv", index=False)
    sensor_context.to_csv(PROCESSED / f"sensor_context_{mode}.csv", index=False)
    districts.to_csv(PROCESSED / f"district_summary_{mode}.csv", index=False)
    profile.to_csv(PROCESSED / f"hourly_profile_{mode}.csv", index=False)
    sensitivity.to_csv(PROCESSED / f"sensitivity_{mode}.csv", index=False)
    save_metrics(metrics, OUTPUTS / f"evaluation_metrics_{mode}.json")

    make_all_figures(
        data["venues"], data["sensors"], profile, venue_features, sensor_context, districts, FIGURES / mode
    )

    summary = {
        "mode": mode,
        "year": year,
        "radius_m": radius_m,
        "rows": {k: int(len(v)) for k, v in data.items()},
        "detected_music_districts": int(
            venue_features.loc[venue_features["cluster_id"] >= 0, "cluster_id"].nunique()
        ),
        "metrics": metrics,
    }
    (OUTPUTS / f"run_summary_{mode}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8"
    )
    return summary
