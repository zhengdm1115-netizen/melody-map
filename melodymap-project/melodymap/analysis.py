from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DEFAULT_RADIUS_M, NIGHT_HOURS
from .geo import cluster_points, counts_and_sums_within, nearest


def percentile(s: pd.Series) -> pd.Series:
    if len(s) == 0:
        return s.astype(float)
    return s.rank(pct=True, method="average").fillna(0.0)


def build_sensor_metrics(peds: pd.DataFrame, sensors: pd.DataFrame) -> pd.DataFrame:
    p = peds.copy()
    p["is_night"] = p["hourday"].isin(NIGHT_HOURS)
    # Average hourly footfall. This keeps sensors comparable even if some dates are missing.
    overall = p.groupby("location_id")["pedestriancount"].mean().rename("avg_hourly_footfall")
    night = p[p["is_night"]].groupby("location_id")["pedestriancount"].mean().rename("avg_night_footfall")
    wknd = p[p["is_night"] & p["is_weekend"]].groupby("location_id")["pedestriancount"].mean().rename("avg_weekend_night_footfall")
    wkdy = p[p["is_night"] & ~p["is_weekend"]].groupby("location_id")["pedestriancount"].mean().rename("avg_weekday_night_footfall")
    nobs = p.groupby("location_id").size().rename("n_hourly_observations")
    out = pd.concat([overall, night, wknd, wkdy, nobs], axis=1).reset_index()
    out = sensors.merge(out, on="location_id", how="left")
    return out


def hourly_profile(peds: pd.DataFrame) -> pd.DataFrame:
    return (
        peds.groupby(["hourday", "is_weekend"], as_index=False)["pedestriancount"]
        .mean()
        .rename(columns={"pedestriancount": "mean_footfall"})
    )


def build_venue_features(
    venues: pd.DataFrame,
    sensors_metrics: pd.DataFrame,
    cafes: pd.DataFrame,
    bars: pd.DataFrame,
    radius_m: int = DEFAULT_RADIUS_M,
) -> pd.DataFrame:
    out = venues.copy().reset_index(drop=True)

    vcount, _ = counts_and_sums_within(out, venues, radius_m)
    out["music_venues_within_radius"] = np.maximum(vcount - 1, 0)  # exclude the focal venue

    cafe_count, cafe_seats = counts_and_sums_within(out, cafes, radius_m, "number_of_seats")
    bar_count, bar_patrons = counts_and_sums_within(out, bars, radius_m, "number_of_patrons")
    out["cafes_within_radius"] = cafe_count
    out["cafe_seats_within_radius"] = cafe_seats
    out["bars_within_radius"] = bar_count
    out["bar_capacity_within_radius"] = bar_patrons

    near = nearest(out, sensors_metrics, "location_id")
    out["nearest_sensor_id"] = near["location_id"].astype(str)
    out["nearest_sensor_distance_m"] = near["distance_m"]
    sensor_cols = [
        "location_id", "avg_night_footfall", "avg_weekend_night_footfall",
        "avg_weekday_night_footfall", "n_hourly_observations",
    ]
    out = out.merge(sensors_metrics[sensor_cols], left_on="nearest_sensor_id", right_on="location_id", how="left", suffixes=("", "_sensor"))
    out = out.drop(columns=["location_id_sensor"], errors="ignore")
    uncovered = out["nearest_sensor_distance_m"] > radius_m
    for col in ["avg_night_footfall", "avg_weekend_night_footfall", "avg_weekday_night_footfall"]:
        out.loc[uncovered, col] = np.nan

    out["music_density_pct"] = percentile(out["music_venues_within_radius"])
    out["night_footfall_pct"] = percentile(out["avg_night_footfall"])
    out["cafe_capacity_pct"] = percentile(out["cafe_seats_within_radius"])
    out["bar_capacity_pct"] = percentile(out["bar_capacity_within_radius"])
    out["hospitality_capacity_pct"] = 0.5 * out["cafe_capacity_pct"] + 0.5 * out["bar_capacity_pct"]
    out["consumption_context_score"] = (
        0.35 * out["music_density_pct"]
        + 0.35 * out["night_footfall_pct"]
        + 0.30 * out["hospitality_capacity_pct"]
    ) * 100

    out["cluster_id"] = cluster_points(out, eps_m=450, min_samples=3)
    return out


def build_sensor_context(
    sensors_metrics: pd.DataFrame,
    venues: pd.DataFrame,
    cafes: pd.DataFrame,
    bars: pd.DataFrame,
    radius_m: int = DEFAULT_RADIUS_M,
) -> pd.DataFrame:
    out = sensors_metrics.copy().reset_index(drop=True)
    venue_count, _ = counts_and_sums_within(out, venues, radius_m)
    cafe_count, cafe_seats = counts_and_sums_within(out, cafes, radius_m, "number_of_seats")
    bar_count, bar_capacity = counts_and_sums_within(out, bars, radius_m, "number_of_patrons")
    out["music_venues_within_radius"] = venue_count
    out["cafes_within_radius"] = cafe_count
    out["cafe_seats_within_radius"] = cafe_seats
    out["bars_within_radius"] = bar_count
    out["bar_capacity_within_radius"] = bar_capacity
    return out


def district_summary(venue_features: pd.DataFrame) -> pd.DataFrame:
    x = venue_features[venue_features["cluster_id"] >= 0].copy()
    if x.empty:
        return pd.DataFrame(columns=["cluster_id", "venue_count"])
    g = x.groupby("cluster_id")
    out = g.agg(
        venue_count=("venue_id", "size"),
        venue_type_count=("space_type", "nunique"),
        mean_context_score=("consumption_context_score", "mean"),
        median_night_footfall=("avg_night_footfall", "median"),
        cafe_seats_median=("cafe_seats_within_radius", "median"),
        bar_capacity_median=("bar_capacity_within_radius", "median"),
        centroid_latitude=("latitude", "mean"),
        centroid_longitude=("longitude", "mean"),
    ).reset_index()
    out["district_label"] = out["cluster_id"].map(lambda z: f"Music district {int(z)+1}")
    return out.sort_values(["venue_count", "mean_context_score"], ascending=False).reset_index(drop=True)
