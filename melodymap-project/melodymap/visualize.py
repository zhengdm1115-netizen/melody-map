from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_venues_and_sensors(venues: pd.DataFrame, sensors: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(venues["longitude"], venues["latitude"], s=22, alpha=0.7, label="Live music venues")
    ax.scatter(sensors["longitude"], sensors["latitude"], s=45, marker="x", label="Pedestrian sensors")
    ax.set_title("Music venues and pedestrian sensors")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    _save(fig, path)


def plot_hourly_profile(profile: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for weekend, label in [(False, "Weekday"), (True, "Weekend")]:
        x = profile[profile["is_weekend"] == weekend].sort_values("hourday")
        ax.plot(x["hourday"], x["mean_footfall"], marker="o", label=label)
    ax.axvspan(18, 23, alpha=0.08)
    ax.set_title("Average pedestrian activity by hour")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean hourly pedestrian count")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    _save(fig, path)


def plot_top_scores(venue_features: pd.DataFrame, path: Path, n: int = 15):
    x = venue_features.nlargest(min(n, len(venue_features)), "consumption_context_score").copy()
    x = x.sort_values("consumption_context_score")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(x["venue_name"], x["consumption_context_score"])
    ax.set_title("Top music places by consumption-context score")
    ax.set_xlabel("Descriptive score (0–100; not spending)")
    _save(fig, path)


def plot_density_vs_footfall(sensor_context: pd.DataFrame, path: Path):
    x = sensor_context[["music_venues_within_radius", "avg_night_footfall"]].dropna()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(x["music_venues_within_radius"], x["avg_night_footfall"], alpha=0.75)
    if len(x) >= 2 and x["music_venues_within_radius"].nunique() >= 2:
        coef = np.polyfit(x["music_venues_within_radius"], x["avg_night_footfall"], 1)
        xx = np.linspace(x["music_venues_within_radius"].min(), x["music_venues_within_radius"].max(), 100)
        ax.plot(xx, coef[0] * xx + coef[1])
    ax.set_title("Music-place density and nighttime pedestrian activity")
    ax.set_xlabel("Music venues within analysis radius")
    ax.set_ylabel("Average hourly footfall, 18:00–23:00")
    _save(fig, path)


def plot_district_summary(districts: pd.DataFrame, path: Path):
    if districts.empty:
        return
    x = districts.sort_values("venue_count")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(x["district_label"], x["venue_count"])
    ax.set_title("Detected music districts")
    ax.set_xlabel("Number of live music venues")
    _save(fig, path)


def make_all_figures(venues, sensors, profile, venue_features, sensor_context, districts, figure_dir: Path):
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_venues_and_sensors(venues, sensors, figure_dir / "venues_and_sensors.png")
    plot_hourly_profile(profile, figure_dir / "hourly_profile.png")
    plot_top_scores(venue_features, figure_dir / "top_consumption_context_scores.png")
    plot_density_vs_footfall(sensor_context, figure_dir / "music_density_vs_night_footfall.png")
    plot_district_summary(districts, figure_dir / "music_districts.png")
