from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .geo import haversine_matrix


def load_case_study(case_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    venues = pd.read_csv(case_dir / 'venues.csv')
    sensors = pd.read_csv(case_dir / 'sensors.csv')
    stats = pd.read_csv(case_dir / 'sensor_reference_stats.csv')
    return venues, sensors, stats


def build_distance_table(venues: pd.DataFrame, sensors: pd.DataFrame, radius_m: float = 500.0) -> pd.DataFrame:
    d = haversine_matrix(venues.latitude, venues.longitude, sensors.latitude, sensors.longitude)
    rows = []
    for i, v in venues.reset_index(drop=True).iterrows():
        for j, s in sensors.reset_index(drop=True).iterrows():
            rows.append({
                'venue_name': v['venue_name'],
                'sensor_description': s['sensor_description'],
                'distance_m': float(d[i, j]),
                'within_radius': bool(d[i, j] <= radius_m),
            })
    return pd.DataFrame(rows)


def summarize_case(venues: pd.DataFrame, sensors: pd.DataFrame, stats: pd.DataFrame, radius_m: float = 500.0) -> dict:
    distances = build_distance_table(venues, sensors, radius_m)
    within = distances[distances['within_radius']]
    links_per_venue = within.groupby('venue_name').size().reindex(venues['venue_name'], fill_value=0)
    nearest = distances.loc[distances.groupby('venue_name')['distance_m'].idxmin()].copy()
    return {
        'n_venues': int(len(venues)),
        'n_sensors': int(len(sensors)),
        'radius_m': float(radius_m),
        'n_links_within_radius': int(len(within)),
        'mean_sensors_within_radius_per_venue': float(links_per_venue.mean()),
        'venues_with_multiple_candidate_sensors': int((links_per_venue > 1).sum()),
        'median_nearest_sensor_distance_m': float(nearest['distance_m'].median()),
        'sensor_reference_mean_range': [float(stats['mean_hourly_pedestrians'].min()), float(stats['mean_hourly_pedestrians'].max())],
    }
