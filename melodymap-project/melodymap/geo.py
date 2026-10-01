from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

EARTH_RADIUS_M = 6_371_008.8


def haversine_matrix(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1 = np.radians(np.asarray(lat1, dtype=float))[:, None]
    lon1 = np.radians(np.asarray(lon1, dtype=float))[:, None]
    lat2 = np.radians(np.asarray(lat2, dtype=float))[None, :]
    lon2 = np.radians(np.asarray(lon2, dtype=float))[None, :]
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    a = np.clip(a, 0, 1)
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def counts_and_sums_within(points: pd.DataFrame, targets: pd.DataFrame, radius_m: float, value_col: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    if len(points) == 0 or len(targets) == 0:
        return np.zeros(len(points), dtype=int), np.zeros(len(points), dtype=float)
    d = haversine_matrix(points.latitude, points.longitude, targets.latitude, targets.longitude)
    mask = d <= radius_m
    counts = mask.sum(axis=1).astype(int)
    if value_col is None:
        sums = counts.astype(float)
    else:
        vals = pd.to_numeric(targets[value_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        sums = (mask * vals[None, :]).sum(axis=1)
    return counts, sums


def nearest(points: pd.DataFrame, targets: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if len(targets) == 0:
        return pd.DataFrame({id_col: [None] * len(points), "distance_m": [np.nan] * len(points)})
    d = haversine_matrix(points.latitude, points.longitude, targets.latitude, targets.longitude)
    idx = d.argmin(axis=1)
    return pd.DataFrame({
        id_col: targets.iloc[idx][id_col].to_numpy(),
        "distance_m": d[np.arange(len(points)), idx],
    })


def cluster_points(points: pd.DataFrame, eps_m: float = 450, min_samples: int = 3) -> np.ndarray:
    if len(points) == 0:
        return np.array([], dtype=int)
    coords = np.radians(points[["latitude", "longitude"]].to_numpy(dtype=float))
    model = DBSCAN(eps=eps_m / EARTH_RADIUS_M, min_samples=min_samples, metric="haversine")
    return model.fit_predict(coords)
