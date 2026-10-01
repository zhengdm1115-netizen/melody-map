from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .analysis import build_sensor_context, build_venue_features
from .geo import nearest


def sensor_coverage(venues: pd.DataFrame, sensors: pd.DataFrame, thresholds=(250, 500, 750)) -> dict[str, float]:
    d = nearest(venues, sensors, "location_id")["distance_m"].to_numpy(dtype=float)
    result = {f"venue_sensor_coverage_{t}m": float(np.mean(d <= t)) for t in thresholds}
    result["median_nearest_sensor_distance_m"] = float(np.nanmedian(d))
    return result


def association_test(sensor_context: pd.DataFrame, permutations: int = 499, seed: int = 42) -> dict[str, float]:
    x = sensor_context[["music_venues_within_radius", "avg_night_footfall"]].dropna()
    if len(x) < 5 or x["music_venues_within_radius"].nunique() < 2:
        return {"spearman_music_density_vs_night_footfall": float("nan"), "permutation_p_value": float("nan"), "n_sensors": int(len(x))}
    rho = float(spearmanr(x["music_venues_within_radius"], x["avg_night_footfall"]).statistic)
    rng = np.random.default_rng(seed)
    observed = abs(rho)
    hits = 0
    y = x["avg_night_footfall"].to_numpy()
    xv = x["music_venues_within_radius"].to_numpy()
    for _ in range(permutations):
        rp = spearmanr(xv, rng.permutation(y)).statistic
        if abs(float(rp)) >= observed:
            hits += 1
    p = (hits + 1) / (permutations + 1)
    return {
        "spearman_music_density_vs_night_footfall": rho,
        "permutation_p_value": float(p),
        "n_sensors": int(len(x)),
    }


def _rank_series(vf: pd.DataFrame) -> pd.Series:
    return vf.set_index("venue_id")["consumption_context_score"].rank(method="average", ascending=False)


def sensitivity_analysis(venues, sensors_metrics, cafes, bars, radii=(300, 500, 700)) -> tuple[pd.DataFrame, dict]:
    frames = {}
    for r in radii:
        frames[r] = build_venue_features(venues, sensors_metrics, cafes, bars, radius_m=r)
    base_r = 500 if 500 in frames else radii[len(radii)//2]
    base_rank = _rank_series(frames[base_r])
    base_top = set(base_rank.nsmallest(min(10, len(base_rank))).index)
    rows = []
    summary = {}
    for r, vf in frames.items():
        rank = _rank_series(vf)
        common = base_rank.index.intersection(rank.index)
        corr = float(spearmanr(base_rank.loc[common], rank.loc[common]).statistic) if len(common) >= 3 else float("nan")
        top = set(rank.nsmallest(min(10, len(rank))).index)
        overlap = len(base_top & top) / max(1, len(base_top | top))
        rows.append({"radius_m": r, "rank_spearman_vs_500m": corr, "top10_jaccard_vs_500m": overlap})
        summary[f"rank_spearman_{r}m_vs_{base_r}m"] = corr
        summary[f"top10_jaccard_{r}m_vs_{base_r}m"] = float(overlap)
    return pd.DataFrame(rows), summary


def evaluate(venues, sensors, sensors_metrics, cafes, bars, sensor_context) -> tuple[dict, pd.DataFrame]:
    metrics = {}
    metrics.update(sensor_coverage(venues, sensors))
    metrics.update(association_test(sensor_context))
    sensitivity_df, sensitivity_metrics = sensitivity_analysis(venues, sensors_metrics, cafes, bars)
    metrics.update(sensitivity_metrics)
    return metrics, sensitivity_df


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        clean[k] = v
    path.write_text(json.dumps(clean, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
