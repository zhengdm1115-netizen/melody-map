from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .geo import haversine_matrix


def _xy_to_latlon(x_m: np.ndarray, y_m: np.ndarray, lat0=-37.814, lon0=144.964):
    lat = lat0 + y_m / 111_320.0
    lon = lon0 + x_m / (111_320.0 * np.cos(np.radians(lat0)))
    return lat, lon


def generate_city(seed: int = 0, n_venues: int = 90, n_sensors: int = 32,
                  effect: float = 0.30, confounding: float = 0.35,
                  noise_sd: float = 0.25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a compact synthetic city with known latent mechanisms.

    `effect` controls the direct relationship from local music density to footfall.
    `confounding` controls a centrality factor that independently raises both venue
    density and pedestrian activity. The output retains the latent ground truth so
    recovery can be evaluated directly.
    """
    rng = np.random.default_rng(seed)

    # Venue positions: central mixture plus two secondary clusters.
    centers = np.array([[0, 0], [850, 450], [-750, -500]])
    probs = np.array([0.60, 0.22, 0.18])
    cid = rng.choice(len(centers), size=n_venues, p=probs)
    vxy = centers[cid] + rng.normal(0, [430, 360], size=(n_venues, 2))
    vlat, vlon = _xy_to_latlon(vxy[:, 0], vxy[:, 1])
    venues = pd.DataFrame({'venue_id': np.arange(n_venues), 'x_m': vxy[:,0], 'y_m': vxy[:,1], 'latitude': vlat, 'longitude': vlon})

    # Sensor positions spread across the same footprint.
    sxy = rng.normal(0, [900, 750], size=(n_sensors, 2))
    slat, slon = _xy_to_latlon(sxy[:, 0], sxy[:, 1])
    sensors = pd.DataFrame({'sensor_id': np.arange(n_sensors), 'x_m': sxy[:,0], 'y_m': sxy[:,1], 'latitude': slat, 'longitude': slon})

    # Local venue density around sensors, scaled to [0,1].
    d = haversine_matrix(sensors.latitude, sensors.longitude, venues.latitude, venues.longitude)
    density = (d <= 500).sum(axis=1).astype(float)
    density_z = (density - density.mean()) / (density.std() + 1e-9)

    # Centrality is observed distance from CBD center, but acts as a confounder.
    centrality = np.exp(-np.sqrt(sensors.x_m**2 + sensors.y_m**2) / 900.0)
    centrality_z = (centrality - centrality.mean()) / (centrality.std() + 1e-9)

    eps = rng.normal(0, noise_sd, size=n_sensors)
    latent = effect * density_z + confounding * centrality_z + eps
    sensors['music_density_500m'] = density
    sensors['centrality'] = centrality
    sensors['true_music_effect'] = effect
    sensors['footfall_index'] = latent
    return venues, sensors


def recover_association(venues: pd.DataFrame, sensors: pd.DataFrame, radius_m: float = 500,
                        coord_noise_m: float = 0.0, sensor_keep: float = 1.0,
                        seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    s = sensors.copy()
    v = venues.copy()

    if sensor_keep < 1:
        keep_n = max(5, int(round(len(s) * sensor_keep)))
        s = s.iloc[rng.choice(len(s), size=keep_n, replace=False)].reset_index(drop=True)

    if coord_noise_m > 0:
        for df in (s, v):
            dx = rng.normal(0, coord_noise_m, size=len(df))
            dy = rng.normal(0, coord_noise_m, size=len(df))
            lat, lon = _xy_to_latlon(df['x_m'].to_numpy()+dx, df['y_m'].to_numpy()+dy)
            df['latitude'] = lat; df['longitude'] = lon

    d = haversine_matrix(s.latitude, s.longitude, v.latitude, v.longitude)
    density = (d <= radius_m).sum(axis=1).astype(float)
    if np.unique(density).size < 2 or len(s) < 5:
        rho = np.nan
    else:
        rho = float(spearmanr(density, s['footfall_index']).statistic)
    return {'rho': rho, 'n_sensors': int(len(s)), 'radius_m': radius_m, 'coord_noise_m': coord_noise_m, 'sensor_keep': sensor_keep}


def run_simulation_lab(replicates: int = 30, seed: int = 2026) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(seed)

    # A: sensor coverage / missing sensors.
    for keep in [1.0, 0.8, 0.6, 0.4]:
        for r in range(replicates):
            s0 = int(rng.integers(0, 2**31-1)); venues, sensors = generate_city(seed=s0)
            out = recover_association(venues, sensors, sensor_keep=keep, seed=s0+1)
            rows.append({'experiment':'coverage','level':keep, 'replicate':r, **out})

    # B: analysis radius.
    for radius in [250, 350, 500, 700, 1000]:
        for r in range(replicates):
            s0 = int(rng.integers(0, 2**31-1)); venues, sensors = generate_city(seed=s0)
            out = recover_association(venues, sensors, radius_m=radius, seed=s0+1)
            rows.append({'experiment':'radius','level':radius, 'replicate':r, **out})

    # C: coordinate noise.
    for noise in [0, 50, 100, 200]:
        for r in range(replicates):
            s0 = int(rng.integers(0, 2**31-1)); venues, sensors = generate_city(seed=s0)
            out = recover_association(venues, sensors, coord_noise_m=noise, seed=s0+1)
            rows.append({'experiment':'coordinate_noise','level':noise, 'replicate':r, **out})

    # D: confounding with true music effect fixed at zero.
    for conf in [0.0, 0.2, 0.4, 0.6]:
        for r in range(replicates):
            s0 = int(rng.integers(0, 2**31-1)); venues, sensors = generate_city(seed=s0, effect=0.0, confounding=conf)
            out = recover_association(venues, sensors, seed=s0+1)
            rows.append({'experiment':'confounding','level':conf, 'replicate':r, 'true_effect':0.0, **out})

    # E: known effect size while centrality confounding remains moderate.
    for effect in [0.0, 0.1, 0.3, 0.5]:
        for r in range(replicates):
            s0 = int(rng.integers(0, 2**31-1)); venues, sensors = generate_city(seed=s0, effect=effect, confounding=0.25)
            out = recover_association(venues, sensors, seed=s0+1)
            rows.append({'experiment':'effect_size','level':effect, 'replicate':r, 'true_effect':effect, **out})

    return pd.DataFrame(rows)


def summarize_simulations(results: pd.DataFrame) -> pd.DataFrame:
    return (results.groupby(['experiment','level'], as_index=False)
            .agg(mean_rho=('rho','mean'), median_rho=('rho','median'), sd_rho=('rho','std'),
                 q10_rho=('rho', lambda x: x.quantile(.1)), q90_rho=('rho', lambda x: x.quantile(.9)),
                 n=('rho','count')))
