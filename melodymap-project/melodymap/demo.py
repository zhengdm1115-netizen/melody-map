from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEMO


def _jitter(rng, center_lat, center_lon, n, scale_lat=0.004, scale_lon=0.005):
    lat = rng.normal(center_lat, scale_lat, n)
    lon = rng.normal(center_lon, scale_lon, n)
    return lat, lon


def generate_demo_data(out_dir: Path = DEMO, seed: int = 42, year: int = 2024) -> dict[str, Path]:
    """Generate schema-compatible synthetic data for smoke tests and UI demos.

    The synthetic data are intentionally designed to contain a modest positive
    association between music-place density, evening footfall, and nearby
    hospitality capacity. They are not intended to reproduce Melbourne values.
    """
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    centers = [
        (-37.8136, 144.9631, "Central"),
        (-37.8010, 144.9665, "North"),
        (-37.8225, 144.9555, "Southwest"),
        (-37.8120, 144.9470, "Docklands"),
    ]

    venue_rows = []
    space_types = ["Bar", "Nightclub", "Theatre", "Performance space", "Hotel"]
    idx = 1
    for c, (lat0, lon0, label) in enumerate(centers):
        n = [34, 24, 18, 14][c]
        lats, lons = _jitter(rng, lat0, lon0, n, 0.0035, 0.0045)
        for lat, lon in zip(lats, lons):
            st = rng.choice(space_types, p=[0.40, 0.12, 0.16, 0.20, 0.12])
            venue_rows.append({
                "property_number": f"SYN{idx:04d}",
                "venue_name": f"Synthetic {label} Venue {idx}",
                "venue_address": f"{idx} Demo Street, Melbourne",
                "space_type": st,
                "website": "https://example.invalid/demo",
                "lat": lat,
                "lon": lon,
                "location": f"POINT ({lon} {lat})",
                "geolocation": f"{lat}, {lon}",
            })
            idx += 1
    venues = pd.DataFrame(venue_rows)
    venues.to_csv(out_dir / "venues.csv", index=False)

    # Sensors: concentrate them around the same urban cores and add a few peripheral sensors.
    sensor_rows = []
    sid = 1
    for c, (lat0, lon0, label) in enumerate(centers):
        n = [8, 6, 5, 4][c]
        lats, lons = _jitter(rng, lat0, lon0, n, 0.0045, 0.0055)
        for lat, lon in zip(lats, lons):
            sensor_rows.append({
                "location_id": str(sid),
                "sensor_description": f"Synthetic pedestrian sensor {sid}",
                "sensor_name": f"SYN_SENSOR_{sid:02d}",
                "installation_date": "2019-01-01",
                "note": "Synthetic demo record",
                "location_type": "Outdoor",
                "status": "A",
                "direction_1": "North",
                "direction_2": "South",
                "latitude": lat,
                "longitude": lon,
                "location": f"POINT ({lon} {lat})",
                "cluster_hint": c,
            })
            sid += 1
    for _ in range(5):
        lat = rng.uniform(-37.845, -37.785)
        lon = rng.uniform(144.925, 144.995)
        sensor_rows.append({
            "location_id": str(sid),
            "sensor_description": f"Synthetic peripheral sensor {sid}",
            "sensor_name": f"SYN_SENSOR_{sid:02d}",
            "installation_date": "2019-01-01",
            "note": "Synthetic demo record",
            "location_type": "Outdoor",
            "status": "A",
            "direction_1": "North",
            "direction_2": "South",
            "latitude": lat,
            "longitude": lon,
            "location": f"POINT ({lon} {lat})",
            "cluster_hint": -1,
        })
        sid += 1
    sensors = pd.DataFrame(sensor_rows)
    sensors.drop(columns=["cluster_hint"]).to_csv(out_dir / "sensors.csv", index=False)

    # Hospitality points with capacity; more capacity around the central music clusters.
    cafe_rows, bar_rows = [], []
    rid = 1
    for c, (lat0, lon0, label) in enumerate(centers):
        nc = [80, 48, 36, 28][c]
        nb = [36, 26, 18, 16][c]
        lats, lons = _jitter(rng, lat0, lon0, nc, 0.006, 0.007)
        for lat, lon in zip(lats, lons):
            cafe_rows.append({
                "census_year": str(year),
                "block_id": str(rng.integers(1, 607)),
                "property_id": f"CP{rid}",
                "base_property_id": f"CB{rid}",
                "building_address": f"{rid} Demo Lane",
                "clue_small_area": label,
                "trading_name": f"Synthetic Cafe {rid}",
                "business_address": f"{rid} Demo Lane, Melbourne",
                "industry_anzsic4_code": "4511",
                "industry_anzsic4_description": "Cafes and Restaurants",
                "seating_type": rng.choice(["Seats - Indoor", "Seats - Outdoor"]),
                "number_of_seats": int(rng.integers(15, 120)),
                "longitude": lon,
                "latitude": lat,
                "location": f"POINT ({lon} {lat})",
            })
            rid += 1
        lats, lons = _jitter(rng, lat0, lon0, nb, 0.006, 0.007)
        for lat, lon in zip(lats, lons):
            bar_rows.append({
                "census_year": str(year),
                "block_id": str(rng.integers(1, 607)),
                "property_id": f"BP{rid}",
                "base_property_id": f"BB{rid}",
                "building_address": f"{rid} Demo Road",
                "clue_small_area": label,
                "trading_name": f"Synthetic Bar {rid}",
                "business_address": f"{rid} Demo Road, Melbourne",
                "number_of_patrons": int(rng.integers(40, 350)),
                "longitude": lon,
                "latitude": lat,
                "location": f"POINT ({lon} {lat})",
            })
            rid += 1
    pd.DataFrame(cafe_rows).to_csv(out_dir / "cafes.csv", index=False)
    pd.DataFrame(bar_rows).to_csv(out_dir / "bars.csv", index=False)

    # Full-year hourly pedestrian data for a compact number of sensors.
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    ped_rows = []
    id_counter = 1
    cluster_strength = {0: 1.35, 1: 1.20, 2: 1.10, 3: 1.00, -1: 0.72}
    # Reconstruct the hidden hint using nearest generated center for realistic profiles.
    for _, s in sensors.iterrows():
        distances = [((s.latitude-lat0)**2 + (s.longitude-lon0)**2, c) for c,(lat0,lon0,_) in enumerate(centers)]
        dmin, c = min(distances)
        if dmin > 0.00045:
            c = -1
        strength = cluster_strength[c]
        for date in dates:
            weekend = date.dayofweek >= 5
            for hour in range(24):
                morning = np.exp(-((hour-8.5)/2.2)**2)
                lunch = 0.65*np.exp(-((hour-13.0)/2.5)**2)
                evening = 1.15*np.exp(-((hour-19.5)/3.1)**2)
                night_music = 0.75*np.exp(-((hour-22.0)/2.5)**2) * strength
                base = 65 + 105*morning + 70*lunch + 115*evening + 120*night_music
                if weekend:
                    base *= (0.78 + 0.42*np.exp(-((hour-20.5)/4.0)**2))
                base *= strength
                count = max(0, int(rng.normal(base, 0.13*max(base, 20))))
                d1 = int(count * rng.uniform(0.43, 0.57))
                ped_rows.append({
                    "id": id_counter,
                    "location_id": str(s.location_id),
                    "sensing_date": date.strftime("%Y-%m-%d"),
                    "hourday": hour,
                    "direction_1": d1,
                    "direction_2": count - d1,
                    "pedestriancount": count,
                    "sensor_name": s.sensor_name,
                    "location": s.location,
                })
                id_counter += 1
    peds = pd.DataFrame(ped_rows)
    peds.to_csv(out_dir / f"pedestrians_{year}.csv", index=False)

    return {
        "venues": out_dir / "venues.csv",
        "sensors": out_dir / "sensors.csv",
        "cafes": out_dir / "cafes.csv",
        "bars": out_dir / "bars.csv",
        "pedestrians": out_dir / f"pedestrians_{year}.csv",
    }
