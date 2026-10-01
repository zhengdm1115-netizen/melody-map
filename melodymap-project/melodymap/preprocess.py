from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_YEAR, DEMO, RAW
from .validate import snake, validate_frame


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [snake(c) for c in out.columns]
    return out


def _rename_first(df: pd.DataFrame, canonical: str, aliases: tuple[str, ...]) -> pd.DataFrame:
    if canonical in df.columns:
        return df
    for a in aliases:
        if a in df.columns:
            return df.rename(columns={a: canonical})
    return df


def _numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _clean_coords(df: pd.DataFrame, lat="latitude", lon="longitude") -> pd.DataFrame:
    df = _numeric(df, [lat, lon])
    return df[df[lat].between(-90, 90) & df[lon].between(-180, 180)].copy()


def read_raw(mode: str = "demo", year: int = DEFAULT_YEAR) -> dict[str, pd.DataFrame]:
    if mode not in {"demo", "real"}:
        raise ValueError("mode must be 'demo' or 'real'")
    base = DEMO if mode == "demo" else RAW
    datasets = {}
    for name in ("venues", "sensors", "cafes", "bars"):
        path = base / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run the relevant data script first.")
        datasets[name] = pd.read_csv(path, low_memory=False)

    if mode == "demo":
        p = base / f"pedestrians_{year}.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run scripts/generate_demo.py first.")
        datasets["pedestrians"] = pd.read_csv(p, low_memory=False)
    else:
        parts = sorted(base.glob(f"pedestrians_{year}_*.csv"))
        if not parts:
            raise FileNotFoundError(f"No monthly pedestrian files for {year} in {base}")
        datasets["pedestrians"] = pd.concat(
            (pd.read_csv(p, low_memory=False) for p in parts), ignore_index=True
        )
    return datasets


def preprocess(datasets: dict[str, pd.DataFrame], year: int = DEFAULT_YEAR) -> dict[str, pd.DataFrame]:
    for name, frame in datasets.items():
        validate_frame(name, frame)

    venues = _norm(datasets["venues"])
    venues = _rename_first(venues, "latitude", ("lat",))
    venues = _rename_first(venues, "longitude", ("lon", "lng"))
    venues = _clean_coords(venues)
    venues["venue_name"] = venues["venue_name"].astype(str).str.strip()
    venues["space_type"] = venues.get("space_type", "Unknown").fillna("Unknown").astype(str).str.strip()
    venues = venues.drop_duplicates(subset=["venue_name", "latitude", "longitude"]).reset_index(drop=True)
    venues["venue_id"] = np.arange(1, len(venues) + 1)

    sensors = _norm(datasets["sensors"])
    sensors = _rename_first(sensors, "location_id", ("sensor_id",))
    sensors = _rename_first(sensors, "latitude", ("lat",))
    sensors = _rename_first(sensors, "longitude", ("lon", "lng"))
    sensors = _clean_coords(sensors)
    sensors["location_id"] = sensors["location_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    if "status" in sensors.columns:
        active = sensors["status"].astype(str).str.upper().isin({"A", "ACTIVE", "1", "TRUE"})
        if active.any():
            sensors = sensors[active].copy()
    sensors = sensors.drop_duplicates(subset=["location_id"], keep="last").reset_index(drop=True)

    peds = _norm(datasets["pedestrians"])
    peds = _rename_first(peds, "location_id", ("sensor_id",))
    peds = _rename_first(peds, "sensing_date", ("date",))
    peds = _rename_first(peds, "hourday", ("hour",))
    peds = _rename_first(peds, "pedestriancount", ("pedestrian_count", "total_of_directions"))
    peds["location_id"] = peds["location_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    peds["sensing_date"] = pd.to_datetime(peds["sensing_date"], errors="coerce", dayfirst=False)
    peds = _numeric(peds, ["hourday", "pedestriancount"])
    peds = peds.dropna(subset=["sensing_date", "hourday", "pedestriancount"])
    peds = peds[(peds["hourday"].between(0, 23)) & (peds["pedestriancount"] >= 0)].copy()
    peds = peds[peds["sensing_date"].dt.year == int(year)].copy()
    peds["hourday"] = peds["hourday"].astype(int)
    peds["weekday"] = peds["sensing_date"].dt.dayofweek
    peds["is_weekend"] = peds["weekday"] >= 5

    cafes = _norm(datasets["cafes"])
    cafes = _rename_first(cafes, "latitude", ("lat",))
    cafes = _rename_first(cafes, "longitude", ("lon", "lng"))
    cafes = _clean_coords(cafes)
    cafes = _numeric(cafes, ["census_year", "number_of_seats"])
    cafes = cafes.dropna(subset=["census_year", "number_of_seats"])
    latest_cafe_year = int(cafes.loc[cafes["census_year"] <= year, "census_year"].max()) if (cafes["census_year"] <= year).any() else int(cafes["census_year"].max())
    cafes = cafes[cafes["census_year"] == latest_cafe_year].copy()
    cafes["number_of_seats"] = cafes["number_of_seats"].clip(lower=0)

    bars = _norm(datasets["bars"])
    bars = _rename_first(bars, "latitude", ("lat",))
    bars = _rename_first(bars, "longitude", ("lon", "lng"))
    bars = _clean_coords(bars)
    bars = _numeric(bars, ["census_year", "number_of_patrons"])
    bars = bars.dropna(subset=["census_year", "number_of_patrons"])
    latest_bar_year = int(bars.loc[bars["census_year"] <= year, "census_year"].max()) if (bars["census_year"] <= year).any() else int(bars["census_year"].max())
    bars = bars[bars["census_year"] == latest_bar_year].copy()
    bars["number_of_patrons"] = bars["number_of_patrons"].clip(lower=0)

    return {
        "venues": venues,
        "sensors": sensors,
        "pedestrians": peds,
        "cafes": cafes,
        "bars": bars,
    }
