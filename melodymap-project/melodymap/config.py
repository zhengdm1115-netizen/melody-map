from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
DEMO = DATA / "demo"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"

MELBOURNE_API = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets"
LEGACY_API = "https://data.melbourne.vic.gov.au/api/v2/catalog/datasets"

DATASETS = {
    "venues": "live-music-venues",
    "sensors": "pedestrian-counting-system-sensor-locations",
    "pedestrians": "pedestrian-counting-system-monthly-counts-per-hour",
    "cafes": "cafes-and-restaurants-with-seating-capacity",
    "bars": "bars-and-pubs-with-patron-capacity",
}

DEFAULT_YEAR = 2024
DEFAULT_RADIUS_M = 500
NIGHT_HOURS = tuple(range(18, 24))
