# Case-study data provenance

This bundled case-study excerpt is intentionally small. It lets the repository reproduce the real-data figures even when the runtime has no network access.

## Live-music venues

The three venue records are verbatim field values returned by the Victorian Government DataVic CKAN API for the City of Melbourne **Live Music Venues** resource. The source dataset contains 227 venue records and is licensed CC BY.

- Dataset page: https://data.melbourne.vic.gov.au/explore/dataset/live-music-venues/
- DataVic dataset: https://discover.data.vic.gov.au/dataset/live-music-venues
- CKAN resource queried during project preparation: `f3240735-7bda-52ef-a5d1-2ac49fda1fe6`

## Pedestrian sensors

The three sensor coordinates are verbatim field values returned by the DataVic CKAN API for **Pedestrian Counting System - Sensor Locations**.

- Dataset page: https://data.melbourne.vic.gov.au/explore/dataset/pedestrian-counting-system-sensor-locations/
- DataVic resource queried during project preparation: `677b5ce2-6352-473a-be2c-cbc350af3f7d`

## Reference pedestrian volumes

The long-run mean hourly volumes and observation counts are reproduced as summary statistics from published analyses using the same City of Melbourne pedestrian sensor system. They are included only to provide a real magnitude reference for the three selected sensors. They are not treated as 2024 nightly averages and are not combined with venue data to estimate a causal effect.

The repository's online workflow can replace this fixed excerpt with current City of Melbourne hourly counts when internet access is available.
