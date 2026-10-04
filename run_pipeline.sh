#!/usr/bin/env bash
# Rebuild everything from data/raw. ~3 minutes (validation runs 400 synthetic worlds).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
$PY src/build_geo.py        # boundaries -> data/processed/districts.geojson
$PY src/digitise.py         # paper figures -> data/raw/digitised_*.csv
$PY src/build_dataset.py    # -> data/processed/district_year.csv
$PY src/model.py            # -> model_output.csv, model_summary.json, forecast.csv
$PY src/validate.py         # -> validation.json, backtest_detail.csv
