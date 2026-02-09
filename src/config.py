from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    """Project file paths."""
    ds1: Path = PROJECT_ROOT / "data" / "raw" / "company_dataset_1.csv"
    ds2: Path = PROJECT_ROOT / "data" / "raw" / "company_dataset_2.csv"
    out_merged: Path = PROJECT_ROOT / "output" / "merged_companies.csv"
    out_metrics: Path = PROJECT_ROOT / "output" / "metrics.json"


# Map dataset columns to a unified schema used across the pipeline.
# The unified schema keys are the column names expected by normalize.py.
DS1_COLS = {
    "customer_id": "custnmbr",
    "address_code": "addrcode",
    "customer_name": "custname",
    "street1": "sStreet1",
    "street2": "sStreet2",
    "city": "sCity",
    "state": "sProvState",
    "country": "sCountry",
    "postal": "sPostalZip",
}

DS2_COLS = {
    "customer_id": "custnmbr",
    "address_code": "addrcode",
    "customer_name": "custname",
    "street1": "address1",
    "street2": "address2",
    "street3": "address3",
    "city": "city",
    "state": "state",
    "country": "country",
    "country_code": "ccode",
    "postal": "zip",
}


# Matching thresholds.
NAME_THRESHOLD_STRONG = 95  # Use when relying mostly on name similarity.
NAME_THRESHOLD_WITH_POSTAL = 86  # Allow lower name score when postal codes match.
