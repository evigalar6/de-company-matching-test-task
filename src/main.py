import json
import sys
from pathlib import Path

import pandas as pd

from src.config import (
    Paths,
    DS1_COLS,
    DS2_COLS,
    NAME_THRESHOLD_STRONG,
    NAME_THRESHOLD_WITH_POSTAL,
)
from src.io_utils import read_csv, write_csv
from src.matching import match_datasets
from src.metrics import compute_metrics
from src.normalize import normalize_dataset


def write_json(data: dict, path: Path) -> None:
    """Write JSON to disk, creating parent directories if needed.

    Args:
        data: JSON-serializable dictionary to write.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _clean_key_series(series: pd.Series) -> list[str]:
    """Return a sorted list of non-empty keys, excluding separator-only junk.

    Args:
        series: Series of key strings.

    Returns:
        Sorted list of unique key strings. Empty strings and values like '||||' are removed.
    """
    values: list[str] = []
    for value in series.astype(str):
        cleaned_value = value.strip()
        if not cleaned_value:
            continue
        if cleaned_value.replace("|", "").strip() == "":
            continue
        values.append(cleaned_value)
    return sorted(set(values))


def _clean_display_text(value: object) -> str:
    """Trim and collapse whitespace for readable output fields.

    Args:
        value: Any object convertible to string.

    Returns:
        Readable string with collapsed internal whitespace.
    """
    if value is None:
        return ""
    text = str(value)
    return " ".join(text.split()).strip()


def main() -> None:
    """Run the pipeline and write merged output + metrics.

    Raises:
        SystemExit: For common input/format errors with a readable message.
    """
    try:
        paths = Paths()

        # Load raw datasets.
        ds1_raw_df = read_csv(paths.ds1)
        ds2_raw_df = read_csv(paths.ds2)

        # Normalize datasets (schema + cleaned fields + block_key + location keys).
        ds1_df = normalize_dataset(ds1_raw_df, DS1_COLS)
        ds2_df = normalize_dataset(ds2_raw_df, DS2_COLS)

        # Address-level matching.
        matches_df = match_datasets(
            ds1_df,
            ds2_df,
            name_threshold_strong=float(NAME_THRESHOLD_STRONG),
            name_threshold_with_postal=float(NAME_THRESHOLD_WITH_POSTAL),
        )

        # Build per-company strict and loose location lists.
        ds1_locations_strict = (
            ds1_df.groupby("customer_id")["location_key"].apply(_clean_key_series).rename("locations_ds1")
        )
        ds2_locations_strict = (
            ds2_df.groupby("customer_id")["location_key"].apply(_clean_key_series).rename("locations_ds2")
        )

        ds1_locations_loose = (
            ds1_df.groupby("customer_id")["location_key_loose"]
            .apply(_clean_key_series)
            .rename("locations_ds1_loose")
        )
        ds2_locations_loose = (
            ds2_df.groupby("customer_id")["location_key_loose"]
            .apply(_clean_key_series)
            .rename("locations_ds2_loose")
        )

        # Representative company names (first non-empty), cleaned for readability.
        ds1_company_names = (
            ds1_df.groupby("customer_id")["customer_name"]
            .apply(lambda series: _clean_display_text(series.dropna().iloc[0]) if len(series.dropna()) else "")
            .rename("company_name_ds1")
        )
        ds2_company_names = (
            ds2_df.groupby("customer_id")["customer_name"]
            .apply(lambda series: _clean_display_text(series.dropna().iloc[0]) if len(series.dropna()) else "")
            .rename("company_name_ds2")
        )

        # Company-level mapping based on address-level matches.
        company_matches = matches_df.drop_duplicates(subset=["ds1_customer_id", "ds2_customer_id"])
        ds2_company_ids_by_ds1 = (
            company_matches.groupby("ds1_customer_id")["ds2_customer_id"]
            .apply(lambda series: sorted(set(series.dropna().astype(str))))
            .rename("matched_company_ids_ds2")
        )

        # Assemble final merged output (one row per DS1 company).
        merged_df = (
            pd.DataFrame({"company_id_ds1": ds1_company_names.index.astype(str)})
            .merge(
                ds1_company_names.reset_index().rename(columns={"customer_id": "company_id_ds1"}),
                on="company_id_ds1",
                how="left",
            )
            .merge(
                ds1_locations_strict.reset_index().rename(columns={"customer_id": "company_id_ds1"}),
                on="company_id_ds1",
                how="left",
            )
            .merge(
                ds1_locations_loose.reset_index().rename(columns={"customer_id": "company_id_ds1"}),
                on="company_id_ds1",
                how="left",
            )
            .merge(
                ds2_company_ids_by_ds1.reset_index().rename(columns={"ds1_customer_id": "company_id_ds1"}),
                on="company_id_ds1",
                how="left",
            )
        )

        def collect_ds2_company_names(ds2_company_ids: object) -> list[str]:
            """Collect representative Dataset 2 names for a list of Dataset 2 company ids."""
            if not isinstance(ds2_company_ids, list):
                return []
            names: list[str] = []
            for company_id in ds2_company_ids:
                name = _clean_display_text(ds2_company_names.get(company_id, ""))
                if name:
                    names.append(name)
            return sorted(set(names))

        def collect_ds2_locations_strict(ds2_company_ids: object) -> list[str]:
            """Collect strict Dataset 2 location keys for a list of Dataset 2 company ids."""
            if not isinstance(ds2_company_ids, list):
                return []
            locations: list[str] = []
            for company_id in ds2_company_ids:
                locations.extend(ds2_locations_strict.get(company_id, []))
            return sorted(set(locations))

        def collect_ds2_locations_loose(ds2_company_ids: object) -> list[str]:
            """Collect loose Dataset 2 location keys for a list of Dataset 2 company ids."""
            if not isinstance(ds2_company_ids, list):
                return []
            locations: list[str] = []
            for company_id in ds2_company_ids:
                locations.extend(ds2_locations_loose.get(company_id, []))
            return sorted(set(locations))

        merged_df["matched_company_names_ds2"] = merged_df["matched_company_ids_ds2"].apply(
            collect_ds2_company_names
        )
        merged_df["locations_ds2"] = merged_df["matched_company_ids_ds2"].apply(collect_ds2_locations_strict)
        merged_df["locations_ds2_loose"] = merged_df["matched_company_ids_ds2"].apply(collect_ds2_locations_loose)

        # Strict overlap: full address key (includes street) — matches the task wording.
        merged_df["overlapping_locations"] = merged_df.apply(
            lambda row: sorted(set(row["locations_ds1"]) & set(row["locations_ds2"]))
            if isinstance(row.get("locations_ds1"), list) and isinstance(row.get("locations_ds2"), list)
            else [],
            axis=1,
        )

        # Loose overlap: ignores street differences (city|state|postal|country) — helpful for analysis.
        merged_df["overlapping_locations_loose"] = merged_df.apply(
            lambda row: sorted(set(row["locations_ds1_loose"]) & set(row["locations_ds2_loose"]))
            if isinstance(row.get("locations_ds1_loose"), list)
            and isinstance(row.get("locations_ds2_loose"), list)
            else [],
            axis=1,
        )

        # Ensure consistent list-like columns and serialize them as JSON strings for CSV output.
        list_column_names = [
            "locations_ds1",
            "locations_ds1_loose",
            "matched_company_ids_ds2",
            "matched_company_names_ds2",
            "locations_ds2",
            "locations_ds2_loose",
            "overlapping_locations",
            "overlapping_locations_loose",
        ]

        for col_name in list_column_names:
            merged_df[col_name] = merged_df[col_name].apply(lambda value: value if isinstance(value, list) else [])

        for col_name in list_column_names:
            merged_df[col_name] = merged_df[col_name].apply(lambda value: json.dumps(value, ensure_ascii=False))

        # Strict requirement: leave the strict overlap cell empty when there is no overlap.
        merged_df["overlapping_locations"] = merged_df["overlapping_locations"].apply(
            lambda value: "" if value == "[]" else value
        )

        # Write deliverables.
        write_csv(merged_df, paths.out_merged)

        metrics = compute_metrics(ds1_df, ds2_df, matches_df)
        write_json(metrics, paths.out_metrics)

        print("Merged rows (DS1 companies):", len(merged_df))
        print("Address-level matches:", len(matches_df))
        print("Merged CSV written to:", paths.out_merged)
        print("Metrics JSON written to:", paths.out_metrics)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
