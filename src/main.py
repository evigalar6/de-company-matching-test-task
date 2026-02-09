import json
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
    """Write JSON to disk, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _clean_key_series(s: pd.Series) -> list[str]:
    """Return a sorted list of non-empty keys, excluding separator-only junk."""
    values: list[str] = []
    for v in s.astype(str):
        vv = v.strip()
        if not vv:
            continue
        if vv.replace("|", "").strip() == "":
            continue
        values.append(vv)
    return sorted(set(values))


def _clean_display_text(value: object) -> str:
    """Trim and collapse whitespace for readable output fields."""
    if value is None:
        return ""
    s = str(value)
    return " ".join(s.split()).strip()


def main() -> None:
    """Run the pipeline and write merged output + metrics."""
    paths = Paths()

    # Load raw datasets.
    df1_raw = read_csv(paths.ds1)
    df2_raw = read_csv(paths.ds2)

    # Normalize datasets (schema + cleaned fields + block_key + location keys).
    df1 = normalize_dataset(df1_raw, DS1_COLS)
    df2 = normalize_dataset(df2_raw, DS2_COLS)

    # Address-level matching.
    matches = match_datasets(
        df1,
        df2,
        name_threshold_strong=float(NAME_THRESHOLD_STRONG),
        name_threshold_with_postal=float(NAME_THRESHOLD_WITH_POSTAL),
    )

    # Build per-company strict and loose location lists.
    loc1_strict = (
        df1.groupby("customer_id")["location_key"]
        .apply(_clean_key_series)
        .rename("locations_ds1")
    )
    loc2_strict = (
        df2.groupby("customer_id")["location_key"]
        .apply(_clean_key_series)
        .rename("locations_ds2_strict")
    )

    loc1_loose = (
        df1.groupby("customer_id")["location_key_loose"]
        .apply(_clean_key_series)
        .rename("locations_ds1_loose")
    )
    loc2_loose = (
        df2.groupby("customer_id")["location_key_loose"]
        .apply(_clean_key_series)
        .rename("locations_ds2_loose")
    )

    # Representative company names (first non-empty), cleaned for readability.
    name1 = (
        df1.groupby("customer_id")["customer_name"]
        .apply(lambda s: _clean_display_text(s.dropna().iloc[0]) if len(s.dropna()) else "")
        .rename("company_name_ds1")
    )
    name2 = (
        df2.groupby("customer_id")["customer_name"]
        .apply(lambda s: _clean_display_text(s.dropna().iloc[0]) if len(s.dropna()) else "")
        .rename("company_name_ds2")
    )

    # Company-level mapping based on address-level matches.
    m_companies = matches.drop_duplicates(subset=["ds1_customer_id", "ds2_customer_id"])
    ds2_list = (
        m_companies.groupby("ds1_customer_id")["ds2_customer_id"]
        .apply(lambda s: sorted(set(s.dropna().astype(str))))
        .rename("matched_company_ids_ds2")
    )

    # Assemble final merged output (one row per DS1 company).
    merged = (
        pd.DataFrame({"company_id_ds1": name1.index.astype(str)})
        .merge(
            name1.reset_index().rename(columns={"customer_id": "company_id_ds1"}),
            on="company_id_ds1",
            how="left",
        )
        .merge(
            loc1_strict.reset_index().rename(columns={"customer_id": "company_id_ds1"}),
            on="company_id_ds1",
            how="left",
        )
        .merge(
            loc1_loose.reset_index().rename(columns={"customer_id": "company_id_ds1"}),
            on="company_id_ds1",
            how="left",
        )
        .merge(
            ds2_list.reset_index().rename(columns={"ds1_customer_id": "company_id_ds1"}),
            on="company_id_ds1",
            how="left",
        )
    )

    def ds2_names(ids: object) -> list[str]:
        """Collect representative DS2 company names for a list of DS2 company ids."""
        if not isinstance(ids, list):
            return []
        out: list[str] = []
        for cid in ids:
            nm = _clean_display_text(name2.get(cid, ""))
            if nm:
                out.append(nm)
        return sorted(set(out))

    def ds2_locations_strict(ids: object) -> list[str]:
        """Collect strict DS2 location keys for a list of DS2 company ids."""
        if not isinstance(ids, list):
            return []
        out: list[str] = []
        for cid in ids:
            out.extend(loc2_strict.get(cid, []))
        return sorted(set(out))

    def ds2_locations_loose(ids: object) -> list[str]:
        """Collect loose DS2 location keys for a list of DS2 company ids."""
        if not isinstance(ids, list):
            return []
        out: list[str] = []
        for cid in ids:
            out.extend(loc2_loose.get(cid, []))
        return sorted(set(out))

    merged["matched_company_names_ds2"] = merged["matched_company_ids_ds2"].apply(ds2_names)
    merged["locations_ds2"] = merged["matched_company_ids_ds2"].apply(ds2_locations_strict)
    merged["locations_ds2_loose"] = merged["matched_company_ids_ds2"].apply(ds2_locations_loose)

    # Human-friendly overlap: ignore street differences (city|state|postal|country).
    merged["overlapping_locations"] = merged.apply(
        lambda r: sorted(set(r["locations_ds1_loose"]) & set(r["locations_ds2_loose"]))
        if isinstance(r.get("locations_ds1_loose"), list) and isinstance(r.get("locations_ds2_loose"), list)
        else [],
        axis=1,
    )

    # Strict overlap retained as an extra column for transparency/debugging.
    merged["overlapping_locations_strict"] = merged.apply(
        lambda r: sorted(set(r["locations_ds1"]) & set(r["locations_ds2"]))
        if isinstance(r.get("locations_ds1"), list) and isinstance(r.get("locations_ds2"), list)
        else [],
        axis=1,
    )

    # Ensure consistent list-like columns and serialize them as JSON strings for CSV output.
    list_cols = [
        "locations_ds1",
        "locations_ds1_loose",
        "matched_company_ids_ds2",
        "matched_company_names_ds2",
        "locations_ds2",
        "locations_ds2_loose",
        "overlapping_locations",
        "overlapping_locations_strict",
    ]

    for c in list_cols:
        merged[c] = merged[c].apply(lambda v: v if isinstance(v, list) else [])

    for c in list_cols:
        merged[c] = merged[c].apply(lambda v: json.dumps(v, ensure_ascii=False))

    # Strict requirement: leave the overlap cell empty when there is no overlap.
    merged["overlapping_locations"] = merged["overlapping_locations"].apply(lambda s: "" if s == "[]" else s)

    # Write deliverables.
    write_csv(merged, paths.out_merged)

    metrics = compute_metrics(df1, df2, matches)
    write_json(metrics, paths.out_metrics)

    print("Merged rows (DS1 companies):", len(merged))
    print("Address-level matches:", len(matches))
    print("Merged CSV written to:", paths.out_merged)
    print("Metrics JSON written to:", paths.out_metrics)


if __name__ == "__main__":
    main()
