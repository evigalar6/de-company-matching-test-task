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


def main() -> None:
    """Run the pipeline and write merged output + metrics."""
    paths = Paths()

    # Load raw datasets.
    df1_raw = read_csv(paths.ds1)
    df2_raw = read_csv(paths.ds2)

    # Normalize datasets (schema + cleaned fields + block_key + location_key).
    df1 = normalize_dataset(df1_raw, DS1_COLS)
    df2 = normalize_dataset(df2_raw, DS2_COLS)

    # Address-level matching.
    matches = match_datasets(
        df1,
        df2,
        name_threshold_strong=float(NAME_THRESHOLD_STRONG),
        name_threshold_with_postal=float(NAME_THRESHOLD_WITH_POSTAL),
    )

    # Build per-company location lists (unique sets).
    loc1 = (
        df1.groupby("customer_id")["location_key"]
        .apply(lambda s: sorted(set(s.dropna().astype(str))))
        .rename("locations_ds1")
    )
    loc2 = (
        df2.groupby("customer_id")["location_key"]
        .apply(lambda s: sorted(set(s.dropna().astype(str))))
        .rename("locations_ds2")
    )

    # Representative company names (first non-empty), cleaned for readability.
    name1 = (
        df1.groupby("customer_id")["customer_name"]
        .apply(lambda s: s.dropna().astype(str).map(str.strip).iloc[0] if len(s.dropna()) else "")
        .rename("company_name_ds1")
    )
    name2 = (
        df2.groupby("customer_id")["customer_name"]
        .apply(lambda s: s.dropna().astype(str).map(str.strip).iloc[0] if len(s.dropna()) else "")
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
            loc1.reset_index().rename(columns={"customer_id": "company_id_ds1"}),
            on="company_id_ds1",
            how="left",
        )
        .merge(
            ds2_list.reset_index().rename(columns={"ds1_customer_id": "company_id_ds1"}),
            on="company_id_ds1",
            how="left",
        )
    )

    def ds2_locations(ids: object) -> list[str]:
        """Collect unique DS2 location keys for a list of DS2 company ids."""
        if not isinstance(ids, list):
            return []
        out: list[str] = []
        for cid in ids:
            out.extend(loc2.get(cid, []))
        return sorted(set(out))

    def ds2_names(ids: object) -> list[str]:
        """Collect representative DS2 company names for a list of DS2 company ids."""
        if not isinstance(ids, list):
            return []
        out: list[str] = []
        for cid in ids:
            nm = str(name2.get(cid, "") or "").strip()
            if nm:
                out.append(nm)
        return sorted(set(out))

    merged["matched_company_names_ds2"] = merged["matched_company_ids_ds2"].apply(ds2_names)
    merged["locations_ds2"] = merged["matched_company_ids_ds2"].apply(ds2_locations)

    merged["overlapping_locations"] = merged.apply(
        lambda r: sorted(set(r["locations_ds1"]) & set(r["locations_ds2"]))
        if isinstance(r.get("locations_ds1"), list) and isinstance(r.get("locations_ds2"), list)
        else [],
        axis=1,
    )

    # Ensure consistent list-like columns and serialize them as JSON strings for CSV output.
    list_cols = [
        "locations_ds1",
        "matched_company_ids_ds2",
        "matched_company_names_ds2",
        "locations_ds2",
        "overlapping_locations",
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
