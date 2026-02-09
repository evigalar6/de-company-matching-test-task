import pandas as pd


def compute_metrics(ds1_df: pd.DataFrame, ds2_df: pd.DataFrame, matches_df: pd.DataFrame) -> dict:
    """Compute basic coverage and ambiguity metrics for the match results.

    Args:
        ds1_df: Normalized Dataset 1 records.
        ds2_df: Normalized Dataset 2 records.
        matches_df: Address-level match DataFrame as produced by `match_datasets(...)`.

    Returns:
        Dictionary with summary metrics (coverage, unmatched rate, one-to-many rates).
    """
    ds1_total = int(ds1_df["customer_id"].nunique())
    ds2_total = int(ds2_df["customer_id"].nunique())

    if matches_df.empty:
        total = ds1_total + ds2_total
        return {
            "ds1_companies_total": ds1_total,
            "ds2_companies_total": ds2_total,
            "ds1_matched_companies": 0,
            "ds2_matched_companies": 0,
            "match_rate_ds1": 0.0,
            "match_rate_ds2": 0.0,
            "unmatched_records": (1.0 if total else 0.0),
            "ds1_one_to_many_companies": 0,
            "one_to_many_rate_ds1_total": 0.0,
            "one_to_many_rate_ds1_matched": 0.0,
            "ds2_one_to_many_companies": 0,
            "one_to_many_rate_ds2_total": 0.0,
            "one_to_many_rate_ds2_matched": 0.0,
            "address_level_matches": 0,
        }

    # Company-level mapping based on address-level matches.
    company_matches = matches_df.drop_duplicates(subset=["ds1_customer_id", "ds2_customer_id"])

    ds1_matched = int(company_matches["ds1_customer_id"].nunique())
    ds2_matched = int(company_matches["ds2_customer_id"].nunique())

    match_rate_ds1 = (ds1_matched / ds1_total) if ds1_total else 0.0
    match_rate_ds2 = (ds2_matched / ds2_total) if ds2_total else 0.0

    # One-to-many from DS1 perspective (one DS1 company matched to multiple DS2 companies).
    ds1_one_to_many = int(
        company_matches.groupby("ds1_customer_id")["ds2_customer_id"].nunique().gt(1).sum()
    )
    one_to_many_rate_ds1_matched = (ds1_one_to_many / ds1_matched) if ds1_matched else 0.0
    one_to_many_rate_ds1_total = (ds1_one_to_many / ds1_total) if ds1_total else 0.0

    # One-to-many from DS2 perspective (one DS2 company matched to multiple DS1 companies).
    ds2_one_to_many = int(
        company_matches.groupby("ds2_customer_id")["ds1_customer_id"].nunique().gt(1).sum()
    )
    one_to_many_rate_ds2_matched = (ds2_one_to_many / ds2_matched) if ds2_matched else 0.0
    one_to_many_rate_ds2_total = (ds2_one_to_many / ds2_total) if ds2_total else 0.0

    # Percent of companies without a match across both datasets.
    total_companies = ds1_total + ds2_total
    unmatched_count = (ds1_total - ds1_matched) + (ds2_total - ds2_matched)
    unmatched_records = (unmatched_count / total_companies) if total_companies else 0.0

    return {
        "ds1_companies_total": ds1_total,
        "ds2_companies_total": ds2_total,
        "ds1_matched_companies": ds1_matched,
        "ds2_matched_companies": ds2_matched,
        "match_rate_ds1": match_rate_ds1,
        "match_rate_ds2": match_rate_ds2,
        "unmatched_records": unmatched_records,
        "ds1_one_to_many_companies": ds1_one_to_many,
        "one_to_many_rate_ds1_total": one_to_many_rate_ds1_total,
        "one_to_many_rate_ds1_matched": one_to_many_rate_ds1_matched,
        "ds2_one_to_many_companies": ds2_one_to_many,
        "one_to_many_rate_ds2_total": one_to_many_rate_ds2_total,
        "one_to_many_rate_ds2_matched": one_to_many_rate_ds2_matched,
        "address_level_matches": int(len(matches_df)),
    }
