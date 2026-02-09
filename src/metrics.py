import pandas as pd


def compute_metrics(df1: pd.DataFrame, df2: pd.DataFrame, matches: pd.DataFrame) -> dict:
    """Compute basic coverage and ambiguity metrics for the match results."""
    ds1_total = int(df1["customer_id"].nunique())
    ds2_total = int(df2["customer_id"].nunique())

    if matches.empty:
        return {
            "ds1_companies_total": ds1_total,
            "ds2_companies_total": ds2_total,
            "ds1_matched_companies": 0,
            "ds2_matched_companies": 0,
            "match_rate_ds1": 0.0,
            "match_rate_ds2": 0.0,
            "unmatched_records": 1.0 if (ds1_total + ds2_total) else 0.0,
            "ds1_one_to_many_companies": 0,
            "one_to_many_rate_ds1": 0.0,
            "address_level_matches": 0,
        }

    # Company-level mapping based on address-level matches.
    m_companies = matches.drop_duplicates(subset=["ds1_customer_id", "ds2_customer_id"])

    ds1_matched = int(m_companies["ds1_customer_id"].nunique())
    ds2_matched = int(m_companies["ds2_customer_id"].nunique())

    # DS1 companies that matched to more than one DS2 company.
    ds1_one_to_many = int(
        m_companies.groupby("ds1_customer_id")["ds2_customer_id"].nunique().gt(1).sum()
    )
    one_to_many_rate = (ds1_one_to_many / ds1_matched) if ds1_matched else 0.0

    # Percent of companies without a match across both datasets.
    unmatched_count = (ds1_total - ds1_matched) + (ds2_total - ds2_matched)
    unmatched_records = (unmatched_count / (ds1_total + ds2_total)) if (ds1_total + ds2_total) else 0.0

    return {
        "ds1_companies_total": ds1_total,
        "ds2_companies_total": ds2_total,
        "ds1_matched_companies": ds1_matched,
        "ds2_matched_companies": ds2_matched,
        "match_rate_ds1": (ds1_matched / ds1_total) if ds1_total else 0.0,
        "match_rate_ds2": (ds2_matched / ds2_total) if ds2_total else 0.0,
        "unmatched_records": unmatched_records,
        "ds1_one_to_many_companies": ds1_one_to_many,
        "one_to_many_rate_ds1": one_to_many_rate,
        "address_level_matches": int(len(matches)),
    }
