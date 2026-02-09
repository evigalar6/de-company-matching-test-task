from dataclasses import dataclass

import pandas as pd
from rapidfuzz import fuzz


@dataclass(frozen=True)
class MatchRow:
    """A single address-level match between Dataset 1 and Dataset 2."""

    ds1_customer_id: str
    ds1_address_code: str
    ds2_customer_id: str
    ds2_address_code: str
    score: float


def _best_match_for_row(
    ds1_row: pd.Series,
    ds2_candidates: pd.DataFrame,
    name_threshold: float,
) -> MatchRow | None:
    """Find the best Dataset 2 candidate for one Dataset 1 row within the same block.

    Args:
        ds1_row: A single Dataset 1 address row (already normalized).
        ds2_candidates: Candidate Dataset 2 rows from the same `block_key`.
        name_threshold: Minimum fuzzy name similarity score required to accept a match.

    Returns:
        A MatchRow if the best candidate meets the threshold; otherwise None.
    """
    if ds2_candidates.empty:
        return None

    ds1_name_norm = str(ds1_row.get("customer_name_norm", "") or "")
    if not ds1_name_norm:
        return None

    best_score = -1.0
    best_candidate = None

    # itertuples is much faster than iterrows and keeps this readable.
    for candidate in ds2_candidates.itertuples(index=False):
        ds2_name_norm = str(getattr(candidate, "customer_name_norm", "") or "")
        if not ds2_name_norm:
            continue

        score = float(fuzz.token_set_ratio(ds1_name_norm, ds2_name_norm))
        if score > best_score:
            best_score = score
            best_candidate = candidate

    if best_candidate is None or best_score < name_threshold:
        return None

    return MatchRow(
        ds1_customer_id=str(ds1_row["customer_id"]),
        ds1_address_code=str(ds1_row["address_code"]),
        ds2_customer_id=str(getattr(best_candidate, "customer_id")),
        ds2_address_code=str(getattr(best_candidate, "address_code")),
        score=best_score,
    )


def match_datasets(
    ds1_df: pd.DataFrame,
    ds2_df: pd.DataFrame,
    name_threshold_strong: float,
    name_threshold_with_postal: float,
) -> pd.DataFrame:
    """Match DS1 to DS2 at the address level using blocking + name similarity.

    Args:
        ds1_df: Normalized Dataset 1 records.
        ds2_df: Normalized Dataset 2 records.
        name_threshold_strong: Similarity threshold used when blocking falls back to city.
        name_threshold_with_postal: Similarity threshold used when postal code is present.

    Returns:
        DataFrame of address-level matches with columns:
        `ds1_customer_id`, `ds1_address_code`, `ds2_customer_id`, `ds2_address_code`, `score`.

    Raises:
        ValueError: If expected normalized columns are missing from the inputs.
    """
    required_cols = {"customer_id", "address_code", "block_key", "postal_norm", "customer_name_norm"}
    missing_ds1 = sorted(required_cols - set(ds1_df.columns))
    missing_ds2 = sorted(required_cols - set(ds2_df.columns))
    if missing_ds1 or missing_ds2:
        raise ValueError(
            "Missing required normalized columns for matching. "
            f"DS1 missing: {missing_ds1 or 'none'}. DS2 missing: {missing_ds2 or 'none'}."
        )

    ds2_by_block = {block: group for block, group in ds2_df.groupby("block_key", sort=False)}

    results: list[MatchRow] = []

    for _, ds1_row in ds1_df.iterrows():
        block_key = str(ds1_row.get("block_key", "") or "")
        ds2_candidates = ds2_by_block.get(block_key)
        if ds2_candidates is None:
            continue

        postal_norm = str(ds1_row.get("postal_norm", "") or "")
        threshold = name_threshold_with_postal if postal_norm else name_threshold_strong

        match = _best_match_for_row(ds1_row, ds2_candidates, threshold)
        if match is not None:
            results.append(match)

    return pd.DataFrame([m.__dict__ for m in results])
