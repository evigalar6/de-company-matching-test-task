from dataclasses import dataclass

import pandas as pd
from rapidfuzz import fuzz


@dataclass(frozen=True)
class MatchRow:
    """A single address-level match between DS1 and DS2."""
    ds1_customer_id: str
    ds1_address_code: str
    ds2_customer_id: str
    ds2_address_code: str
    score: float


def _best_match_for_row(
    row: pd.Series,
    candidates: pd.DataFrame,
    name_threshold: float,
) -> MatchRow | None:
    """Find the best DS2 candidate for one DS1 row within the same block."""
    if candidates.empty:
        return None

    name1 = str(row.get("customer_name_norm", "") or "")
    if not name1:
        return None

    best_score = -1.0
    best_cand: pd.Series | None = None

    # itertuples is much faster than iterrows and keeps this readable.
    for cand in candidates.itertuples(index=False):
        name2 = str(getattr(cand, "customer_name_norm", "") or "")
        if not name2:
            continue

        score = float(fuzz.token_set_ratio(name1, name2))
        if score > best_score:
            best_score = score
            best_cand = cand

    if best_cand is None or best_score < name_threshold:
        return None

    return MatchRow(
        ds1_customer_id=str(row["customer_id"]),
        ds1_address_code=str(row["address_code"]),
        ds2_customer_id=str(getattr(best_cand, "customer_id")),
        ds2_address_code=str(getattr(best_cand, "address_code")),
        score=best_score,
    )


def match_datasets(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    name_threshold_strong: float,
    name_threshold_with_postal: float,
) -> pd.DataFrame:
    """Match DS1 to DS2 at the address level using blocking + name similarity.

    Strategy:
    - Compare only records with the same block_key.
    - If block_key is based on postal, allow a slightly lower name threshold.
    - Otherwise (city-based fallback), require a strong name match.
    """
    df2_by_block = {k: g for k, g in df2.groupby("block_key", sort=False)}

    results: list[MatchRow] = []

    for _, r in df1.iterrows():
        block_key = str(r.get("block_key", "") or "")
        candidates = df2_by_block.get(block_key)
        if candidates is None:
            continue

        postal_norm = str(r.get("postal_norm", "") or "")
        threshold = name_threshold_with_postal if postal_norm else name_threshold_strong

        match = _best_match_for_row(r, candidates, threshold)
        if match is not None:
            results.append(match)

    return pd.DataFrame([m.__dict__ for m in results])
