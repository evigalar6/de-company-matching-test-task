from dataclasses import dataclass
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz


@dataclass(frozen=True)
class MatchResult:
    """A single best-match result for a DS1 row."""
    ds1_customer_id: str
    ds1_address_code: str
    ds2_customer_id: str
    ds2_address_code: str
    score: float


def build_block_index(df2: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Group DS2 rows by block_key for fast candidate lookup."""
    return {k: g for k, g in df2.groupby("block_key", sort=False)}


def pick_best_match(ds1_row: pd.Series, candidates: pd.DataFrame) -> Optional[tuple[pd.Series, float]]:
    """Return the best candidate row and its fuzzy score."""
    name1 = ds1_row["customer_name_norm"]
    if not name1 or candidates.empty:
        return None

    best_score = -1.0
    best_row = None

    for _, cand in candidates.iterrows():
        name2 = cand["customer_name_norm"]
        if not name2:
            continue
        score = float(fuzz.token_set_ratio(name1, name2))
        if score > best_score:
            best_score = score
            best_row = cand

    if best_row is None:
        return None
    return best_row, best_score


def match_datasets(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    name_threshold_strong: float = 95.0,
    name_threshold_with_postal: float = 86.0,
) -> pd.DataFrame:
    """Match DS1 to DS2 using block_key + fuzzy name similarity."""
    block_index = build_block_index(df2)

    results: list[MatchResult] = []

    for _, row1 in df1.iterrows():
        block_key = row1["block_key"]
        candidates = block_index.get(block_key)
        if candidates is None:
            continue

        best = pick_best_match(row1, candidates)
        if best is None:
            continue

        best_row, score = best

        postal_present = bool(row1["postal_norm"])
        threshold = name_threshold_with_postal if postal_present else name_threshold_strong

        if score < threshold:
            continue

        results.append(
            MatchResult(
                ds1_customer_id=str(row1["customer_id"]),
                ds1_address_code=str(row1["address_code"]),
                ds2_customer_id=str(best_row["customer_id"]),
                ds2_address_code=str(best_row["address_code"]),
                score=score,
            )
        )

    return pd.DataFrame([r.__dict__ for r in results])
