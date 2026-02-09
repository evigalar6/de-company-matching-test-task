"""Dataset schema unification and normalization helpers.

The pipeline reads two CSV datasets with different column names, maps them into a
unified schema, and creates normalized fields/keys used for matching and overlap
calculations.
"""

import re

import pandas as pd

_NAME_NOISE = {
    "inc", "incorporated",
    "corp", "corporation",
    "co", "company",
    "llc",
    "ltd", "limited",
    "plc",
    "gmbh",
    "group", "holdings", "holding",
    "the",
    "&", "and",
}

_CANADA_PROVINCES = {
    "ONTARIO": "ON",
    "BRITISH COLUMBIA": "BC",
    "ALBERTA": "AB",
    "SASKATCHEWAN": "SK",
    "MANITOBA": "MB",
    "QUEBEC": "QC",
    "NOVA SCOTIA": "NS",
    "NEW BRUNSWICK": "NB",
    "PRINCE EDWARD ISLAND": "PE",
    "NEWFOUNDLAND AND LABRADOR": "NL",
    "NORTHWEST TERRITORIES": "NT",
    "NUNAVUT": "NU",
    "YUKON": "YT",
}


def apply_schema(df: pd.DataFrame, col_map: dict[str, str]) -> pd.DataFrame:
    """Rename source columns to a unified schema.

    Args:
        df: Input DataFrame in the source schema.
        col_map: Mapping of unified column name -> source column name.

    Returns:
        A copy of the DataFrame with source columns renamed to unified names.

    Raises:
        ValueError: If required source columns are missing from the input DataFrame.
    """
    # Some fields are optional and are allowed to be missing in a dataset.
    optional_unified_cols = {"street2", "street3", "country_code"}

    missing_source_cols = [
        src
        for unified, src in col_map.items()
        if unified not in optional_unified_cols and src not in df.columns
    ]
    if missing_source_cols:
        available = ", ".join(map(str, df.columns))
        missing = ", ".join(missing_source_cols)
        raise ValueError(
            "Input dataset is missing required columns. "
            f"Missing: {missing}. Available: {available}"
        )

    reverse_map = {src: dst for dst, src in col_map.items()}
    return df.rename(columns=reverse_map).copy()


def normalize_text(value: object) -> str:
    """Normalize generic text fields (lowercase, collapse whitespace).

    Args:
        value: Any object representing a text field.

    Returns:
        Normalized text; empty string if the input is missing.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip().lower()
    return re.sub(r"\s+", " ", text).strip()


def normalize_postal(value: object) -> str:
    """Normalize postal/zip codes (strip spaces, uppercase).

    Args:
        value: Any object representing a postal/zip code.

    Returns:
        Postal code in uppercase with whitespace removed; empty string if missing.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    postal = str(value).strip().upper()
    return re.sub(r"\s+", "", postal)


def normalize_state(value: object) -> str:
    """Normalize province/state values to a compact form.

    Canadian province full names are mapped to their 2-letter codes. Other values
    are uppercased and whitespace-normalized.

    Args:
        value: Any object representing province/state.

    Returns:
        Normalized state/province; empty string if missing.
    """
    state = normalize_text(value).upper()
    if not state:
        return ""
    state = re.sub(r"\s+", " ", state).strip()
    return _CANADA_PROVINCES.get(state, state)


def normalize_customer_name(name: object) -> str:
    """Normalize a customer/company name for fuzzy matching.

    This function aims to reduce false mismatches by removing legal suffixes and
    punctuation, and by applying small abbreviation tweaks.

    Args:
        name: Any object representing a company name.

    Returns:
        A normalized name string suitable for fuzzy matching; empty string if missing.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""

    normalized = str(name).strip().lower()
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    normalized = re.sub(r"\bsaint\b", "st", normalized)
    normalized = " ".join(token for token in normalized.split() if len(token) > 1)

    tokens = [token for token in normalized.split() if token and token not in _NAME_NOISE]
    return " ".join(tokens)


def normalize_street(value: object) -> str:
    """Normalize street strings for strict location overlap checks.

    Args:
        value: Any object representing a street/address line.

    Returns:
        Lowercased, punctuation-stripped street string; empty string if missing.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    street = str(value).strip().lower()
    street = re.sub(r"[^a-z0-9\s]", " ", street)
    return re.sub(r"\s+", " ", street).strip()


def build_street_full(df: pd.DataFrame) -> pd.Series:
    """Combine street lines into a single street string.

    Args:
        df: DataFrame containing any of the columns `street1`, `street2`, `street3`.

    Returns:
        A Series with a concatenated street string per row (missing parts skipped).
    """
    street_cols = [col for col in ("street1", "street2", "street3") if col in df.columns]
    if not street_cols:
        return pd.Series([""] * len(df), index=df.index)

    street_parts = [df[col].fillna("").astype(str).str.strip() for col in street_cols]
    full_street = street_parts[0]
    for part in street_parts[1:]:
        full_street = (full_street + " " + part).str.strip()
    return full_street


def is_canadian_postal(postal_norm: str) -> bool:
    """Detect Canadian postal format like `A1A1A1`.

    Args:
        postal_norm: Postal code normalized by :func:`normalize_postal`.

    Returns:
        True if the postal code matches the Canadian pattern.
    """
    return bool(re.fullmatch(r"[A-Z]\d[A-Z]\d[A-Z]\d", postal_norm))


def _make_key(parts: list[str]) -> str:
    """Build a pipe-separated key; return empty string if all parts are empty.

    Args:
        parts: Key parts already normalized as strings.

    Returns:
        Pipe-separated key string, or empty string if all parts are empty.
    """
    joined = "|".join(parts)
    if joined.replace("|", "").strip() == "":
        return ""
    return joined


def normalize_dataset(df: pd.DataFrame, col_map: dict[str, str]) -> pd.DataFrame:
    """Apply schema and add normalized columns used by matching and overlap.

    Args:
        df: Input DataFrame (raw dataset).
        col_map: Mapping of unified column name -> source column name.

    Returns:
        Normalized DataFrame with unified columns and additional keys:
        `block_key`, `location_key`, and `location_key_loose`.

    Raises:
        ValueError: If required columns are missing after applying the schema.
    """
    unified_df = apply_schema(df, col_map)

    required_unified_cols = {"customer_id", "address_code", "customer_name", "city", "state", "postal", "country"}
    missing_unified_cols = sorted(required_unified_cols - set(unified_df.columns))
    if missing_unified_cols:
        raise ValueError(
            "Dataset is missing required columns after schema unification: "
            + ", ".join(missing_unified_cols)
        )

    unified_df["street_full"] = build_street_full(unified_df)

    unified_df["customer_name_norm"] = unified_df["customer_name"].apply(normalize_customer_name)
    unified_df["city_norm"] = unified_df["city"].apply(normalize_text)
    unified_df["state_norm"] = unified_df["state"].apply(normalize_state)
    unified_df["postal_norm"] = unified_df["postal"].apply(normalize_postal)

    country_values = unified_df["country"].fillna("")
    unified_df["country_norm"] = country_values.apply(normalize_text)

    # Dataset 1 has `country` missing for many rows. We infer Canada in a few safe cases
    # to keep blocking stable across datasets.
    if "country_code" in unified_df.columns:
        is_canada_code = unified_df["country_code"].fillna("").astype(str).str.upper().eq("CA")
        unified_df.loc[(unified_df["country_norm"] == "") & is_canada_code, "country_norm"] = "canada"

    unified_df.loc[
        (unified_df["country_norm"] == "") & unified_df["postal_norm"].apply(is_canadian_postal),
        "country_norm",
    ] = "canada"

    # Blocking key keeps candidate comparisons conservative and fast:
    # prefer (country|postal) when available; fall back to (country|city) otherwise.
    unified_df["block_key"] = unified_df["country_norm"] + "|" + unified_df["postal_norm"]
    unified_df.loc[unified_df["postal_norm"] == "", "block_key"] = (
        unified_df["country_norm"] + "|" + unified_df["city_norm"]
    )

    unified_df["street_norm"] = unified_df["street_full"].apply(normalize_street)
    unified_df["location_key"] = unified_df.apply(
        lambda row: _make_key(
            [
                str(row.get("street_norm", "") or ""),
                str(row.get("city_norm", "") or ""),
                str(row.get("state_norm", "") or ""),
                str(row.get("postal_norm", "") or ""),
                str(row.get("country_norm", "") or ""),
            ]
        ),
        axis=1,
    )

    unified_df["location_key_loose"] = unified_df.apply(
        lambda row: _make_key(
            [
                str(row.get("city_norm", "") or ""),
                str(row.get("state_norm", "") or ""),
                str(row.get("postal_norm", "") or ""),
                str(row.get("country_norm", "") or ""),
            ]
        ),
        axis=1,
    )

    return unified_df
