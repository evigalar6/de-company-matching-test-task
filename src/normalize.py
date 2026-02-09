import re

import pandas as pd


# Common legal suffixes and noise tokens for customer/company names.
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

# Map Canadian province full names to 2-letter codes for consistency.
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
    """Rename source columns to a unified schema."""
    # col_map is {unified_name: source_name}; pandas.rename needs {source_name: unified_name}.
    reverse_map = {src: dst for dst, src in col_map.items()}
    return df.rename(columns=reverse_map).copy()


def normalize_text(value: object) -> str:
    """Normalize generic text fields (lowercase, collapse whitespace)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip().lower()
    return re.sub(r"\s+", " ", s).strip()


def normalize_postal(value: object) -> str:
    """Normalize postal/zip codes (strip spaces, uppercase)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip().upper()
    return re.sub(r"\s+", "", s)


def normalize_state(value: object) -> str:
    """Normalize Canadian province/state values to a compact form."""
    s = normalize_text(value).upper()
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    return _CANADA_PROVINCES.get(s, s)


def normalize_customer_name(name: object) -> str:
    """Normalize a customer/company name for fuzzy matching."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""

    s = str(name).strip().lower()

    # Replace punctuation with spaces (e.g., "A-B" -> "A B").
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Normalize common abbreviations to reduce false mismatches.
    s = re.sub(r"\bsaint\b", "st", s)

    # Drop 1-char tokens introduced by punctuation splits (e.g., "mary's" -> "mary s").
    s = " ".join(t for t in s.split() if len(t) > 1)

    tokens = [t for t in s.split() if t and t not in _NAME_NOISE]
    return " ".join(tokens)


def normalize_street(value: object) -> str:
    """Normalize street lines for strict location overlap checks."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def build_street_full(df: pd.DataFrame) -> pd.Series:
    """Combine street lines into a single street string."""
    cols = [c for c in ("street1", "street2", "street3") if c in df.columns]
    if not cols:
        return pd.Series([""] * len(df), index=df.index)

    parts = [df[c].fillna("").astype(str).str.strip() for c in cols]
    street = parts[0]
    for p in parts[1:]:
        street = (street + " " + p).str.strip()
    return street


def is_canadian_postal(postal_norm: str) -> bool:
    """Detect Canadian postal format like A1A1A1."""
    return bool(re.fullmatch(r"[A-Z]\d[A-Z]\d[A-Z]\d", postal_norm))


def _make_key(parts: list[str]) -> str:
    """Build a pipe-separated key; return empty string if all parts are empty."""
    joined = "|".join(parts)
    # If joined contains only separators (e.g., "||||"), treat as empty.
    if joined.replace("|", "").strip() == "":
        return ""
    return joined


def normalize_dataset(df: pd.DataFrame, col_map: dict[str, str]) -> pd.DataFrame:
    """Apply schema and add normalized columns used by matching and overlap."""
    out = apply_schema(df, col_map)

    # Build full street string from available address lines.
    out["street_full"] = build_street_full(out)

    # Normalize fields used for matching.
    out["customer_name_norm"] = out["customer_name"].apply(normalize_customer_name)
    out["city_norm"] = out.get("city", pd.Series([""] * len(out))).apply(normalize_text)
    out["state_norm"] = out.get("state", pd.Series([""] * len(out))).apply(normalize_state)
    out["postal_norm"] = out.get("postal", pd.Series([""] * len(out))).apply(normalize_postal)

    # Normalize country and fill missing values when possible.
    country_raw = out.get("country", pd.Series([""] * len(out))).fillna("")
    out["country_norm"] = country_raw.apply(normalize_text)

    if "country_code" in out.columns:
        is_ca = out["country_code"].fillna("").astype(str).str.upper().eq("CA")
        out.loc[(out["country_norm"] == "") & is_ca, "country_norm"] = "canada"

    out.loc[(out["country_norm"] == "") & out["postal_norm"].apply(is_canadian_postal), "country_norm"] = "canada"

    # Build a blocking key to reduce candidate comparisons.
    out["block_key"] = out["country_norm"] + "|" + out["postal_norm"]
    out.loc[out["postal_norm"] == "", "block_key"] = out["country_norm"] + "|" + out["city_norm"]

    # Strict location key (includes street) for exact overlap checks.
    out["street_norm"] = out["street_full"].apply(normalize_street)
    out["location_key"] = out.apply(
        lambda r: _make_key(
            [
                str(r.get("street_norm", "") or ""),
                str(r.get("city_norm", "") or ""),
                str(r.get("state_norm", "") or ""),
                str(r.get("postal_norm", "") or ""),
                str(r.get("country_norm", "") or ""),
            ]
        ),
        axis=1,
    )

    # Loose location key (no street) for human-friendly overlap checks.
    out["location_key_loose"] = out.apply(
        lambda r: _make_key(
            [
                str(r.get("city_norm", "") or ""),
                str(r.get("state_norm", "") or ""),
                str(r.get("postal_norm", "") or ""),
                str(r.get("country_norm", "") or ""),
            ]
        ),
        axis=1,
    )

    return out
