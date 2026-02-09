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
    """Rename source columns to a unified schema."""
    reverse_map = {src: dst for dst, src in col_map.items()}
    return df.rename(columns=reverse_map).copy()


def normalize_customer_name(name: object) -> str:
    """Normalize a customer/company name for fuzzy matching."""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return ""

    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Normalize common abbreviations.
    s = re.sub(r"\bsaint\b", "st", s)
    s = " ".join(t for t in s.split() if len(t) > 1)

    tokens = [t for t in s.split() if t and t not in _NAME_NOISE]
    return " ".join(tokens)


def normalize_postal(value: object) -> str:
    """Normalize postal/zip codes (strip spaces, uppercase)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip().upper()
    return re.sub(r"\s+", "", s)


def normalize_text(value: object) -> str:
    """Normalize generic text fields (lowercase, collapse spaces)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value).strip().lower()
    return re.sub(r"\s+", " ", s).strip()


def normalize_state(value: object) -> str:
    """Normalize Canadian province/state values to a compact form."""
    s = normalize_text(value).upper()
    if not s:
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    return _CANADA_PROVINCES.get(s, s)


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


def normalize_dataset(df: pd.DataFrame, col_map: dict[str, str]) -> pd.DataFrame:
    """Apply schema and add normalized columns used by the pipeline."""
    out = apply_schema(df, col_map)

    out["street_full"] = build_street_full(out)
    out["customer_name_norm"] = out["customer_name"].apply(normalize_customer_name)

    out["city_norm"] = out.get("city", pd.Series([""] * len(out))).apply(normalize_text)
    out["state_norm"] = out.get("state", pd.Series([""] * len(out))).apply(normalize_state)

    out["postal_norm"] = out.get("postal", pd.Series([""] * len(out))).apply(normalize_postal)

    country_raw = out.get("country", pd.Series([""] * len(out))).fillna("")
    out["country_norm"] = country_raw.apply(normalize_text)

    # Fill missing country using country code or postal pattern when possible.
    if "country_code" in out.columns:
        out.loc[(out["country_norm"] == "") & (out["country_code"].fillna("").astype(str).str.upper() == "CA"), "country_norm"] = "canada"
    out.loc[(out["country_norm"] == "") & (out["postal_norm"].apply(is_canadian_postal)), "country_norm"] = "canada"

    # Build a blocking key to reduce candidate comparisons.
    out["block_key"] = out["country_norm"] + "|" + out["postal_norm"]
    out.loc[out["postal_norm"] == "", "block_key"] = out["country_norm"] + "|" + out["city_norm"]

    return out
