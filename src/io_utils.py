from pathlib import Path

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file into a pandas DataFrame."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}\n"
            "Place the provided CSVs under data/raw/ (see README.md) and rerun."
        )

    # Read as strings to avoid type inference surprises (e.g., IDs losing leading zeros).
    # Empty fields stay as empty strings instead of NaN to simplify downstream normalization.
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV with a proper header and no index column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, header=True, encoding="utf-8")
