from pathlib import Path

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file into a pandas DataFrame.

    Args:
        path: Path to the input CSV file.

    Returns:
        DataFrame containing the CSV contents. All columns are read as strings.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is not `.csv`, or the file cannot be parsed as CSV.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}\n"
            "Place the provided CSVs under data/raw/ (see README.md) and rerun."
        )

    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv input file, got: {path}")

    # Read as strings to avoid type-inference surprises (e.g., IDs losing leading zeros).
    # Empty fields stay as empty strings instead of NaN to simplify downstream normalization.
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Failed to decode CSV as UTF-8: {path}. "
            "If the file is not UTF-8 encoded, re-save it as UTF-8 and retry."
        ) from exc
    except pd.errors.ParserError as exc:
        raise ValueError(
            f"Failed to parse CSV: {path}. "
            "Check that the file is a valid comma-separated CSV with a header row."
        ) from exc


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to CSV with a proper header and no index column.

    Args:
        df: DataFrame to write.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, header=True, encoding="utf-8")
