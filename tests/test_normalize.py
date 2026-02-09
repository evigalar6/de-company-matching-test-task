"""Tests for normalization and schema handling.

Run:
    python3 -m unittest discover -s tests

These tests focus on the core transformations used for matching:
- company name normalization
- schema validation (required columns)
- key generation (block/location keys)
"""

import unittest

import pandas as pd

from src.config import DS1_COLS
from src.normalize import apply_schema, normalize_customer_name, normalize_dataset


class TestNormalizeCustomerName(unittest.TestCase):
    """Unit tests for company-name normalization used in fuzzy matching."""

    def test_removes_noise_and_punctuation(self) -> None:
        """Legal suffixes/punctuation are removed and small abbreviations are normalized."""
        self.assertEqual(normalize_customer_name("The ACME, Inc."), "acme")
        self.assertEqual(normalize_customer_name("Saint Mary's Company"), "st mary")


class TestSchemaAndDatasetNormalization(unittest.TestCase):
    """Unit tests for schema application and dataset-level normalization."""

    def test_apply_schema_raises_on_missing_required_source_columns(self) -> None:
        """Missing required input columns should raise a readable ValueError."""
        # DS1 requires a `custname` source column (mapped to `customer_name`).
        raw_df = pd.DataFrame(
            {
                "custnmbr": ["1"],
                "addrcode": ["A"],
                # "custname" missing on purpose
                "sStreet1": ["1 Main St"],
                "sStreet2": [""],
                "sCity": ["Toronto"],
                "sProvState": ["ON"],
                "sCountry": ["Canada"],
                "sPostalZip": ["M5H 2N2"],
            }
        )
        with self.assertRaises(ValueError):
            apply_schema(raw_df, DS1_COLS)

    def test_normalize_dataset_creates_keys(self) -> None:
        """Normalization should always create key columns and keep empty keys empty."""
        raw_df = pd.DataFrame(
            {
                "custnmbr": ["1", "2"],
                "addrcode": ["A", "B"],
                "custname": ["ACME Inc.", "No Location Ltd"],
                "sStreet1": ["1 Main St", ""],
                "sStreet2": ["", ""],
                "sCity": ["Toronto", ""],
                "sProvState": ["ON", ""],
                "sCountry": ["Canada", ""],
                "sPostalZip": ["M5H 2N2", ""],
            }
        )
        normalized = normalize_dataset(raw_df, DS1_COLS)

        self.assertIn("block_key", normalized.columns)
        self.assertIn("location_key", normalized.columns)
        self.assertIn("location_key_loose", normalized.columns)

        # When all location parts are empty, the key should be empty (not "||||").
        row2 = normalized.loc[normalized["customer_id"] == "2"].iloc[0]
        self.assertEqual(row2["location_key"], "")
        self.assertEqual(row2["location_key_loose"], "")
