import unittest

import pandas as pd

from src.matching import match_datasets


class TestMatchDatasets(unittest.TestCase):
    def test_matches_within_same_block_key(self) -> None:
        ds1_df = pd.DataFrame(
            [
                {
                    "customer_id": "1",
                    "address_code": "A",
                    "block_key": "canada|M9W4Y1",
                    "postal_norm": "M9W4Y1",
                    "customer_name_norm": "acme",
                }
            ]
        )
        ds2_df = pd.DataFrame(
            [
                {
                    "customer_id": "X",
                    "address_code": "Z",
                    "block_key": "canada|M9W4Y1",
                    "postal_norm": "M9W4Y1",
                    "customer_name_norm": "acme",
                }
            ]
        )
        matches = match_datasets(ds1_df, ds2_df, name_threshold_strong=95, name_threshold_with_postal=86)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches.iloc[0]["ds1_customer_id"], "1")
        self.assertEqual(matches.iloc[0]["ds2_customer_id"], "X")

    def test_no_match_across_different_blocks(self) -> None:
        ds1_df = pd.DataFrame(
            [
                {
                    "customer_id": "1",
                    "address_code": "A",
                    "block_key": "canada|M9W4Y1",
                    "postal_norm": "M9W4Y1",
                    "customer_name_norm": "acme",
                }
            ]
        )
        ds2_df = pd.DataFrame(
            [
                {
                    "customer_id": "X",
                    "address_code": "Z",
                    "block_key": "canada|OTHER",
                    "postal_norm": "OTHER",
                    "customer_name_norm": "acme",
                }
            ]
        )
        matches = match_datasets(ds1_df, ds2_df, name_threshold_strong=95, name_threshold_with_postal=86)
        self.assertTrue(matches.empty)

