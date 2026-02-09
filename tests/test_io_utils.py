"""Tests for CSV I/O utilities.

Run:
    python3 -m unittest discover -s tests

These tests cover a couple of common error cases:
- missing input file
- incorrect file extension
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.io_utils import read_csv


class TestReadCsv(unittest.TestCase):
    """Unit tests for `read_csv(...)` defensive behavior."""

    def test_raises_on_missing_file(self) -> None:
        """A missing file should raise FileNotFoundError."""
        with TemporaryDirectory() as tmp_dir:
            missing_path = Path(tmp_dir) / "missing.csv"
            with self.assertRaises(FileNotFoundError):
                read_csv(missing_path)

    def test_rejects_non_csv_extension(self) -> None:
        """Non-CSV paths should be rejected early to avoid confusing parser errors."""
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "not_csv.txt"
            path.write_text("a,b\n1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_csv(path)
