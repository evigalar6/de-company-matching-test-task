# DE Company Matching Test Task

This project matches companies between two CSV datasets and produces:

* `output/merged_companies.csv` — one row per unique company from Dataset 1, enriched with matched Dataset 2 info and location overlap
* `output/metrics.json` — basic matching coverage and ambiguity metrics

The implementation uses Python + Pandas and a lightweight fuzzy string match.

## Quick start

1. (Optional) Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Put input files into `data/raw/company_dataset_1.csv` and `data/raw/company_dataset_2.csv`

Note: `data/raw/` is excluded from git via `.gitignore`; place the provided CSV files locally before running the pipeline.

4. Run:

```bash
python3 -m src.main
```

5. Outputs: `output/merged_companies.csv` and `output/metrics.json`

## Tests

The project includes a small `unittest` suite to demonstrate code quality and catch
basic regressions:

```bash
python3 -m unittest discover -s tests
```

## Project structure

```text
data/
  raw/
    company_dataset_1.csv
    company_dataset_2.csv
output/
  merged_companies.csv
  metrics.json
src/
  config.py       # paths, column mappings, thresholds
  io_utils.py     # read/write helpers
  normalize.py    # schema unification + normalization + keys for matching/overlap
  matching.py     # address-level matching (blocking + fuzzy name similarity)
  metrics.py      # match-rate and ambiguity metrics
  main.py         # orchestration: read -> normalize -> match -> export
```

## What the pipeline does

### 1) Schema unification

The two datasets use different column names. In `src/config.py`, each dataset is mapped into a unified schema:

* `customer_id`, `address_code`, `customer_name`
* `street1`, `street2`, `street3` (optional)
* `city`, `state`, `country`, `country_code` (optional), `postal`

This allows the rest of the pipeline to work on the same column names regardless of the source dataset.

### 2) Normalization

`src/normalize.py` creates additional normalized fields used for matching and location overlap:

* `customer_name_norm`: cleaned company name used for fuzzy matching

  * lowercased, punctuation removed
  * common legal suffixes removed (e.g. ltd/inc/corp/company)
  * small abbreviation tweaks (e.g. "saint" → "st")
* `city_norm`, `state_norm`, `postal_norm`, `country_norm`: standardized location parts

  * postal codes are uppercased and spaces removed
  * Canadian province names are mapped to 2-letter codes when possible
  * country is inferred as "canada" when `ccode == "CA"` or postal looks Canadian

It also builds two keys:

* `block_key`:

  * `country|postal` when postal exists (strong blocking)
  * `country|city` when postal is missing (fallback blocking)
    This reduces comparisons and keeps matching conservative.

* `location_key`:

  * `street_norm|city_norm|state_norm|postal_norm|country_norm`
    This is used to compute overlapping locations between matched companies.

### 3) Address-level matching

`src/matching.py` performs matching at the address level:

* candidates are compared only within the same `block_key`
* fuzzy similarity is computed with RapidFuzz `token_set_ratio` on `customer_name_norm`
* for each DS1 address row, the best DS2 candidate in the same block is selected
* thresholds:

  * when blocking by postal (stronger constraint), a slightly lower name threshold is allowed
  * when blocking by city only, a stronger name threshold is required

The result is an address-level matches dataframe:

* `ds1_customer_id`, `ds1_address_code`
* `ds2_customer_id`, `ds2_address_code`
* `score`

### 4) Company-level merged output

`src/main.py` aggregates the address-level matches into a company-level output with one row per DS1 company.

Columns in `output/merged_companies.csv`:

* `company_id_ds1`
* `company_name_ds1`
* `locations_ds1` (list of DS1 `location_key`)
* `locations_ds1_loose` (list of DS1 `location_key_loose`, i.e. city|state|postal|country)
* `matched_company_ids_ds2` (list; empty if no match)
* `matched_company_names_ds2` (list; empty if no match)
* `locations_ds2` (list of DS2 `location_key` aggregated across matched DS2 ids)
* `locations_ds2_loose` (list of DS2 `location_key_loose`, i.e. city|state|postal|country)
* `overlapping_locations` (strict full-address overlap: intersection of `locations_ds1` and `locations_ds2`)
* `overlapping_locations_loose` (loose overlap ignoring street differences: intersection of `locations_ds1_loose` and `locations_ds2_loose`)

List columns output are consistent:

* `"[]"` when empty instead of blank/NaN
* `"[...]"` when populated
* list-like columns are stored as JSON strings in the CSV (parse with `json.loads(...)` if needed)

### 5) Metrics

`src/metrics.py` produces `output/metrics.json` with:

* total unique companies in each dataset
* matched unique companies in each dataset
* match rates
* `unmatched_records`: percent of companies without any match across both datasets
* one-to-many statistics (we report one-to-many rates both over all DS1 companies and over matched DS1 companies)
* address-level match count

## Data quality issues found

During implementation, several data-quality issues were observed:

- Missing country in Dataset 1: `sCountry` is blank for most rows (≈84%). To keep blocking consistent, the pipeline infers `country_norm = "canada"` when it can be confidently derived from Dataset 2 (`ccode == "CA"`) or from a Canadian postal-code pattern (A1A1A1).
- Duplicate companies with multiple addresses in Dataset 2: Dataset 2 contains more rows than unique companies (e.g., ~1123 rows but ~938 unique `custnmbr`). Matching is performed at the address level and then aggregated to one row per Dataset 1 company in the merged output.
- Incomplete street/address lines: address line fields are frequently missing (e.g., Dataset 1 `sStreet2` empty ≈85%; Dataset 2 `address2` empty ≈63% and `address3` empty ≈94%). This makes exact location overlap strict (it may undercount overlap when addresses are partial), so name similarity becomes the primary matching signal.
- Occasional missing required fields: a small fraction of Dataset 2 rows has empty `addrcode`, `ccode`, `city`, `country` (≈1–2%), and `postal/zip` is empty for a small fraction as well (≈2%). When postal is missing, blocking falls back to `country|city`.
- Non-standardized country and province/state values: country may appear as `ca`, `canada`, `united states`, or blank; province/state may appear as full names (e.g., `BRITISH COLUMBIA`) or abbreviations. The pipeline normalizes text casing/whitespace and maps common Canadian province names to 2-letter codes where possible.
- Trailing/irregular whitespace in text fields: some company names and location fields include padded spacing (fixed-width style), so display fields are trimmed/collapsed for readable output.


## Notes

* Matching is conservative by design due to blocking on `country|postal` where possible.
  If two datasets do not share the same postal for a company, that company may not be considered a candidate match.
* This approach prioritizes precision over recall for a simple, readable test-task solution.
* `overlapping_locations` is left as an empty cell when no overlap is found to comply with the task requirements.
* `overlapping_locations_loose` is provided as an auxiliary signal because real-world street strings may differ across sources.
* Matching is performed from Dataset 1 to Dataset 2 as required by the output schema (one row per Dataset 1 company). Location fields are used for blocking to reduce comparisons; within a block, the best match is selected by normalized name similarity. Reciprocal checks/top-k analysis are possible extensions but were not required for this task.

## Dependencies

Minimal requirements:

* `pandas==2.3.3`
* `rapidfuzz==3.14.3`
