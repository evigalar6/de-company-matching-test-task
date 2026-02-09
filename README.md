# DE Company Matching Test Task

This project matches companies between two CSV datasets and produces:

* `output/merged_companies.csv` — one row per unique company from Dataset 1, enriched with matched Dataset 2 info and location overlap
* `output/metrics.json` — basic matching coverage and ambiguity metrics

The implementation uses Python + Pandas and a lightweight fuzzy string match.

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

## How to run

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Put input files into:

* `data/raw/company_dataset_1.csv`
* `data/raw/company_dataset_2.csv`

Note: The input data directory (data/raw/) is excluded from git via .gitignore; place the provided CSV files locally before running the pipeline.

3. Run:

```bash
python3 -m src.main
```

Outputs will appear in `output/`.

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
* `matched_company_ids_ds2` (list; empty if no match)
* `matched_company_names_ds2` (list; empty if no match)
* `locations_ds2` (list of DS2 `location_key` aggregated across matched DS2 ids)
* `overlapping_locations` (list intersection of `locations_ds1` and `locations_ds2`)

List columns output are consistent:

* `"[]"` when empty instead of blank/NaN
* `"[...]"` when populated

### 5) Metrics

`src/metrics.py` produces `output/metrics.json` with:

* total unique companies in each dataset
* matched unique companies in each dataset
* match rates
* `unmatched_records`: percent of companies without any match across both datasets
* one-to-many statistics (DS1 companies matching to multiple DS2 companies)
* address-level match count

## Notes / assumptions

* Matching is conservative by design due to blocking on `country|postal` where possible.
  If two datasets do not share the same postal for a company, that company may not be considered a candidate match.
* This approach prioritizes precision over recall for a simple, readable test-task solution.
* CSV output stores multi-valued fields as JSON strings for consistent downstream parsing.

## Dependencies

Minimal requirements:

* `pandas==2.2.3`
* `rapidfuzz==3.9.7`
