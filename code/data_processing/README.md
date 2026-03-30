# Data Processing Pipeline

This folder contains the complete Julia data processing pipeline for the SMM estimation project.

## Files

### `cps_basic.jl`
Processes CPS Basic Monthly data from IPUMS-CPS.

**Functions:**
- `load_cps_basic(raw_path)` — Load raw CPS Basic extract, apply sample restrictions
- `compute_stock_moments(df, window)` — Unemployment rates and labour force composition
- `compute_transition_rates(df, window)` — Panel-matched job-finding, separation, EE transition rates
- `compute_industry_skill_shares(df, window)` — Skill composition by 2-digit NAICS industry
- `covid_bls_correction!(df)` — Reclassify employed-absent during Mar-Jun 2020

**Output moments:**
- Unemployment rates: `ur_total`, `ur_U`, `ur_S`
- Labour force: `skilled_share`, `training_share`
- Transitions: `jfr_U`, `jfr_S`, `sep_rate_U`, `sep_rate_S`, `ee_rate_S`, `training_rate`

### `cps_asec.jl`
Processes CPS ASEC wage data from IPUMS-CPS.

**Functions:**
- `load_cps_asec(raw_path, cpi_deflator)` — Load ASEC, construct real hourly wages, trim outliers
- `compute_wage_moments(df, window)` — Wage distribution moments (mean, median, variance, skewness)
- `compute_influence_functions_wages(df, window, moments)` — Influence functions ψ for wage moments

**Output moments:**
- Wage levels: `mean_wage_U`, `mean_wage_S`, `p50_wage_U`, `p50_wage_S`
- Wage dispersion: `wage_sd_U`, `wage_sd_S`, `emp_var_U`, `emp_var_S`
- Wage premium: `wage_premium`
- Higher moments: `emp_cm3_U`, `emp_cm3_S` (third central moments)

### `jolts.jl`
Processes JOLTS vacancy data from BLS.

**Functions:**
- `load_jolts(raw_dir)` — Load JOLTS CSV files, convert from thousands to counts
- `compute_tightness(jolts_df, cps_basic_df, window, industry_skill_shares)` — Allocate vacancies to skill, compute θ_U and θ_S
- `compute_influence_functions_tightness(...)` — Influence functions for tightness moments

**Output moments:**
- Market tightness: `theta_U`, `theta_S`
- Component counts: `vacancies_U`, `vacancies_S`, `unemp_U`, `unemp_S`

### `main.jl`
Main entry point orchestrating the full pipeline.

**Usage:**
```bash
julia --threads auto code/data_processing/main.jl
```

**Pipeline:**
1. Load raw CPS Basic, CPS ASEC, JOLTS data
2. Apply COVID correction to CPS Basic (employed-absent reclassification)
3. For each estimation window, compute all 25 moments
4. Compute influence functions ψ_i for each moment and observation
5. Form variance-covariance matrix Σ̂ = (1/N) Σ_i ψ_i ψ_i'
6. Invert to get weight matrix W = Σ̂^{-1}
7. Save outputs:
   - `derived/moments_{window}.csv` — Moment estimates with standard errors
   - `derived/sigma_{window}.csv` — Variance-covariance matrix
   - `derived/W_{window}.csv` — Weight matrix for SMM
   - `derived/all_moments.csv` — All moments stacked across windows

## Estimation Windows

```julia
:base_fc      → (2003-01) to (2007-12)  # Pre-FC baseline
:crisis_fc    → (2008-01) to (2009-06)  # Financial crisis
:base_covid   → (2015-01) to (2019-12)  # Pre-COVID baseline
:crisis_covid → (2020-03) to (2021-12)  # COVID crisis
```

## Configuration

**In `main.jl`, set the following paths:**

```julia
const CPS_BASIC_FILE = "data/raw/cps_basic.csv"      # IPUMS-CPS Basic extract
const CPS_ASEC_FILE = "data/raw/cps_asec.csv"        # IPUMS-CPS ASEC extract
const JOLTS_DATA_DIR = "data/raw/jolts/"             # BLS JOLTS files
```

**CPI deflator:**
Populate the `CPI_DEFLATOR` dict with actual CPI-U factors (monthly, 2000=1.0 base).
Currently set to 1.0 (no deflation); replace with actual values from BLS.

## Data Requirements

### CPS Basic Monthly (IPUMS-CPS)
Required variables: `YEAR`, `MONTH`, `CPSIDP`, `MISH`, `AGE`, `EDUC`, `EMPSTAT`, `LABFORCE`, `CLASSWKR`, `IND`

### CPS ASEC (IPUMS-CPS)
Required variables: `YEAR`, `MONTH`, `AGE`, `EDUC`, `INCWAGE`, `WKSWORK1`, `UHRSWORKLY`, `CLASSWKR`

### JOLTS (BLS)
Expected files:
- `jolts_total.csv` — Total vacancies by month
- `jolts_by_industry.csv` — Vacancies by 2-digit NAICS industry
Columns: `YEAR`, `MONTH`, `NAICS` (or `INDUSTRY`), `VACANCIES` (in thousands)

## Sample Restrictions (Applied Across All Data)

1. **Age:** 16–64, civilian labour force
2. **Employment:** Exclude armed forces
3. **Wages (ASEC):** Wage/salary workers, positive income/weeks/hours
4. **Skill Classification:**
   - Unskilled: `EDUC < 111` (no bachelor's degree)
   - Skilled: `EDUC ≥ 111` (bachelor's degree or higher)

## Panel Matching (Transitions)

- Match on `CPSIDP` across consecutive months
- Restrict to `MISH ∈ {1,2,3,5,6,7}` to avoid rotation group breaks
- Lagged state at month t, current state at month t+1

## Influence Functions

The pipeline computes influence functions ψ_i for standard errors via the sandwich formula.

**Mean:** ψ_i = y_i - m̂

**Median:** ψ_i = (1{y_i ≤ m̂} − 0.5) / (2 f̂(m̂))

**Variance:** ψ_i = (y_i − ȳ)² − σ̂²

**Ratio (tightness, premium):** Delta method applied to component influence functions

## Output Format

### `moments_{window}.csv`
```
moment,value,std_error
ur_total,0.0546,0.0012
ur_U,0.0623,0.0015
...
```

### `sigma_{window}.csv`
Variance-covariance matrix (25 × 25) in wide CSV format.

### `all_moments.csv`
```
window,moment,value
base_fc,ur_total,0.0546
base_fc,ur_U,0.0623
...
crisis_covid,theta_U,1.234
```

## Dependencies

- `DataFrames.jl` — Data manipulation
- `CSV.jl` / `Arrow.jl` — I/O
- `Statistics.jl` / `StatsBase.jl` — Statistical functions
- `Dates.jl` — Date handling

## Notes

- The COVID correction (Mar–Jun 2020) reclassifies employed-absent workers per BLS revision
- JOLTS vacancies are reported in thousands; scripts multiply by 1000
- Wage trimming (1st, 99th percentiles) applied independently per window
- Industry skill shares used for allocating JOLTS vacancies to skill segments
- Placeholder VCV computation in `main.jl`; update `compute_vcov_matrix_window()` with actual influence functions
