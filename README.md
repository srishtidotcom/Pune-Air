# Discovering Daily Air Pollution Patterns in Pune Using Clustering

This project groups **local 24-hour PM2.5 profiles** from the 11 supplied Pune station files. It asks which profiles recur and how their frequency and concentration vary across stations, months, and weekdays. Clusters are descriptions of measurements, **not evidence of causes**.

For the clustering concepts, implementation choices, model selection, and interpretation of the current results, see [clustering.md](clustering.md).

## Reproduce

From the repository root, with Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python -m pune_air.pipeline --dataset dataset --output outputs
streamlit run app.py
```

The input CSVs stay untouched in `dataset/`. Every generated CSV, PNG, and JSON goes to `outputs/` (ignored by Git). The pipeline is deterministic with seed 42. The current workspace outputs were generated with scikit-learn. In an offline environment lacking scikit-learn, the pipeline can run with a SciPy fallback and records that backend in `outputs/summary.json`; minor label and metric differences across backends are possible.

## Data audit and scope

The files contain `Timestamp` strings with `+0000` offsets and `PM2.5 (µg/m³)`. The timestamp is converted from UTC to `Asia/Kolkata`; dates and weekdays refer to Pune local time. Each file also contains one trailing `2025` line with no timestamp or PM2.5; the audit counts it as one bad timestamp. No duplicate timestamps, off-grid 15-minute timestamps, negative PM2.5 values, or other nonnumeric PM2.5 entries were found. Blank PM2.5 cells are missing. There is no upper cutoff: high positive values remain in the analysis and may include sensor anomalies.

The shared analysis window is **2024-01-01 00:00 to 2025-01-01 00:00 IST**. All 11 stations meet the prespecified inclusion rule of at least 90 eligible days and at least three months with 10 eligible days. This does **not** mean they cover every month: Bhosari has no eligible days from August through December, and Bhumkar Nagar starts producing eligible days in June. Hadapsar starts in May and has only 99 eligible days. The January 1 local day has no eligible profile because the source files generally begin at 00:00 UTC (05:30 IST). See `outputs/station_audit.csv` for full-file ranges and invalid counts, and `outputs/monthly_coverage.csv` for observed hours and eligible days in each local month. Months with zero eligible days are excluded from month-specific percentages.

## Preparation and completeness

1. Parse timestamps with their source UTC offset. Convert to IST. Remove duplicate timestamps, keeping the first (none occur in this dataset).
2. Keep numeric, finite PM2.5 values at or above zero, in the source unit µg/m³. Average valid quarter-hour observations within each local clock hour. An hour needs **at least two** valid observations. The `hourly.csv` file records `observations` for every hour with valid samples; underfilled hours have a blank PM2.5 mean.
3. Build one vector for each station and local calendar day. Require **at least 20 measured hours**. Fill only interior missing runs of **at most two consecutive hours** using linear interpolation. Do not extrapolate across midnight. Reject any day with a longer gap or missing first/last hour. `daily_profiles.csv` retains each hourly value, its contributing observation count (`n00`–`n23`), and interpolation flags (`filled_00`–`filled_23`). An interpolated hour has zero or one valid source observation; its source count remains visible.
4. Include a station if it has at least 90 eligible days and at least three local months with 10 or more eligible days. Station exclusion decisions appear in the audit file.

## Clustering and assessment

Each station-day is one observation. The **absolute** run uses its 24 hourly µg/m³ values. The **shape** run uses within-day z scores, subtracting each day's mean and dividing by its standard deviation. A day with standard deviation below 1 µg/m³ becomes an all-zero vector so that nearly flat profiles remain finite; 91 such days occur here.

For both runs, K-Means (20 starts, seed 42) and Ward agglomerative clustering are compared for **2–5 clusters**. The pipeline reports sampled silhouette and smallest/largest cluster size in the model comparison CSVs. A candidate is considered viable when every cluster has at least 10 days and at least 3% of all eligible days. The viable candidate with the highest silhouette is selected, breaking ties toward fewer clusters. The silhouette is sampled at up to 1,000 days with seed 42. Curve plots and the cluster size threshold are used to judge interpretability; the selected curves should be read as broad patterns, not fine distinctions. Cluster IDs are arbitrary.

Group names are generated from the median observed curves. A low-range curve is called `fairly even`; otherwise its highest hour determines an overnight, morning, afternoon, or evening description. Absolute groups also receive a higher/lower level description based on their median daily concentration relative to the full sample. The names are descriptions of the curves, not labels supplied in advance.

## Outputs and observed findings from this run

The 2024 local window yielded **2,408 eligible station-days** across all 11 stations, from **January 2 through December 31**. K-Means with two groups won both comparisons in the scikit-learn run: silhouette **0.455** for absolute concentrations and **0.199** for shape. The lower shape silhouette indicates overlap between shapes.

The absolute groups had median daily PM2.5 of **56.3 µg/m³** (878 days) and **22.2 µg/m³** (1,530 days). Both median curves peaked around 14:00 IST, so the absolute result mainly separates concentration levels. The higher group appeared on **75.6% of 201 eligible January station-days**, **0.8% of 238 July station-days**, and **85.2% of 230 November station-days**. These are pooled station-day shares, affected by which stations were operating in each month.

The shape groups were **overnight high** (1,081 days) and **afternoon high** (1,327 days), based on their median curves. The afternoon group occurred on **81.6% of 201 eligible January station-days** and **29.4% of 238 July station-days**. Its weekday share was **55.4% of 1,710 eligible station-days**, versus **54.3% of 698 weekend station-days**; this small pooled difference should not be interpreted as a causal weekday effect. The overnight shape share ranged from **25.2% at Bhosari (159 eligible days)** to **76.6% at Karve Road (274 eligible days)**. Per-station, per-month, and weekday/weekend counts and percentages are in the `*_by_*.csv` files.

The responsive Streamlit dashboard uses a muted autumn theme. Its station, month, day-type, and analysis filters sit together at the top. The page then walks through a summary, typical hourly curves, month and weekday comparisons, station comparisons, a daily calendar, and measured coverage. Each section pairs the chart or table with a short explanation. Counts, eligible-day denominators, and percentages remain visible. The clickable Pune map selects a station, and the calendar can show a selected day's 24 hourly readings. Download the current station-day view as CSV. Sites without eligible days under a filter stay on the map in gray. The map basemap requires internet access, but the station selector and data views work offline. Marker positions come from [published station coordinates](https://huggingface.co/datasets/Zeel/C1/blob/main/station_data.csv) and a [Maharashtra/CPCB station annexure](https://www.greentribunal.gov.in/sites/default/files/news_updates/Affidavit%20in%20Reply%20by%20R-1%20in%20OA%20No.48-2025%20%28page%20nos.245-352%29.pdf); their sources are recorded in `pune_air/station_locations.csv`. Markers identify monitoring sites, not boundaries of measured regions. Static `*_curves.png` and `*_calendar_example.png` files provide quick inspection without Streamlit.

## Limits

The source files are labelled 15-minute data, but observed missing blocks and uneven station operation restrict comparisons. Pooled monthly shares can shift as station coverage changes. The pipeline does not check calibration, instrument changes, or whether repeated quarter-hour values reflect actual readings. Interpolation affects at most four missing hours per accepted day under the 20-hour rule. K-Means and Ward impose compact groups and cannot prove natural boundaries. The two shape groups are broad, overlapping summaries; a separate morning or evening cluster was not supported by the selected model. PM2.5 alone cannot identify emission sources, weather effects, traffic effects, or causes of temporal patterns.
