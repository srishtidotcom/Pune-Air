# Clustering daily PM2.5 patterns in Pune

This document explains the clustering concepts, the choices implemented in this project, and how to interpret the generated results. The research unit is a **station-day**: one monitoring station on one Pune calendar day, represented by 24 hourly PM2.5 concentrations. A cluster is a set of station-days with similar profiles. It describes recurring measurements; it does not identify their causes.

## From source readings to a 24-hour profile

The 11 source files in `dataset/` contain nominal 15-minute observations. The pipeline reads the `Timestamp` and `PM2.5 (µg/m³)` columns, parses each timestamp with its UTC offset, and converts it to `Asia/Kolkata`. All dates, hours, and weekday labels below use that local time. The shared analysis window is 2024-01-01 00:00 through 2025-01-01 00:00, with the end excluded. Source files remain untouched.

Only finite, numeric PM2.5 values at or above zero are valid. For each station and local hour, the pipeline takes the arithmetic mean of valid readings. It retains the number of contributing readings and treats an hour with fewer than **two** as unobserved. The audit records missing or invalid values, timestamps, and monthly coverage before clustering.

For each station-day, the 24 hourly means become a vector

`[PM at 00:00, PM at 01:00, ..., PM at 23:00]`.

A day needs at least **20 observed hours**. Remaining gaps must be inside the day and at most **two consecutive hours** long; those gaps are filled by linear interpolation between the surrounding observed hours. A day with a missing first or last hour, a longer gap, or fewer than 20 observed hours is ineligible. Consequently, an eligible day can contain at most four filled hours. `daily_profiles.csv` keeps the final values (`h00`–`h23`), source counts (`n00`–`n23`), fill flags (`filled_00`–`filled_23`), and totals of observed and filled hours. Its `daily_mean` is the mean of the final 24-hour profile, including any filled values.

A station enters the shared analysis if it has at least **90 eligible 2024 days** and at least **three months with 10 or more eligible days**. The current audit includes all 11 stations, though their coverage differs substantially. `station_audit.csv` records each decision; `monthly_coverage.csv` records observed hours and eligible days for every station-month. An absent station-month contributes no days to its frequency table.

## Why run two analyses?

Each eligible day has 24 numeric features, one per hour. The project fits two separate models to the same eligible station-days:

| Analysis | Features given to clustering | What similarity emphasizes |
| --- | --- | --- |
| **Absolute** | Hourly PM2.5 in µg/m³ | Both concentration level and the timing/size of peaks |
| **Shape** | Each hour's deviation from that day's mean, divided by that day's standard deviation | Relative timing and shape within the day |

For the shape analysis, each day's transformed hour is `(hourly PM2.5 − daily mean) / daily standard deviation`. The standard deviation uses the 24 final hourly values. A profile with standard deviation below **1 µg/m³** becomes an all-zero vector, avoiding unstable division for a nearly flat day. The original PM2.5 values and daily mean remain available for describing the resulting shape groups. The current run contains **91 nearly flat days**.

Both methods compare profiles using Euclidean distance in their respective 24-dimensional feature spaces. In absolute space, a persistently higher day can be far from a lower day even if their peak hours match. Within-day standardization removes the day's overall level from the shape model, so profiles with different concentrations can group together when their relative hourly patterns are similar.

## Algorithms and model selection

The pipeline compares two algorithms separately for absolute and shape features:

1. **K-Means** assigns days to `k` centers and updates those centers to reduce within-group squared distance. It uses 20 initializations and random seed 42.
2. **Ward agglomerative clustering** starts with individual days and repeatedly merges groups while limiting the increase in within-group variation.

For each algorithm it tries **2, 3, 4, and 5 clusters**. The comparison table records the silhouette score and the smallest and largest group sizes. A day's silhouette is `(b − a) / max(a, b)`, where `a` is its average distance to other days in its own group and `b` is its average distance to the nearest other group. Higher values indicate clearer separation under the chosen distance. The score uses a reproducible sample of up to 1,000 station-days, with seed 42. It measures separation, not whether the groups are naturally occurring or scientifically causal.

The pipeline calls a candidate *viable* when its smallest group has at least `max(10, int(0.03 × number of eligible days))` members. With 2,408 days, that threshold is **72 days**. It selects the viable candidate with the highest silhouette, resolving a tie in favor of fewer groups and then the method name. If no candidate meets the size rule, it selects the highest-scoring candidate overall. The model comparison CSVs show every tried option, including those not selected. The selection is automatic; the plotted curves are used to interpret the result.

The normal installation uses scikit-learn. `summary.json` records the backend because a SciPy fallback is available when scikit-learn is unavailable; the fallback may produce slightly different labels or scores.

## How cluster names and charts are made

After fitting, the pipeline computes each group's **median PM2.5 at each clock hour**, using the original absolute profiles even for the shape groups. The resulting 24-point curve is a typical measured concentration profile, rather than a model center. If its peak-to-trough range is below the greater of 5 µg/m³ or 15% of its mean, the name is `fairly even`. Otherwise, the hour of its maximum assigns `overnight high` (00:00–05:00), `morning high` (06:00–10:00), `afternoon high` (11:00–16:00), or `evening high` (17:00–23:00). Absolute groups also get `higher level` or `lower level` according to whether their median daily mean is at least the overall median. Those level words are relative to this sample; they are not health categories or standards.

Names are assigned **after** clustering from the observed group curves. More than one group can receive the same shape name. Numeric cluster IDs have no intrinsic order and can change between runs or backends; use the names, curves, counts, and concentration summaries together. The charts use per-hour group medians, so their highest hour need not match every individual day's peak.

The frequency tables group eligible days by station, month, weekday/weekend, station-month, and station-weekend. For any row, `percent = 100 × cluster days / eligible days` in that row's station or time group. The denominator is displayed beside the percentage. A calendar cell is colored only when that station-day met the eligibility rules; a blank cell does not mean zero pollution.

## What the current 2024 run found

The scikit-learn run produced **2,408 eligible station-days** across all 11 stations, from **January 2 through December 31, 2024**. Both analyses selected **K-Means with two groups**.

| Analysis | Selected silhouette | Group | Eligible days | Median daily PM2.5 | Peak of median curve |
| --- | ---: | --- | ---: | ---: | --- |
| Absolute | 0.455 | Afternoon high, lower level | 1,530 | 22.2 µg/m³ | 14:00 IST |
| Absolute | 0.455 | Afternoon high, higher level | 878 | 56.3 µg/m³ | 14:00 IST |
| Shape | 0.199 | Overnight high | 1,081 | 23.0 µg/m³ | 01:00 IST |
| Shape | 0.199 | Afternoon high | 1,327 | 42.8 µg/m³ | 14:00 IST |

The absolute groups share an afternoon peak and differ strongly in concentration level. The shape groups summarize an overnight-high pattern and an afternoon-high pattern. The shape silhouette of 0.199 signals substantial overlap, so these are broad summaries rather than sharply separated daily types. The selected model did not yield distinct morning or evening groups.

The higher absolute group accounted for **75.6% of 201 eligible January station-days**, **0.8% of 238 July station-days**, and **85.2% of 230 November station-days**. The afternoon shape group accounted for **55.4% of 1,710 eligible weekday station-days** and **54.3% of 698 eligible weekend station-days**. These pooled percentages describe the observed eligible days. Uneven station coverage can change the mix of stations across months and day types, so these contrasts do not establish a seasonal, weekday, traffic, weather, or emission-source cause.

## Reproduce and inspect

From the repository root, using the dependencies in `requirements.txt`:

```bash
python -m pune_air.pipeline --dataset dataset --output outputs
streamlit run app.py
```

The pipeline writes generated files only to `outputs/`. Start with `summary.json` for the selected models and sample size, `station_audit.csv` and `monthly_coverage.csv` for coverage, `daily_profiles.csv` for day-level values and flags, `absolute_model_comparison.csv` and `shape_model_comparison.csv` for all candidate models, and `*_cluster_names.csv` for group descriptions. The `*_by_*.csv` tables, `*_curves.png`, and `*_calendar_example.png` show frequencies and typical curves. The dashboard lets you filter these results by station, month, and day type and inspect a selected day's hourly readings.

Implementation lives in [`pune_air/core.py`](pune_air/core.py) for cleaning and profile construction, [`pune_air/clustering.py`](pune_air/clustering.py) for algorithms, selection, and names, and [`pune_air/pipeline.py`](pune_air/pipeline.py) for the audit and outputs. [`README.md`](README.md) has the full setup commands and project overview.

## Interpretation limits

These are descriptive clusters of the supplied PM2.5 measurements. The models pool station-days without reweighting stations, so stations with more eligible days influence fitting more. Missing periods, local interpolation, possible sensor anomalies, and differing station operation affect the sample. The pipeline does not validate instrument calibration or measure meteorology, traffic, or emissions. Cluster membership and frequency differences alone cannot identify why a pattern occurred.
