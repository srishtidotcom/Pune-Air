from __future__ import annotations

import argparse
import json
import calendar
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from .core import PM, TZ, START, END, read_station, hourly_aggregate, build_profiles, normalize_profiles
from .clustering import BACKEND, compare_models, name_clusters

HOURS = [f"h{i:02d}" for i in range(24)]


def audit_station(df: pd.DataFrame, filename: str) -> dict:
    t = df.local_time
    v = df.pm_numeric
    in_2024 = t.between(START, END, inclusive="left")
    return {"file": filename, "station": df.station.iloc[0], "rows": len(df),
            "first_local": str(t.min()), "last_local": str(t.max()),
            "bad_timestamp": int(t.isna().sum()), "duplicate_timestamps": int(t.duplicated().sum()),
            "off_15_minute_grid": int((t.notna() & ((t.dt.minute % 15 != 0) | t.dt.second.ne(0))).sum()),
            "pm_missing": int(df.pm_raw.eq("").sum()),
            "pm_nonnumeric": int((df.pm_raw.ne("") & v.isna()).sum()),
            "pm_negative": int(v.lt(0).sum()), "pm_nonfinite": int((v.notna() & ~np.isfinite(v)).sum()),
            "valid_pm_all": int(df.valid.sum()), "rows_local_2024": int(in_2024.sum()),
            "valid_pm_local_2024": int((in_2024 & df.valid).sum())}


def save_curve_plot(profiles: pd.DataFrame, names: pd.DataFrame, mode: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for row in names.itertuples():
        subset = profiles.loc[profiles[f"{mode}_cluster"].eq(row.cluster), HOURS]
        y = subset.median().to_numpy()
        ax.plot(range(24), y, label=f"{row.cluster}: {row.name} (n={row.eligible_days})")
    ax.set(xlabel="Hour in Pune (IST)", ylabel="Median PM2.5 (µg/m³)", xticks=range(0, 24, 2),
           title=f"Typical daily curves: {mode} clustering")
    ax.legend(fontsize=8); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def frequency_tables(profiles: pd.DataFrame, mode: str, out: Path) -> None:
    col = f"{mode}_cluster"
    for by in (["station"], ["month"], ["weekend"], ["station", "month"], ["station", "weekend"]):
        counts = profiles.groupby(by + [col]).size().rename("days").reset_index()
        totals = profiles.groupby(by).size().rename("eligible_days").reset_index()
        counts = counts.merge(totals, on=by)
        counts["percent"] = (100 * counts.days / counts.eligible_days).round(1)
        counts.to_csv(out / f"{mode}_by_{'_'.join(by)}.csv", index=False)


def save_calendar(profiles: pd.DataFrame, names: pd.DataFrame, mode: str, station: str, out: Path) -> None:
    """Save an example calendar; the dashboard supports every station and month."""
    k = len(names)
    cmap = ListedColormap(plt.cm.tab10(np.linspace(0, 1, k)))
    fig, axes = plt.subplots(3, 4, figsize=(13, 11))
    station_days = profiles[profiles.station.eq(station)].copy()
    for month, ax in enumerate(axes.flat, 1):
        first, last = calendar.monthrange(2024, month)
        grid = np.full((6, 7), np.nan)
        lookup = station_days[station_days.month.eq(month)].set_index("date")[f"{mode}_cluster"].to_dict()
        for day in range(1, last + 1):
            row, col = divmod(first + day - 1, 7)
            key = f"2024-{month:02d}-{day:02d}"
            if key in lookup:
                grid[row, col] = lookup[key]
            ax.text(col, row, str(day), ha="center", va="center", fontsize=7)
        ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=-.5, vmax=k-.5)
        ax.set(xticks=range(7), xticklabels=list(calendar.day_abbr), yticks=[], title=calendar.month_name[month])
        ax.set_xticks(np.arange(-.5, 7, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
        ax.grid(which="minor", color="lightgray", linewidth=.5)
    fig.suptitle(f"{station}: {mode} cluster calendar (blank = ineligible)")
    fig.text(.5, .01, "  |  ".join(f"{r.cluster}: {r.name}" for r in names.itertuples()), ha="center")
    fig.subplots_adjust(top=.92, bottom=.06, hspace=.48, wspace=.22)
    fig.savefig(out, dpi=150); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("dataset"))
    ap.add_argument("--output", type=Path, default=Path("outputs"))
    args = ap.parse_args()
    out = args.output; out.mkdir(parents=True, exist_ok=True)
    files = sorted(args.dataset.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSVs in {args.dataset}")

    audits, months, all_hourly = [], [], []
    for path in files:
        raw = read_station(path)
        audits.append(audit_station(raw, path.name))
        raw = raw.drop_duplicates("local_time", keep="first")
        h = hourly_aggregate(raw)
        h = h.loc[h.hour_time.between(START, END, inclusive="left")].copy()
        all_hourly.append(h)
        for month in range(1, 13):
            begin = pd.Timestamp(2024, month, 1, tz=TZ)
            finish = begin + pd.offsets.MonthBegin(1)
            part = h.loc[h.hour_time.between(begin, finish, inclusive="left")]
            months.append({"station": raw.station.iloc[0], "month": month,
                           "observed_hours": int(part.pm25.notna().sum()),
                           "possible_hours": int((finish - begin).total_seconds() / 3600),
                           "hour_coverage_percent": round(100 * part.pm25.notna().sum() / ((finish - begin).total_seconds() / 3600), 1)})
    audit = pd.DataFrame(audits)
    audit.to_csv(out / "station_audit.csv", index=False)
    hourly = pd.concat(all_hourly, ignore_index=True)
    hourly.to_csv(out / "hourly.csv", index=False)
    coverage = pd.DataFrame(months)
    provisional = build_profiles(hourly, audit.station.tolist())
    if provisional.empty:
        raise SystemExit("No eligible 2024 station-days; inspect station_audit.csv")
    eligible = provisional.groupby("station").agg(days=("date", "size"), months_10=("month", lambda x: int((x.value_counts() >= 10).sum())))
    selected = eligible.index[(eligible.days >= 90) & (eligible.months_10 >= 3)].tolist()
    audit["eligible_2024_days"] = audit.station.map(eligible.days).fillna(0).astype(int)
    audit["months_with_10_days"] = audit.station.map(eligible.months_10).fillna(0).astype(int)
    audit["included"] = audit.station.isin(selected)
    audit["decision"] = np.where(audit.included, "included", "excluded: <90 eligible days or <3 months with >=10 days")
    audit.to_csv(out / "station_audit.csv", index=False)
    coverage["eligible_days"] = coverage.apply(lambda r: int(((provisional.station == r.station) & (provisional.month == r.month)).sum()), axis=1)
    coverage.to_csv(out / "monthly_coverage.csv", index=False)
    if not selected:
        raise SystemExit("No stations meet 2024 coverage rule; inspect audit files")
    profiles = provisional[provisional.station.isin(selected)].reset_index(drop=True)
    x = profiles[HOURS].to_numpy(dtype=float)
    normalized, flat = normalize_profiles(x)
    profiles["nearly_flat"] = flat
    summary = {"time_zone": TZ, "window_local": [str(START), str(END)], "stations": selected,
               "eligible_station_days": len(profiles), "first_eligible_date": profiles.date.min(),
               "last_eligible_date": profiles.date.max(), "nearly_flat_days": int(flat.sum()), "backend": BACKEND,
               "models": {}}
    for mode, features in [("absolute", x), ("shape", normalized)]:
        comparison, labels, chosen = compare_models(features)
        comparison.to_csv(out / f"{mode}_model_comparison.csv", index=False)
        profiles[f"{mode}_cluster"] = labels
        names = name_clusters(x, labels, mode)
        names.to_csv(out / f"{mode}_cluster_names.csv", index=False)
        frequency_tables(profiles, mode, out)
        save_curve_plot(profiles, names, mode, out / f"{mode}_curves.png")
        save_calendar(profiles, names, mode, "Revenue Colony-Shivajinagar, Pune - IITM", out / f"{mode}_calendar_example.png")
        summary["models"][mode] = chosen
    profiles.to_csv(out / "daily_profiles.csv", index=False)
    with (out / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
