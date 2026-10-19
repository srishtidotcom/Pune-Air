"""Pure data preparation for the interactive dashboard."""
from __future__ import annotations

import calendar
import numpy as np
import pandas as pd

HOURS = [f"h{i:02d}" for i in range(24)]
ALL_STATIONS = "All stations"
PALETTES = {
    "absolute": ["#A9543D", "#89907A", "#D1A05B", "#75575D", "#AF7960"],
    "shape": ["#8D745C", "#BC6548", "#C59B54", "#81907B", "#735963"],
}
NO_DATA = "#CEC6BB"


def rgba(hex_color: str, alpha: int = 220) -> list[int]:
    return [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)] + [alpha]


def filtered_profiles(profiles: pd.DataFrame, month: int | None, day_type: str,
                      station: str = ALL_STATIONS) -> pd.DataFrame:
    result = profiles
    if month is not None:
        result = result[result.month.eq(month)]
    if day_type == "Weekdays":
        result = result[~result.weekend]
    elif day_type == "Weekends":
        result = result[result.weekend]
    if station != ALL_STATIONS:
        result = result[result.station.eq(station)]
    return result


def frequency(data: pd.DataFrame, mode: str, by: list[str]) -> pd.DataFrame:
    col = f"{mode}_cluster"
    if data.empty:
        return pd.DataFrame(columns=by + ["cluster", "days", "eligible_days", "percent"])
    counts = data.groupby(by + [col], dropna=False).size().rename("days").reset_index()
    if by:
        totals = data.groupby(by, dropna=False).size().rename("eligible_days").reset_index()
        counts = counts.merge(totals, on=by)
    else:
        counts["eligible_days"] = len(data)
    counts["percent"] = 100 * counts.days / counts.eligible_days
    return counts.rename(columns={col: "cluster"})


def frequency_matrix(data: pd.DataFrame, mode: str, group: str,
                     names: dict[int, str], all_groups: list | None = None) -> pd.DataFrame:
    """Compact counts and percentages with the eligible denominator visible."""
    rows = frequency(data, mode, [group])
    if rows.empty and all_groups is None:
        return pd.DataFrame()
    totals = rows[[group, "eligible_days"]].drop_duplicates().set_index(group) if not rows.empty else pd.DataFrame(columns=["eligible_days"])
    if all_groups is not None:
        totals = totals.reindex(all_groups, fill_value=0)
        totals.index.name = group
    result = totals.copy()
    for cluster in sorted(names):
        subset = rows[rows.cluster.eq(cluster)].set_index(group)
        counts = subset.days.reindex(result.index, fill_value=0)
        percent = 100 * counts / result.eligible_days.replace(0, np.nan)
        result[f"Group {cluster}"] = [f"{n} ({p:.1f}%)" if pd.notna(p) else "—"
                                        for n, p in zip(counts, percent)]
    return result.reset_index()


def map_markers(locations: pd.DataFrame, profiles: pd.DataFrame, mode: str,
                names: dict[int, str], selected_station: str) -> pd.DataFrame:
    """All sites remain selectable, including sites with no days in a filter."""
    color_map = {cluster: PALETTES[mode][cluster % len(PALETTES[mode])] for cluster in names}
    records = []
    for row in locations.itertuples():
        subset = profiles[profiles.station.eq(row.station)]
        n = len(subset)
        dominant = int(subset[f"{mode}_cluster"].value_counts().idxmax()) if n else None
        color = color_map[dominant] if dominant is not None else NO_DATA
        records.append({"station": row.station, "latitude": row.latitude,
                        "longitude": row.longitude, "eligible_days": n,
                        "median_pm": round(float(subset.daily_mean.median()), 1) if n else None,
                        "pattern": names[dominant] if dominant is not None else "No eligible days",
                        "color": rgba(color), "radius": 850 if row.station == selected_station else 600,
                        "line_color": rgba("#372F29" if row.station == selected_station else "#FFF9F1", 255),
                        "line_width": 3 if row.station == selected_station else 2})
    return pd.DataFrame(records)


def selected_station_from_map(event: object, known_stations: set[str]) -> str | None:
    """Read Streamlit's PydeckState safely, ignoring stale or unknown objects."""
    if not isinstance(event, dict):
        return None
    selection = event.get("selection", {})
    if not isinstance(selection, dict):
        return None
    objects = selection.get("objects", {})
    if not isinstance(objects, dict):
        return None
    picked = objects.get("stations", [])
    if not picked or not isinstance(picked[0], dict):
        return None
    station = picked[0].get("station")
    return station if station in known_stations else None


def calendar_frame(data: pd.DataFrame, mode: str, year: int, month: int,
                   names: dict[int, str]) -> pd.DataFrame:
    """One row per calendar date, including unavailable days."""
    col = f"{mode}_cluster"
    first_weekday, days = calendar.monthrange(year, month)
    lookup = data.set_index(data.date.dt.day) if not data.empty else pd.DataFrame()
    records = []
    for day in range(1, days + 1):
        value = lookup.loc[day] if day in lookup.index else None
        cluster = int(value[col]) if value is not None else None
        records.append({"date": f"{year}-{month:02d}-{day:02d}", "day": day,
                        "weekday": (first_weekday + day - 1) % 7,
                        "week": (first_weekday + day - 1) // 7,
                        "pattern": names[cluster] if cluster is not None else "No eligible profile",
                        "mean_pm": round(float(value.daily_mean), 1) if value is not None else np.nan})
    return pd.DataFrame(records)


def selected_day_from_chart(event: object) -> str | None:
    if not isinstance(event, dict):
        return None
    selection = event.get("selection", {})
    items = selection.get("calendar_pick", []) if isinstance(selection, dict) else []
    return items[0].get("date") if items and isinstance(items[0], dict) else None
