from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

PM = "PM2.5 (µg/m³)"
TZ = "Asia/Kolkata"
START = pd.Timestamp("2024-01-01", tz=TZ)
END = pd.Timestamp("2025-01-01", tz=TZ)
MIN_SAMPLES_PER_HOUR = 2
MIN_OBSERVED_HOURS = 20
MAX_GAP_HOURS = 2


def parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse source offset timestamps as UTC, then convert to Pune local time."""
    return pd.to_datetime(values, utc=True, errors="coerce").dt.tz_convert(TZ)


def read_station(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["Station ID", "Station Name", "Timestamp", PM], dtype=str)
    df["local_time"] = parse_timestamps(df["Timestamp"])
    df["pm_raw"] = df[PM].fillna("").str.strip()
    df["pm_numeric"] = pd.to_numeric(df["pm_raw"], errors="coerce")
    df["valid"] = df["pm_numeric"].ge(0) & np.isfinite(df["pm_numeric"])
    df["pm25"] = df["pm_numeric"].where(df["valid"])
    df["station"] = df["Station Name"].fillna(path.stem).iloc[0]
    return df


def hourly_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Two or more valid quarter-hour samples produce an observed hour."""
    good = df.loc[df["valid"] & df["local_time"].notna()].copy()
    good["hour_time"] = good["local_time"].dt.floor("h")
    hourly = good.groupby(["station", "hour_time"], as_index=False).agg(
        pm25=("pm25", "mean"), observations=("pm25", "size")
    )
    hourly.loc[hourly["observations"] < MIN_SAMPLES_PER_HOUR, "pm25"] = np.nan
    return hourly


def eligible_day(values: np.ndarray) -> tuple[bool, np.ndarray, np.ndarray]:
    """Require >=20 observed hours; interpolate interior gaps of <=2 hours only."""
    x = np.asarray(values, dtype=float).copy()
    observed = np.isfinite(x)
    if observed.sum() < MIN_OBSERVED_HOURS:
        return False, x, np.zeros(24, dtype=bool)
    missing = np.flatnonzero(~observed)
    if len(missing) == 0:
        return True, x, np.zeros(24, dtype=bool)
    runs = np.split(missing, np.where(np.diff(missing) != 1)[0] + 1)
    if any(len(run) > MAX_GAP_HOURS or run[0] == 0 or run[-1] == 23 for run in runs):
        return False, x, np.zeros(24, dtype=bool)
    filled = ~observed
    x[filled] = np.interp(np.flatnonzero(filled), np.flatnonzero(observed), x[observed])
    return True, x, filled


def build_profiles(hourly: pd.DataFrame, stations: list[str]) -> pd.DataFrame:
    records = []
    for station in stations:
        h = hourly.loc[hourly.station.eq(station) & hourly.hour_time.between(START, END, inclusive="left")].copy()
        h["date"] = h.hour_time.dt.date
        h["hour"] = h.hour_time.dt.hour
        lookup = h.set_index(["date", "hour"])
        for day in pd.date_range(START, END - pd.Timedelta(days=1), freq="D"):
            date = day.date()
            values = np.full(24, np.nan)
            counts = np.zeros(24, dtype=int)
            for hour in range(24):
                if (date, hour) in lookup.index:
                    row = lookup.loc[(date, hour)]
                    values[hour] = row.pm25
                    counts[hour] = row.observations
            ok, profile, filled = eligible_day(values)
            if ok:
                record = {"station": station, "date": str(date), "month": day.month,
                          "weekday": day.day_name(), "weekend": day.dayofweek >= 5,
                          "observed_hours": int(np.isfinite(values).sum()),
                          "filled_hours": int(filled.sum()), "daily_mean": float(profile.mean())}
                record.update({f"h{i:02d}": profile[i] for i in range(24)})
                record.update({f"n{i:02d}": counts[i] for i in range(24)})
                record.update({f"filled_{i:02d}": bool(filled[i]) for i in range(24)})
                records.append(record)
    return pd.DataFrame(records)


def normalize_profiles(x: np.ndarray, eps: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Within-day z scores; profiles with SD <1 µg/m³ become zero vectors."""
    mean = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    flat = (sd[:, 0] < eps)
    z = np.divide(x - mean, sd, out=np.zeros_like(x), where=sd >= eps)
    return z, flat
