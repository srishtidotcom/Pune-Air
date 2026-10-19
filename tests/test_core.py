import numpy as np
import pandas as pd
from pune_air.core import parse_timestamps, hourly_aggregate, eligible_day, normalize_profiles


def test_timestamp_crosses_local_day():
    parsed = parse_timestamps(pd.Series(["2024-01-01T20:00:00.000000+0000", "bad"]))
    assert str(parsed.iloc[0]) == "2024-01-02 01:30:00+05:30"
    assert pd.isna(parsed.iloc[1])


def test_hour_requires_two_valid_observations():
    times = parse_timestamps(pd.Series(["2024-01-01T00:00:00+0000", "2024-01-01T00:15:00+0000", "2024-01-01T01:00:00+0000"]))
    df = pd.DataFrame({"station": ["A"]*3, "local_time": times, "pm25": [10., 30., 50.], "valid": [True]*3})
    h = hourly_aggregate(df)
    assert h.observations.tolist() == [2, 1]
    assert h.pm25.iloc[0] == 20
    assert np.isnan(h.pm25.iloc[1])


def test_day_short_interior_gap_only():
    x = np.arange(24, dtype=float)
    x[8:10] = np.nan
    ok, filled, flags = eligible_day(x)
    assert ok and flags.sum() == 2 and filled[8] == 8 and filled[9] == 9
    x[12:15] = np.nan
    assert not eligible_day(x)[0]
    y = np.arange(24, dtype=float); y[0] = np.nan
    assert not eligible_day(y)[0]


def test_flat_normalization_is_finite():
    z, flat = normalize_profiles(np.array([[10.]*24, list(range(24))]))
    assert flat.tolist() == [True, False]
    assert np.isfinite(z).all() and np.all(z[0] == 0)
