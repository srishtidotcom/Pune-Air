from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from pune_air.dashboard_data import (
    ALL_STATIONS, calendar_frame, filtered_profiles, frequency, frequency_matrix, map_markers,
    selected_day_from_chart, selected_station_from_map,
)

ROOT = Path(__file__).resolve().parents[1]


def test_map_selection_and_filter_denominators():
    profiles = pd.DataFrame({
        "station": ["A", "A", "B"], "month": [1, 1, 1],
        "weekend": [False, True, False], "absolute_cluster": [0, 1, 0],
        "daily_mean": [10., 20., 30.],
    })
    locations = pd.DataFrame({"station": ["A", "B", "C"],
                              "latitude": [18.5, 18.6, 18.7], "longitude": [73.8, 73.9, 74.0]})
    weekdays = filtered_profiles(profiles, 1, "Weekdays", "A")
    assert len(weekdays) == 1
    shares = frequency(profiles, "absolute", ["station"])
    assert shares[shares.station.eq("A")].eligible_days.eq(2).all()
    assert shares[shares.station.eq("A")].percent.sum() == 100
    matrix = frequency_matrix(profiles, "absolute", "station", {0: "low", 1: "high"}, ["A", "B", "C"])
    assert matrix.loc[matrix.station.eq("A"), "Group 1"].iloc[0] == "1 (50.0%)"
    assert matrix.loc[matrix.station.eq("C"), "eligible_days"].iloc[0] == 0
    assert matrix.loc[matrix.station.eq("C"), "Group 0"].iloc[0] == "—"
    markers = map_markers(locations, profiles, "absolute", {0: "low", 1: "high"}, ALL_STATIONS)
    assert markers.eligible_days.tolist() == [2, 1, 0]
    event = {"selection": {"objects": {"stations": [{"station": "B"}]}}}
    assert selected_station_from_map(event, {"A", "B"}) == "B"
    assert selected_station_from_map(event, {"A"}) is None


def test_calendar_keeps_unavailable_dates_and_day_selection():
    data = pd.DataFrame({"date": pd.to_datetime(["2024-02-03"]),
                         "shape_cluster": [1], "daily_mean": [42.]})
    days = calendar_frame(data, "shape", 2024, 2, {0: "overnight", 1: "afternoon"})
    assert len(days) == 29
    assert days.loc[days.day.eq(3), "pattern"].iloc[0] == "afternoon"
    assert days.loc[days.day.eq(4), "pattern"].iloc[0] == "No eligible profile"
    assert selected_day_from_chart({"selection": {"calendar_pick": [{"date": "2024-02-03"}]}}) == "2024-02-03"


def test_dashboard_station_filter_and_empty_state():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
    assert not at.exception
    assert at.metric[0].value == "2,408"
    assert len(at.get("vega_lite_chart")) == 5  # curves, month, weekday, station, calendar
    assert len(at.dataframe) >= 5  # counts and coverage remain visible without switching views
    at.selectbox(key="station_select").select("Hadapsar, Pune - IITM").run()
    assert not at.exception
    assert at.metric[0].value == "99"
    at.selectbox(key="month_filter").select(1).run()
    assert not at.exception
    assert at.warning
