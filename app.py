"""Interactive explorer for the reproducible 2024 Pune PM2.5 analysis."""
from __future__ import annotations

import calendar
import json
from pathlib import Path

import altair as alt
import pandas as pd
import pydeck as pdk
import streamlit as st

from pune_air.dashboard_data import (
    ALL_STATIONS, HOURS, NO_DATA, PALETTES, calendar_frame, filtered_profiles,
    frequency, frequency_matrix, map_markers, selected_day_from_chart, selected_station_from_map,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
LOCATIONS = ROOT / "pune_air" / "station_locations.csv"

st.set_page_config(page_title="Pune daily air patterns", page_icon=":material/air:",
                   layout="wide", initial_sidebar_state="collapsed")


@st.cache_data(max_entries=2, show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict[str, pd.DataFrame]]:
    profiles = pd.read_csv(OUT / "daily_profiles.csv", parse_dates=["date"])
    locations = pd.read_csv(LOCATIONS)
    summary = json.loads((OUT / "summary.json").read_text())
    names = {mode: pd.read_csv(OUT / f"{mode}_cluster_names.csv") for mode in ("absolute", "shape")}
    return profiles, locations, summary, names


def choose_station_from_map() -> None:
    picked = selected_station_from_map(st.session_state.get("station_map"), set(st.session_state.station_options))
    if picked:
        st.session_state.station_select = picked


def curve_chart(data: pd.DataFrame, mode: str, names: dict[int, str]) -> alt.Chart:
    counts = data[f"{mode}_cluster"].value_counts().to_dict()
    medians = data.groupby(f"{mode}_cluster")[HOURS].median().reset_index()
    long = medians.melt(id_vars=f"{mode}_cluster", value_vars=HOURS, var_name="hour", value_name="pm25")
    long["hour"] = long.hour.str[1:].astype(int)
    long["pattern"] = long[f"{mode}_cluster"].map(lambda c: f"{c} · {names[c]}")
    long["eligible_days"] = long[f"{mode}_cluster"].map(counts)
    domain = [f"{c} · {names[c]}" for c in sorted(counts)]
    colors = [PALETTES[mode][c % len(PALETTES[mode])] for c in sorted(counts)]
    return (alt.Chart(long).mark_line(point=True, strokeWidth=3)
            .encode(x=alt.X("hour:Q", title="Hour in Pune (IST)", scale=alt.Scale(domain=[0, 23]),
                            axis=alt.Axis(values=list(range(0, 24, 3)))),
                    y=alt.Y("pm25:Q", title="Median PM2.5 (µg/m³)", scale=alt.Scale(zero=True)),
                    color=alt.Color("pattern:N", scale=alt.Scale(domain=domain, range=colors), title="Pattern"),
                    tooltip=[alt.Tooltip("pattern:N", title="Pattern"), alt.Tooltip("hour:Q", title="Hour"),
                             alt.Tooltip("pm25:Q", title="Median PM2.5", format=".1f"),
                             alt.Tooltip("eligible_days:Q", title="Eligible days")])
            .properties(height=255))


def mix_chart(table: pd.DataFrame, mode: str, names: dict[int, str], group: str) -> alt.Chart:
    view = table.copy()
    view["pattern"] = view.cluster.map(lambda c: f"{c} · {names[c]}")
    if group == "month":
        view["group"] = view.month.map(lambda m: calendar.month_abbr[m])
        sort = list(calendar.month_abbr)[1:]
    elif group == "weekend":
        view["group"] = view.weekend.map({False: "Weekday", True: "Weekend"})
        sort = ["Weekday", "Weekend"]
    else:
        view["group"] = view.station.str.split(",").str[0].str.replace("Panchawati_Pashan", "Panchawati Pashan")
        sort = None
    domain = [f"{c} · {names[c]}" for c in sorted(view.cluster.unique())]
    colors = [PALETTES[mode][c % len(PALETTES[mode])] for c in sorted(view.cluster.unique())]
    return (alt.Chart(view).mark_bar(cornerRadiusEnd=3)
            .encode(x=alt.X("percent:Q", title="Share of eligible days (%)", stack="zero",
                            scale=alt.Scale(domain=[0, 100])),
                    y=alt.Y("group:N", title=None, sort=sort),
                    color=alt.Color("pattern:N", title="Pattern", scale=alt.Scale(domain=domain, range=colors)),
                    tooltip=[alt.Tooltip("group:N", title="Group"), alt.Tooltip("pattern:N", title="Pattern"),
                             alt.Tooltip("days:Q", title="Days"), alt.Tooltip("eligible_days:Q", title="Eligible days"),
                             alt.Tooltip("percent:Q", title="Share (%)", format=".1f")])
            .properties(height=345 if group == "station" else 245 if group == "month" else 120))


def calendar_chart(days: pd.DataFrame, mode: str, names: dict[int, str]) -> alt.LayerChart:
    domain = [names[c] for c in sorted(names)] + ["No eligible profile"]
    colors = [PALETTES[mode][c % len(PALETTES[mode])] for c in sorted(names)] + [NO_DATA]
    pick = alt.selection_point(fields=["date"], name="calendar_pick", on="click", clear="dblclick")
    base = alt.Chart(days).encode(
        x=alt.X("weekday:O", title=None, axis=alt.Axis(labelExpr="['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][datum.value]", labelAngle=0)),
        y=alt.Y("week:O", title=None, axis=None, sort="ascending"),
    )
    cells = (base.mark_rect(cornerRadius=5, stroke="#FBF7F1", strokeWidth=3)
             .encode(color=alt.Color("pattern:N", scale=alt.Scale(domain=domain, range=colors), title="Pattern"),
                     tooltip=[alt.Tooltip("date:N", title="Date"), alt.Tooltip("pattern:N", title="Pattern"),
                              alt.Tooltip("mean_pm:Q", title="Daily PM2.5", format=".1f")])
             .add_params(pick))
    numbers = base.mark_text(color="#342D29", fontSize=13).encode(text="day:Q")
    return (cells + numbers).properties(height=270)


st.caption("PUNE · 2024 · PM2.5 IN µg/m³")
st.title("Pune's daily air patterns")
st.write("Follow the sections to see what a typical day looks like, when each pattern appears, and how results differ across monitoring sites.")
st.caption("Patterns describe measurements, not their causes. Times and dates use Pune local time (IST).")

if not (OUT / "daily_profiles.csv").exists():
    st.error("Analysis files are missing. Run `python -m pune_air.pipeline` first.")
    st.stop()

profiles, locations, summary, all_names = load_data()
station_options = [ALL_STATIONS] + sorted(profiles.station.unique())
st.session_state.station_options = station_options
st.session_state.setdefault("station_select", ALL_STATIONS)

with st.container(border=True):
    st.subheader("1. Choose what to explore", icon=":material/tune:")
    st.caption("Start with all stations and all days, then narrow the view. Selecting a map point also changes the station filter.")
    first_filters = st.columns(2, gap="medium", wrap=True)
    with first_filters[0]:
        station = st.selectbox("Monitoring station", station_options, key="station_select")
    with first_filters[1]:
        mode_label = st.selectbox("Compare patterns by", ["Absolute level + shape", "Normalized shape"], key="mode_filter")
    second_filters = st.columns(2, gap="medium", wrap=True)
    with second_filters[0]:
        month_option = st.selectbox("Month", ["All months"] + list(range(1, 13)),
                                    format_func=lambda m: calendar.month_name[m] if isinstance(m, int) else m,
                                    key="month_filter")
    with second_filters[1]:
        day_type = st.selectbox("Day type", ["All days", "Weekdays", "Weekends"], key="day_type_filter")
    mode = "absolute" if mode_label == "Absolute level + shape" else "shape"
    month = month_option if isinstance(month_option, int) else None
    base = filtered_profiles(profiles, month, day_type)
    focus = filtered_profiles(profiles, month, day_type, station)
    name_rows = all_names[mode]
    names = dict(zip(name_rows.cluster.astype(int), name_rows.name))
    labels = {c: f"{c} · {name}" for c, name in names.items()}
    map_main, map_help = st.columns([3, 1.15], gap="large", wrap=True)
    with map_main:
        markers = map_markers(locations, base, mode, names, station)
        layer = pdk.Layer("ScatterplotLayer", id="stations", data=markers,
                          get_position="[longitude, latitude]", get_fill_color="color",
                          get_line_color="line_color", get_line_width="line_width", get_radius="radius",
                          radius_min_pixels=10, radius_max_pixels=22, line_width_min_pixels=2,
                          stroked=True, filled=True, pickable=True)
        deck = pdk.Deck(
            layers=[layer], map_style=pdk.map_styles.CARTO_LIGHT,
            initial_view_state=pdk.ViewState(latitude=18.565, longitude=73.840, zoom=10.5, pitch=0),
            tooltip={"html": "<b>{station}</b><br/>{eligible_days} eligible days<br/>{pattern}<br/>Median daily PM2.5: {median_pm} µg/m³",
                     "style": {"backgroundColor": "#3D332C", "color": "#FFF9F1"}},
        )
        st.pydeck_chart(deck, key="station_map", on_select=choose_station_from_map,
                        selection_mode="single-object", height=285,
                        alt="Map of 11 Pune monitoring sites; select a marker to focus its PM2.5 results")
    with map_help:
        st.markdown("**How to use the map**")
        st.write("Each point is a monitoring station. Click one to see its results below. Gray means that site has no eligible days under the current month and day filters.")
        st.caption("Points mark stations, not neighborhood boundaries. The basemap needs internet access.")

if focus.empty:
    with st.container(border=True):
        st.warning("No eligible days match this selection. Try another station, month, or day type.")
        st.write("The coverage table shows which months have enough measured hours and eligible days at this station.")
        coverage = pd.read_csv(OUT / "monthly_coverage.csv")
        if station != ALL_STATIONS:
            coverage = coverage[coverage.station.eq(station)]
        st.dataframe(coverage[["station", "month", "eligible_days", "hour_coverage_percent"]],
                     hide_index=True, alt="Monthly eligible days and measured-hour coverage")
    st.stop()

freq = frequency(focus, mode, [])
top = freq.sort_values("days", ascending=False).iloc[0]

with st.container(border=True):
    st.subheader("2. What is in this view?", icon=":material/analytics:")
    summary_main, summary_help = st.columns([3, 1.15], gap="large", wrap=True)
    with summary_main:
        with st.container(horizontal=True, wrap=True, gap="small"):
            st.metric("Eligible station-days", f"{len(focus):,}", border=True)
            st.metric("Median daily PM2.5", f"{focus.daily_mean.median():.1f} µg/m³", border=True)
            st.metric("Most common pattern", f"{top.percent:.1f}%", border=True)
        st.caption(f"{labels[int(top.cluster)]}: {int(top.days):,} of {len(focus):,} eligible station-days. {focus.station.nunique()} station(s) in this view.")
    with summary_help:
        st.markdown("**Read these numbers**")
        st.write("A station-day is one station's 24-hour profile on one date. Eligible days have at least 20 measured hours and only short interior gaps. The median is the middle daily average across this selection.")
        st.caption("Shares use eligible station-days as the denominator, so counts matter when comparing them.")

with st.container(border=True):
    st.subheader("3. What does a typical day look like?", icon=":material/show_chart:")
    curve_main, curve_help = st.columns([3, 1.15], gap="large", wrap=True)
    with curve_main:
        st.altair_chart(curve_chart(focus, mode, names), alt="Median hourly PM2.5 by pattern")
        display = freq.assign(pattern=freq.cluster.map(lambda c: labels[c]),
                              percent=freq.percent.round(1))[["pattern", "days", "eligible_days", "percent"]]
        display.columns = ["Pattern", "Days", "Eligible days", "Share (%)"]
        st.dataframe(display, hide_index=True, alt="Pattern counts, eligible denominators, and percentages")
    with curve_help:
        st.markdown("**How to read the lines**")
        st.write("Each line shows the median PM2.5 at every hour for days in one cluster. A rise shows when that group tends to have higher readings. Hover over a point for its value and group size.")
        st.caption("Absolute analysis groups by level and shape. Normalized analysis groups mainly by the timing and shape of each day's changes. The lines still show measured µg/m³.")

month_data = filtered_profiles(profiles, None, day_type, station)
month_freq = frequency(month_data, mode, ["month"])
month_matrix = frequency_matrix(month_data, mode, "month", names, list(range(1, 13)))
month_matrix.insert(0, "Month", month_matrix.pop("month").map(lambda m: calendar.month_abbr[m]))
with st.container(border=True):
    st.subheader("4. How do patterns vary through the year?", icon=":material/calendar_month:")
    month_main, month_help = st.columns([3, 1.15], gap="large", wrap=True)
    with month_main:
        st.altair_chart(mix_chart(month_freq, mode, names, "month"),
                        alt="Pattern frequency by month, with eligible counts in tooltips")
        st.dataframe(month_matrix, hide_index=True, height=390,
                     alt="Monthly eligible days and counts and shares by pattern")
    with month_help:
        st.markdown("**How to read the bars**")
        st.write("Each bar is the mix of patterns among eligible station-days in that month. The table shows each month's total and each group's count and share, such as 20 days (40%).")
        st.caption("All months stay visible for comparison, even if you selected one above. Month-to-month changes can reflect which stations supplied data.")

weekday_data = filtered_profiles(profiles, month, "All days", station)
weekday_freq = frequency(weekday_data, mode, ["weekend"])
weekday_matrix = frequency_matrix(weekday_data, mode, "weekend", names)
weekday_matrix.insert(0, "Day", weekday_matrix.pop("weekend").map({False: "Weekday", True: "Weekend"}))
with st.container(border=True):
    st.subheader("5. Are weekdays and weekends different?", icon=":material/date_range:")
    weekday_main, weekday_help = st.columns([3, 1.15], gap="large", wrap=True)
    with weekday_main:
        st.altair_chart(mix_chart(weekday_freq, mode, names, "weekend"),
                        alt="Pattern frequency for weekdays and weekends")
        st.dataframe(weekday_matrix, hide_index=True,
                     alt="Weekday and weekend eligible counts and pattern shares")
    with weekday_help:
        st.markdown("**Compare the shares and counts**")
        st.write("Each bar totals 100% of eligible days of that type. Use the table to see how many days are behind each percentage.")
        st.caption("Both day types remain visible even if one is selected above. A difference in shares does not establish a weekday cause.")

station_freq = frequency(base, mode, ["station"])
station_matrix = frequency_matrix(base, mode, "station", names, locations.station.tolist())
station_matrix["station"] = station_matrix.station.str.replace(", Pune - IITM", "", regex=False).str.replace(", Pune - MPCB", "", regex=False).str.replace("Panchawati_Pashan", "Panchawati Pashan", regex=False)
with st.container(border=True):
    st.subheader("6. How do stations compare?", icon=":material/stacked_bar_chart:")
    station_main, station_help = st.columns([3, 1.15], gap="large", wrap=True)
    with station_main:
        st.altair_chart(mix_chart(station_freq, mode, names, "station"),
                        alt="Pattern shares across monitoring stations")
        st.dataframe(station_matrix, hide_index=True, height=400,
                     alt="Station eligible days and pattern counts and percentages")
    with station_help:
        st.markdown("**Compare sites carefully**")
        st.write("A station's bar shows its own eligible-day mix under the month and day filters. The table gives the number of days behind every share.")
        st.caption("Stations have different amounts of data. A short bar sample may be less representative of the year.")

with st.container(border=True):
    st.subheader("7. Look at an individual day", icon=":material/today:")
    calendar_main, calendar_help = st.columns([3, 1.15], gap="large", wrap=True)
    with calendar_main:
        if station == ALL_STATIONS:
            default_site = "Revenue Colony-Shivajinagar, Pune - IITM"
            calendar_station = st.selectbox("Calendar station", station_options[1:],
                                            index=station_options[1:].index(default_site), key="calendar_station")
        else:
            calendar_station = station
            st.caption(f"Calendar station: {calendar_station}")
        cal_month = month if month else st.selectbox("Calendar month", list(range(1, 13)),
                                                      format_func=lambda m: calendar.month_name[m],
                                                      key="calendar_month")
        calendar_profiles = filtered_profiles(profiles, month, day_type, calendar_station)
        days = calendar_frame(calendar_profiles[calendar_profiles.month.eq(cal_month)], mode, 2024, cal_month, names)
        cal_event = st.altair_chart(calendar_chart(days, mode, names), key="calendar_chart",
                                    on_select="rerun", alt=f"{calendar.month_name[cal_month]} 2024 calendar colored by PM2.5 pattern")
        picked_date = selected_day_from_chart(cal_event)
        if picked_date:
            chosen = calendar_profiles[calendar_profiles.date.eq(pd.Timestamp(picked_date))]
            if chosen.empty:
                st.caption("This date has no eligible profile under the current filters.")
            else:
                day = chosen.iloc[0]
                st.markdown(f"**{picked_date} · {labels[int(day[f'{mode}_cluster'])]}**")
                with st.container(horizontal=True, wrap=True):
                    st.metric("Daily mean", f"{day.daily_mean:.1f} µg/m³", border=True)
                    st.metric("Measured hours", f"{int(day.observed_hours)}/24", border=True)
                    st.metric("Filled hours", f"{int(day.filled_hours)}", border=True)
                hourly = pd.DataFrame({"hour": list(range(24)), "PM2.5 (µg/m³)": [day[h] for h in HOURS]})
                st.line_chart(hourly, x="hour", y="PM2.5 (µg/m³)",
                              alt=f"Hourly PM2.5 at {calendar_station} on {picked_date}")
    with calendar_help:
        st.markdown("**Use the calendar**")
        st.write("A colored day has a complete enough 24-hour profile and belongs to the matching pattern. Click a date to inspect its hourly readings and how many hours were filled.")
        st.caption("Blank dates have no eligible profile under the current filters. Filled hours were estimated from neighboring measured hours; they were not observed readings.")

with st.container(border=True):
    st.subheader("8. Check coverage and download", icon=":material/data_usage:")
    coverage_main, coverage_help = st.columns([3, 1.15], gap="large", wrap=True)
    with coverage_main:
        if station == ALL_STATIONS:
            audit = pd.read_csv(OUT / "station_audit.csv")
            coverage_view = audit[["station", "eligible_2024_days", "months_with_10_days"]].copy()
            coverage_view.columns = ["Station", "Eligible days", "Months with ≥10 days"]
        else:
            coverage = pd.read_csv(OUT / "monthly_coverage.csv")
            coverage_view = coverage[coverage.station.eq(station)][
                ["month", "eligible_days", "hour_coverage_percent"]].copy()
            coverage_view["month"] = coverage_view.month.map(lambda m: calendar.month_abbr[m])
            coverage_view.columns = ["Month", "Eligible days", "Measured hours (%)"]
        st.dataframe(coverage_view, hide_index=True, height=330,
                     alt="Measured 2024 coverage and eligible-day counts")
        st.download_button("Download selected station-days as CSV", focus.to_csv(index=False).encode("utf-8"),
                           file_name="pune_selected_station_days.csv", mime="text/csv", icon=":material/download:")
    with coverage_help:
        st.markdown("**Why coverage matters**")
        st.write("The analysis includes only station-days that pass the completeness rules. A month with no eligible days cannot contribute to a pattern percentage.")
        st.caption("The CSV keeps each hourly value, source observation count, and gap-fill flag. The root clustering.md file explains the method in full.")

with st.expander("Method and limitations", icon=":material/info:"):
    st.markdown("Hourly values require at least two valid 15-minute readings. A day requires at least 20 measured hours; only interior gaps of up to two hours are filled. Absolute clustering reflects level and shape. Within-day normalization emphasizes shape, with nearly flat days handled safely. Frequency percentages describe **eligible days only**. Uneven coverage can affect comparisons. Patterns do not identify pollution sources or causes. See the root `clustering.md` file for details.")
