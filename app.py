"""Interactive explorer for batch-distillation sensor recordings and annotations.

Run with: streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yaml

from sensors import REQUIRED_SENSORS


st.set_page_config(page_title="Batch Distillation Sensor Explorer", layout="wide")

DATA_DIR = Path(__file__).with_name("batch_distillation_dataset")
CLOCK_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}$")
SENSOR_DESCRIPTIONS = {
    "FT703": "Flow in the top product stream",
    "FT704": "Flow in the reflux stream",
    "FYI702": "Cooling-water flow signal",
    "PDI701": "Pressure difference along the column",
    "PDI702": "Pressure difference associated with buffer vessel V002",
    "PY23": "Pressure in the top vessel G001",
    "T703": "Liquid temperature in the reboiler vessel V001",
    "T705": "Temperature at the top of column section C003",
    "T709": "Temperature below column section C001",
    "T711": "Temperature between column sections C001 and C002",
    "T712": "Temperature between column sections C002 and C003",
}


def recording_type(path: Path) -> str:
    """Use the dataset's original `anormal` spelling for anomalous recordings."""
    if "anormal" in path.stem:
        return "Anomalous"
    if "normal" in path.stem:
        return "Normal"
    return "Unknown"


def parse_clock(value: object) -> int | None:
    """Strictly parse a YAML/CSV HH:MM:SS value without changing it."""
    if not isinstance(value, str) or not CLOCK_PATTERN.fullmatch(value):
        return None
    hours, minutes, seconds = map(int, value.split(":"))
    if hours > 23 or minutes > 59 or seconds > 59:
        return None
    return hours * 3600 + minutes * 60 + seconds


def as_list(value: object) -> list[object]:
    return value if isinstance(value, list) else [value]


def annotation_details(entry: dict) -> tuple[str | None, str | None, str | None]:
    mode = entry.get("PerturbationMode") or {}
    local_effect = mode.get("hasLocalEffect", {}).get("LocalEffect", {})
    sensor = local_effect.get("madeBySensor", {}).get("Sensor", {}).get("id")
    system = mode.get("happenedAt", {}).get("System", {})
    return sensor, system.get("id"), local_effect.get("label")


@st.cache_data(show_spinner=False)
def load_recording(path: str) -> pd.DataFrame:
    """Load just Time and the project's selected sensors for visualization."""
    return pd.read_csv(path, usecols=["Time", *REQUIRED_SENSORS])


@st.cache_data(show_spinner=False)
def load_annotation_events(annotation_path: str, recording_path: str) -> list[dict]:
    """Expand scalar and recurrent YAML intervals while retaining unresolved rows."""
    recording_times = pd.read_csv(recording_path, usecols=["Time"])["Time"].astype(str).tolist()
    time_index = {value: index for index, value in enumerate(recording_times)}

    with Path(annotation_path).open(encoding="utf-8") as handle:
        anomalies = (yaml.safe_load(handle) or {}).get("anomalies", [])

    events: list[dict] = []
    for entry_index, entry in enumerate(anomalies, start=1):
        starts, ends = as_list(entry.get("hasBeginning")), as_list(entry.get("hasEnd"))
        local_sensor, equipment_id, local_effect = annotation_details(entry)
        list_mismatch = len(starts) != len(ends)

        for occurrence_index in range(1, max(len(starts), len(ends), 1) + 1):
            start = starts[occurrence_index - 1] if occurrence_index <= len(starts) else None
            end = ends[occurrence_index - 1] if occurrence_index <= len(ends) else None
            issue = None
            start_index = end_index = None

            if parse_clock(start) is None or parse_clock(end) is None:
                issue = "Start or end is missing or not a literal HH:MM:SS timestamp."
            elif start not in time_index or end not in time_index:
                issue = "At least one event endpoint does not occur in this recording."
            else:
                start_index, end_index = time_index[start], time_index[end]
                if end_index < start_index:
                    issue = "The end precedes the start on this recording timeline; it is not shaded."

            state = "valid" if issue is None else "unresolved"

            events.append(
                {
                    "key": f"entry-{entry_index}-occurrence-{occurrence_index}",
                    "entry_index": entry_index,
                    "occurrence_index": occurrence_index,
                    "anomaly_id": entry.get("id"),
                    "category": entry.get("hasCategory"),
                    "start": start,
                    "end": end,
                    "start_index": start_index,
                    "end_index": end_index,
                    "state": state,
                    "valid": state == "valid",
                    "issue": issue,
                    "list_mismatch": list_mismatch,
                    "affected_sensor": local_sensor,
                    "equipment_id": equipment_id,
                    "local_effect": local_effect,
                    "perturbation": entry.get("Perturbation"),
                    "perturbation_mode_start": (entry.get("PerturbationMode") or {}).get("hasBeginning"),
                    "perturbation_mode_end": (entry.get("PerturbationMode") or {}).get("hasEnd"),
                }
            )
    return events


def event_label(event: dict) -> str:
    sensor = event["affected_sensor"] or "sensor unspecified"
    status = event["state"]
    return (
        f"Entry {event['entry_index']} · occurrence {event['occurrence_index']} · "
        f"{event['anomaly_id'] or 'missing ID'} · {sensor} · "
        f"{event['start']}–{event['end']} [{status}]"
    )


def event_by_key(events: list[dict], key: str) -> dict | None:
    return next((event for event in events if event["key"] == key), None)


def annotation_state_counts(events: list[dict]) -> dict[str, int]:
    """Counts are based only on parser state, never on the chart's zoom range."""
    valid = sum(event["state"] == "valid" for event in events)
    unresolved = sum(event["state"] == "unresolved" for event in events)
    return {"total": len(events), "valid": valid, "unresolved": unresolved}


def show_unresolved_warning(events: list[dict]) -> None:
    counts = annotation_state_counts(events)
    if counts["unresolved"]:
        st.warning(
            f"{counts['unresolved']} of {counts['total']} annotation intervals could not be mapped reliably and are not shaded."
        )


def build_chart(data: pd.DataFrame, sensor: str, events: list[dict], selected_event: dict | None) -> go.Figure:
    chart_data = data.copy()
    chart_data["plot_time"] = pd.to_datetime(chart_data["Time"], format="%H:%M:%S")
    figure = go.Figure(
        go.Scatter(
            x=chart_data["plot_time"],
            y=chart_data[sensor],
            mode="lines",
            name=sensor,
            line=dict(color="#1f77b4", width=1.5),
            hovertemplate="Time: %{x|%H:%M:%S}<br>Value: %{y}<extra></extra>",
        )
    )

    for event in events:
        if not event["valid"]:
            continue
        is_selected = selected_event is not None and event["key"] == selected_event["key"]
        figure.add_vrect(
            x0=chart_data["plot_time"].iloc[event["start_index"]],
            x1=chart_data["plot_time"].iloc[event["end_index"]],
            fillcolor="#d62728" if is_selected else "#ff9896",
            opacity=0.22 if is_selected else 0.12,
            line_width=0,
            layer="below",
        )

    if selected_event is not None and selected_event["valid"]:
        context_seconds = max(120, min(600, selected_event["end_index"] - selected_event["start_index"]))
        left = max(0, selected_event["start_index"] - context_seconds)
        right = min(len(chart_data) - 1, selected_event["end_index"] + context_seconds)
        figure.update_xaxes(range=[chart_data["plot_time"].iloc[left], chart_data["plot_time"].iloc[right]])

    figure.update_layout(
        title=f"{sensor} — {SENSOR_DESCRIPTIONS[sensor]}",
        xaxis_title="Recording time",
        yaxis_title=sensor,
        height=500,
        margin=dict(l=30, r=30, t=60, b=40),
        hovermode="x unified",
        showlegend=False,
    )
    figure.update_xaxes(tickformat="%H:%M:%S")
    return figure


def elapsed_minutes(data: pd.DataFrame) -> pd.Series:
    """Elapsed time calculated separately from the start of this recording."""
    timestamps = pd.to_datetime(data["Time"], format="%H:%M:%S")
    seconds = timestamps.dt.hour * 3600 + timestamps.dt.minute * 60 + timestamps.dt.second
    gaps = seconds.diff().fillna(0)
    gaps = gaps.where(gaps >= 0, gaps + 24 * 60 * 60)
    return gaps.cumsum() / 60


def annotation_path_for(op: str, recording_path: Path) -> Path:
    return DATA_DIR / op / "annotations" / f"{recording_path.stem}.yaml"


def events_for(op: str, recording_path: Path) -> list[dict]:
    path = annotation_path_for(op, recording_path)
    return load_annotation_events(str(path), str(recording_path)) if path.exists() else []


@st.cache_data(show_spinner=False)
def anomaly_catalog(op: str) -> list[dict]:
    """Group annotation entries by recording and anomaly ID without losing observations."""
    grouped: dict[str, dict] = {}
    for recording_path in sorted((DATA_DIR / op / "measurements").glob("*.csv")):
        if recording_type(recording_path) != "Anomalous":
            continue
        for event in events_for(op, recording_path):
            anomaly_id = event["anomaly_id"] or "missing-id"
            key = f"{recording_path.stem}::{anomaly_id}"
            card = grouped.setdefault(
                key,
                {
                    "key": key,
                    "recording_path": str(recording_path),
                    "recording_stem": recording_path.stem,
                    "anomaly_id": event["anomaly_id"],
                    "events": [],
                    "categories": set(),
                    "sensors": set(),
                    "equipment": set(),
                    "perturbations": set(),
                    "local_effects": set(),
                },
            )
            card["events"].append(event)
            if event["category"]:
                card["categories"].add(event["category"])
            if event["affected_sensor"]:
                card["sensors"].add(event["affected_sensor"])
            if event["equipment_id"]:
                card["equipment"].add(event["equipment_id"])
            if event["perturbation"]:
                card["perturbations"].add(event["perturbation"])
            if event["local_effect"]:
                card["local_effects"].add(event["local_effect"])

    cards = []
    for card in grouped.values():
        card["categories"] = sorted(card["categories"])
        card["sensors"] = sorted(card["sensors"])
        card["equipment"] = sorted(card["equipment"])
        card["perturbations"] = sorted(card["perturbations"])
        card["local_effects"] = sorted(card["local_effects"])
        cards.append(card)
    return sorted(cards, key=lambda card: (card["recording_stem"], card["anomaly_id"] or ""))


def anomaly_counts_by_op(operating_points: list[str]) -> dict[str, int]:
    """Distinct YAML anomaly IDs, used only for treemap rectangle area."""
    return {
        op: len({card["anomaly_id"] for card in anomaly_catalog(op) if card["anomaly_id"] is not None})
        for op in operating_points
    }


def build_operating_point_treemap(counts: dict[str, int], selected_op: str) -> go.Figure:
    operating_points = list(counts)
    colors = ["#0f766e" if op == selected_op else "#93c5fd" for op in operating_points]
    figure = go.Figure(
        go.Treemap(
            ids=operating_points,
            labels=operating_points,
            parents=[""] * len(operating_points),
            values=[counts[op] for op in operating_points],
            customdata=operating_points,
            marker=dict(colors=colors, line=dict(color="white", width=3), pad=dict(t=12, l=10, r=10, b=10)),
            texttemplate="<b>%{label}</b><br>%{value} anomalies",
            hovertemplate="%{label}<br>%{value} distinct annotated anomaly IDs<extra></extra>",
            root_color="white",
            tiling=dict(packing="squarify"),
        )
    )
    figure.update_traces(textfont=dict(size=17, color="#0f172a"))
    figure.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0))
    return figure


def selected_treemap_op(selection: object, operating_points: list[str]) -> str | None:
    """Accept Streamlit Plotly selection data without depending on a JS bridge."""
    try:
        points = selection.selection.points
    except AttributeError:
        try:
            points = selection.get("selection", {}).get("points", [])
        except AttributeError:
            return None
    for point in points:
        candidate = point.get("id") or point.get("customdata") or point.get("label")
        if candidate in operating_points:
            return candidate
    return None


def short_text(value: str, limit: int = 130) -> str:
    return value if len(value) <= limit else f"{value[: limit - 1].rstrip()}…"


def short_anomaly_id(anomaly_id: str | None) -> str:
    """Return the compact trailing A-number without changing the source ID."""
    match = re.search(r"(A\d+)$", anomaly_id or "")
    return match.group(1) if match else (anomaly_id or "—")


def show_event_info(event: dict) -> None:
    """Present only what the selected annotation documents."""
    st.subheader("Selected annotation")
    if not event["valid"]:
        st.warning(event["issue"])
    if event["list_mismatch"]:
        st.caption("This entry has different numbers of start and end values; it is retained without correction.")

    left, middle, right = st.columns(3)
    left.metric("Anomaly ID", event["anomaly_id"] or "Missing")
    middle.metric("Category", event["category"] or "Not provided")
    right.metric("Documented sensor", event["affected_sensor"] or "Not specified")
    st.write(f"**Annotated event interval:** {event['start']} → {event['end']}")
    st.write(f"**Perturbation:** {event['perturbation'] or 'Not provided'}")
    st.write(f"**Documented local effect:** {event['local_effect'] or 'Not provided'}")
    st.caption(
        "PerturbationMode application time: "
        f"{event['perturbation_mode_start']} → {event['perturbation_mode_end']}"
    )
    if event["equipment_id"]:
        st.caption(f"Documented equipment/valve: {event['equipment_id']}")


def build_comparison_chart(
    normal_data: pd.DataFrame,
    anomalous_data: pd.DataFrame,
    sensor: str,
    anomalous_events: list[dict],
    normal_name: str,
    anomalous_name: str,
) -> go.Figure:
    """Two independent elapsed-time panels; no physical-stage equivalence is implied."""
    normal_elapsed = elapsed_minutes(normal_data)
    anomalous_elapsed = elapsed_minutes(anomalous_data)
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        subplot_titles=(f"Normal — {normal_name}", f"Anomalous — {anomalous_name}"),
    )
    figure.add_trace(
        go.Scatter(x=normal_elapsed, y=normal_data[sensor], mode="lines", line=dict(color="#1f77b4"), name="Normal"),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(x=anomalous_elapsed, y=anomalous_data[sensor], mode="lines", line=dict(color="#d62728"), name="Anomalous"),
        row=2,
        col=1,
    )
    for event in anomalous_events:
        if event["valid"]:
            figure.add_vrect(
                x0=anomalous_elapsed.iloc[event["start_index"]],
                x1=anomalous_elapsed.iloc[event["end_index"]],
                fillcolor="#ff9896",
                opacity=0.16,
                line_width=0,
                layer="below",
                row=2,
                col=1,
            )
    figure.update_layout(
        title=f"{sensor} — normal vs anomalous recordings",
        height=700,
        margin=dict(l=30, r=30, t=80, b=45),
        showlegend=False,
        hovermode="x unified",
    )
    figure.update_yaxes(title_text=sensor, row=1, col=1)
    figure.update_yaxes(title_text=sensor, row=2, col=1)
    figure.update_xaxes(title_text="Elapsed minutes from this recording's start", row=1, col=1)
    figure.update_xaxes(title_text="Elapsed minutes from this recording's start", row=2, col=1)
    return figure


def build_multi_sensor_chart(
    data: pd.DataFrame,
    sensors: list[str],
    event: dict,
    context_seconds: int,
) -> go.Figure:
    """Stacked sensor panels with a shared recording-time axis and one selected event."""
    chart_data = data.copy()
    chart_data["plot_time"] = pd.to_datetime(chart_data["Time"], format="%H:%M:%S")
    if event["valid"]:
        left = max(0, event["start_index"] - context_seconds)
        right = min(len(chart_data) - 1, event["end_index"] + context_seconds)
        chart_data = chart_data.iloc[left : right + 1]

    figure = make_subplots(rows=len(sensors), cols=1, shared_xaxes=True, vertical_spacing=0.06)
    for row, sensor in enumerate(sensors, start=1):
        figure.add_trace(
            go.Scatter(
                x=chart_data["plot_time"],
                y=chart_data[sensor],
                mode="lines",
                line=dict(color="#1f77b4", width=1.5),
                name=sensor,
                hovertemplate="Time: %{x|%H:%M:%S}<br>Value: %{y}<extra></extra>",
            ),
            row=row,
            col=1,
        )
        figure.update_yaxes(title_text=sensor, row=row, col=1)

    if event["valid"]:
        source_data = data.copy()
        source_data["plot_time"] = pd.to_datetime(source_data["Time"], format="%H:%M:%S")
        figure.add_vrect(
            x0=source_data["plot_time"].iloc[event["start_index"]],
            x1=source_data["plot_time"].iloc[event["end_index"]],
            fillcolor="#d62728",
            opacity=0.18,
            line_width=0,
            layer="below",
            row="all",
            col=1,
        )

    figure.update_layout(
        title="Multi-sensor event view",
        height=max(420, 250 * len(sensors)),
        margin=dict(l=30, r=30, t=60, b=40),
        showlegend=False,
        hovermode="x unified",
    )
    figure.update_xaxes(title_text="Recording time", tickformat="%H:%M:%S", row=len(sensors), col=1)
    return figure


def robust_baseline_response(values: pd.Series, baseline: pd.Series) -> tuple[pd.Series, str]:
    """Scale against pre-event data only, using robust median/MAD safeguards.

    The normal case is a robust z-like response: (x - median) / (1.4826 * MAD).
    With a nearly constant baseline, the fallback scale is 1% of the absolute
    pre-event median (or 1e-6), never data from during/after the event.
    """
    finite_baseline = baseline.replace([np.inf, -np.inf], np.nan).dropna()
    if finite_baseline.empty:
        return pd.Series(np.nan, index=values.index), "no pre-event baseline"

    median = finite_baseline.median()
    mad = (finite_baseline - median).abs().median()
    robust_scale = 1.4826 * mad
    if not np.isfinite(robust_scale) or robust_scale <= 1e-9:
        robust_scale = max(abs(median) * 0.01, 1e-6)
        method = "near-constant fallback scale"
    else:
        method = "median/MAD scale"
    return (values - median) / robust_scale, method


def merged_valid_intervals(events: list[dict]) -> list[tuple[int, int]]:
    """Merge only adjoining/overlapping valid spans for legible chart shading."""
    spans = sorted((event["start_index"], event["end_index"]) for event in events if event["valid"])
    merged: list[list[int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def event_view_data(data: pd.DataFrame, focus_event: dict, context_seconds: int, full_timeline: bool) -> pd.DataFrame:
    if full_timeline:
        view = data.copy()
    else:
        left = max(0, focus_event["start_index"] - context_seconds)
        right = min(len(data) - 1, focus_event["end_index"] + context_seconds)
        view = data.iloc[left : right + 1].copy()
    view["plot_time"] = pd.to_datetime(view["Time"], format="%H:%M:%S")
    return view


def overview_render_data(data: pd.DataFrame, full_timeline: bool, max_points: int = 2_500) -> tuple[pd.DataFrame, int]:
    """Limit only the dense full-recording overview; detailed views retain every sample."""
    if not full_timeline or len(data) <= max_points:
        return data, 1
    stride = int(np.ceil(len(data) / max_points))
    sampled = pd.concat([data.iloc[::stride], data.iloc[[-1]]]).drop_duplicates(subset="Time")
    return sampled, stride


def add_event_regions(figure: go.Figure, event_times: pd.Series, events: list[dict], row: int | str | None = None) -> None:
    """Shade original valid intervals once, even if YAML entries overlap."""
    # A paper-height shape spans a shared-axis overview without multiplying each
    # recurrent interval by all eleven subplot rows. Adding them as one layout
    # update also keeps long recurrent recordings responsive.
    if isinstance(row, int):
        xref = "x" if row == 1 else f"x{row}"
        yref = "y domain" if row == 1 else f"y{row} domain"
    else:
        xref, yref = "x", "paper"
    shapes = [
        dict(
            type="rect", xref=xref, yref=yref,
            x0=event_times.iloc[start_index], x1=event_times.iloc[end_index], y0=0, y1=1,
            fillcolor="#dc2626", opacity=0.15, line=dict(width=0), layer="below",
        )
        for start_index, end_index in merged_valid_intervals(events)
    ]
    figure.update_layout(shapes=[*figure.layout.shapes, *shapes])


def build_response_overview(
    data: pd.DataFrame,
    events: list[dict],
    documented_sensors: list[str],
    context_seconds: int,
    full_timeline: bool,
    baseline_seconds: int = 300,
    normal_reference: pd.DataFrame | None = None,
) -> tuple[go.Figure, list[str]]:
    """Eleven independently scaled sensor rows for valid card-level intervals."""
    focus_event = min(events, key=lambda event: event["start_index"])
    display_data = event_view_data(data, focus_event, context_seconds, full_timeline)
    render_data, render_stride = overview_render_data(display_data, full_timeline)
    baseline_start = max(0, focus_event["start_index"] - baseline_seconds)
    baseline_data = data.iloc[baseline_start : focus_event["start_index"]]

    figure = make_subplots(rows=len(REQUIRED_SENSORS), cols=1, shared_xaxes=True, vertical_spacing=0.018)
    methods = []
    for row, sensor in enumerate(REQUIRED_SENSORS, start=1):
        pre_event_values = baseline_data[sensor]
        if pre_event_values.replace([np.inf, -np.inf], np.nan).dropna().empty and normal_reference is not None:
            scaled, reference_method = robust_baseline_response(render_data[sensor], normal_reference[sensor])
            method = f"normal-recording reference ({reference_method})"
        else:
            scaled, method = robust_baseline_response(render_data[sensor], pre_event_values)
        methods.append(f"{sensor}: {method}")
        documented = sensor in documented_sensors
        figure.add_trace(
            go.Scatter(
                x=render_data["plot_time"],
                y=scaled,
                customdata=render_data[sensor],
                mode="lines",
                line=dict(color="#dc2626" if documented else "#2563eb", width=2.1 if documented else 1.05),
                hovertemplate=(
                    "Time: %{x|%H:%M:%S}<br>Baseline-relative response: %{y:.3f}"
                    "<br>Original value: %{customdata}<extra>" + sensor + "</extra>"
                ),
            ),
            row=row,
            col=1,
        )
        figure.update_yaxes(title_text=sensor, row=row, col=1, zeroline=True, zerolinecolor="#d1d5db")

    event_times = pd.to_datetime(data["Time"], format="%H:%M:%S")
    add_event_regions(figure, event_times, events, row="all")
    figure.update_layout(
        height=1060,
        margin=dict(l=45, r=24, t=16, b=38),
        showlegend=False,
        hovermode="x unified",
    )
    figure.update_xaxes(title_text="Recording time", tickformat="%H:%M:%S", row=len(REQUIRED_SENSORS), col=1)
    return figure, methods, render_stride


def build_detailed_sensor_chart(
    data: pd.DataFrame,
    sensor: str,
    events: list[dict],
    context_seconds: int,
    full_timeline: bool,
) -> go.Figure:
    """Original-unit view for either one focused event or all recurrent intervals."""
    focus_event = min(events, key=lambda event: event["start_index"])
    chart_data = event_view_data(data, focus_event, context_seconds, full_timeline)
    event_times = pd.to_datetime(data["Time"], format="%H:%M:%S")
    figure = go.Figure(
        go.Scatter(
            x=chart_data["plot_time"], y=chart_data[sensor], mode="lines",
            line=dict(color="#2563eb", width=1.6),
            hovertemplate="Time: %{x|%H:%M:%S}<br>Value: %{y}<extra></extra>",
        )
    )
    add_event_regions(figure, event_times, events)
    if not full_timeline and len(events) == 1:
        figure.add_vline(
            x=event_times.iloc[focus_event["start_index"]], line_color="#b91c1c", line_dash="dot",
            annotation_text=f"Start {focus_event['start']}", annotation_position="top left",
        )
        figure.add_vline(
            x=event_times.iloc[focus_event["end_index"]], line_color="#b91c1c", line_dash="dot",
            annotation_text=f"End {focus_event['end']}", annotation_position="top right",
        )
    figure.update_layout(
        title=f"Anomalous recording — {sensor}", xaxis_title="Recording time", yaxis_title=sensor,
        height=420, margin=dict(l=40, r=24, t=50, b=40), hovermode="x unified",
    )
    figure.update_xaxes(tickformat="%H:%M:%S")
    return figure


def build_sensor_comparison_chart(
    anomalous_data: pd.DataFrame,
    normal_data: pd.DataFrame,
    sensor: str,
    events: list[dict],
    context_seconds: int,
    full_timeline: bool,
    normal_name: str,
) -> go.Figure:
    """Original-unit anomalous and normal views without a false time alignment."""
    focus_event = min(events, key=lambda event: event["start_index"])
    anomalous_view = event_view_data(anomalous_data, focus_event, context_seconds, full_timeline)
    anomalous_times = pd.to_datetime(anomalous_data["Time"], format="%H:%M:%S")
    figure = make_subplots(
        rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.13,
        subplot_titles=(f"Anomalous recording — {sensor}", f"Normal recording — {normal_name}"),
    )
    figure.add_trace(
        go.Scatter(x=anomalous_view["plot_time"], y=anomalous_view[sensor], mode="lines", line=dict(color="#2563eb", width=1.6)),
        row=1, col=1,
    )
    figure.add_trace(
        go.Scatter(x=elapsed_minutes(normal_data), y=normal_data[sensor], mode="lines", line=dict(color="#64748b", width=1.5)),
        row=2, col=1,
    )
    add_event_regions(figure, anomalous_times, events, row=1)
    figure.update_layout(height=680, margin=dict(l=42, r=24, t=58, b=38), showlegend=False, hovermode="x unified")
    figure.update_yaxes(title_text=sensor, row=1, col=1)
    figure.update_yaxes(title_text=sensor, row=2, col=1)
    figure.update_xaxes(title_text="Recording time", tickformat="%H:%M:%S", row=1, col=1)
    figure.update_xaxes(title_text="Elapsed minutes from normal recording start", row=2, col=1)
    return figure


def build_normal_comparison_chart(normal_data: pd.DataFrame, sensor: str, normal_name: str) -> go.Figure:
    """Optional separate normal-run reference using its own elapsed-time origin."""
    figure = go.Figure(
        go.Scatter(
            x=elapsed_minutes(normal_data),
            y=normal_data[sensor],
            mode="lines",
            line=dict(color="#64748b", width=1.5),
            hovertemplate="Elapsed minutes: %{x:.2f}<br>Value: %{y}<extra></extra>",
        )
    )
    figure.update_layout(
        title=f"Normal-operation reference — {normal_name}",
        xaxis_title="Elapsed minutes from normal recording start",
        yaxis_title=sensor,
        height=340,
        margin=dict(l=40, r=30, t=55, b=40),
        hovermode="x unified",
    )
    return figure


def clear_sensor_detail() -> None:
    """Clear detail widgets before Streamlit rebuilds them on the next rerun."""
    st.session_state["detail_sensor_choice"] = None
    st.session_state["compare_with_normal"] = False
    st.session_state.pop("normal_recording_choice", None)


def select_anomaly_card(card_key: str) -> None:
    """Use a native button callback so the full card remains the selection target."""
    st.session_state["nav_selected_card"] = card_key
    clear_sensor_detail()


st.title("Chemical Process Anomaly Explorer")
st.caption("Explore how sensor readings respond to disturbances in a batch distillation process, from documented anomalies to detailed signal behavior.")

operating_points = sorted(path.name for path in DATA_DIR.glob("OP*") if path.is_dir())
if not operating_points:
    st.error("No operating-point directories were found in the dataset.")
    st.stop()
if "nav_selected_op" not in st.session_state:
    st.session_state["nav_selected_op"] = "OP17" if "OP17" in operating_points else operating_points[0]

counts = anomaly_counts_by_op(operating_points)
nav_left, nav_right = st.columns([1, 1], gap="large")
with nav_left:
    st.subheader("Operating Points")
    st.caption("Each rectangle represents an operating point, sized by its number of distinct annotated anomalies. Select one to explore its events.")
    treemap_selection = st.plotly_chart(
        build_operating_point_treemap(counts, st.session_state["nav_selected_op"]),
        key="operating_point_treemap",
        on_select="rerun",
        selection_mode="points",
        width="stretch",
    )
    clicked_op = selected_treemap_op(treemap_selection, operating_points)
    if clicked_op and clicked_op != st.session_state["nav_selected_op"]:
        st.session_state["nav_selected_op"] = clicked_op
        st.rerun()
    selected_op = st.selectbox("Operating point", operating_points, key="nav_selected_op")
    st.caption("Area represents distinct annotated anomaly IDs, not severity or duration.")

cards = anomaly_catalog(selected_op)
if not cards:
    st.error(f"No annotated anomalous recording was found for {selected_op}.")
    st.stop()
card_keys = [card["key"] for card in cards]
if st.session_state.get("nav_selected_card") not in card_keys:
    st.session_state["nav_selected_card"] = card_keys[0]

with nav_right:
    st.subheader("Anomalies")
    st.caption("Choose an anomaly to examine its documented disturbance and sensor responses.")
    card_columns = st.columns(2, gap="small")
    for index, card in enumerate(cards):
        selected = card["key"] == st.session_state["nav_selected_card"]
        with card_columns[index % 2]:
            st.button(
                f"{short_anomaly_id(card['anomaly_id'])}\n\n{card['anomaly_id'] or 'Missing anomaly ID'}",
                key=f"anomaly_card_{card['key']}",
                help=f"Select {card['anomaly_id'] or 'this anomaly'}",
                on_click=select_anomaly_card,
                args=(card["key"],),
                type="primary" if selected else "secondary",
                use_container_width=True,
                wrap=True,
            )

selected_card = next(card for card in cards if card["key"] == st.session_state["nav_selected_card"])
with nav_right:
    show_unresolved_warning(selected_card["events"])

st.divider()
st.header("Sensor Responses")

valid_events = [event for event in selected_card["events"] if event["valid"]]
if not valid_events:
    st.warning("This anomaly has no reliably mapped intervals, so no event regions are drawn.")
    with st.expander("Annotation details"):
        show_event_info(selected_card["events"][0])
    st.stop()

selected_recording_path = Path(selected_card["recording_path"])
selected_recording = load_recording(str(selected_recording_path))
normal_paths = [
    path
    for path in sorted((DATA_DIR / selected_op / "measurements").glob("*.csv"))
    if recording_type(path) == "Normal"
]
documented_sensors = [sensor for sensor in selected_card["sensors"] if sensor in REQUIRED_SENSORS]
if not documented_sensors:
    documented_sensors = sorted({event["affected_sensor"] for event in valid_events if event["affected_sensor"] in REQUIRED_SENSORS})

categories = {str(category).strip().lower() for category in selected_card["categories"]}
full_timeline = len(valid_events) > 1 and "transient" not in categories
active_events = valid_events
if not full_timeline and len(valid_events) > 1:
    event_keys = [event["key"] for event in valid_events]
    selected_event_key = st.selectbox(
        "Annotated interval",
        event_keys,
        key=f"transient_interval_{selected_card['key']}",
        format_func=lambda key: event_label(event_by_key(valid_events, key)),
    )
    active_events = [event_by_key(valid_events, selected_event_key)]

first_active_event = min(active_events, key=lambda event: event["start_index"])
normal_reference_path = normal_paths[0] if first_active_event["start_index"] == 0 and normal_paths else None
normal_reference_data = load_recording(str(normal_reference_path)) if normal_reference_path else None

detail_scope = f"{selected_op}:{selected_card['key']}:{active_events[0]['key'] if not full_timeline else 'all'}"
if st.session_state.get("detail_scope") != detail_scope:
    st.session_state["detail_scope"] = detail_scope
    clear_sensor_detail()

if full_timeline:
    st.caption("Compare all 11 sensor signals across the recording; all valid occurrences of this recurrent anomaly are shown.")
else:
    st.caption("Compare all 11 sensor signals around the selected anomaly. Documented sensors are highlighted; select any sensor for a detailed view.")

controls_left, controls_right = st.columns([1, 1])
with controls_left:
    if full_timeline:
        context_seconds = 300
        st.caption("All valid intervals are shown across the full recording.")
    else:
        context_seconds = st.slider(
            "Context (seconds)", min_value=60, max_value=900, value=300, step=30,
            key=f"response_context_{detail_scope}",
        )
with controls_right:
    detailed_sensor = st.selectbox(
        "Inspect sensor", [None, *REQUIRED_SENSORS], key="detail_sensor_choice",
        format_func=lambda sensor: "Select a sensor…" if sensor is None else sensor,
    )

with st.expander("How overview scaling works"):
    st.write(
        "For each sensor, the baseline is up to 300 seconds immediately before the selected event start "
        "(or the available pre-event samples if fewer). The plotted response is "
        "`(value − pre-event median) / (1.4826 × pre-event MAD)`. For a nearly constant baseline, "
        "the fallback scale is 1% of the absolute pre-event median, with a minimum of `1e-6`. "
        "No during-event or post-event samples are used for the baseline."
    )

overview, scaling_methods, render_stride = build_response_overview(
    selected_recording,
    active_events,
    documented_sensors,
    context_seconds,
    full_timeline=full_timeline,
    baseline_seconds=min(300, context_seconds),
    normal_reference=normal_reference_data,
)
st.plotly_chart(overview, width="stretch")
if normal_reference_path:
    st.caption(
        f"No pre-event sample exists before the first annotated occurrence; scaling uses the same-OP normal recording "
        f"{normal_reference_path.name} as a reference distribution. Recording clocks are not aligned."
    )
if render_stride > 1:
    st.caption(f"Full-recording overview is sampled every {render_stride} seconds for display; interval shading uses the original timestamps.")
fallback_sensors = [method.split(":", 1)[0] for method in scaling_methods if "fallback" in method or "no pre-event" in method]
if fallback_sensors:
    st.caption("Pre-event baseline safeguard used for: " + ", ".join(fallback_sensors) + ".")

if detailed_sensor is not None:
    detail_heading, close_detail = st.columns([6, 1])
    detail_heading.subheader(f"Sensor Detail — {detailed_sensor}")
    detail_heading.caption("Inspect the selected sensor in its original measurement scale to explore changes during the anomaly and its behavior afterward.")
    close_detail.button("Close", key="close_sensor_detail", on_click=clear_sensor_detail, use_container_width=True)

    compare_normal = st.checkbox("Compare with normal", key="compare_with_normal", disabled=not normal_paths)
    if compare_normal:
        normal_path = st.selectbox(
            "Normal recording", normal_paths, key="normal_recording_choice", format_func=lambda path: path.name,
        )
        normal_data = load_recording(str(normal_path))
        st.caption("Compare with a separate normal recording from the same operating point. The experiments are not necessarily synchronized.")
        st.plotly_chart(
            build_sensor_comparison_chart(
                selected_recording, normal_data, detailed_sensor, active_events,
                context_seconds, full_timeline, normal_path.name,
            ),
            width="stretch",
        )
    else:
        st.plotly_chart(
            build_detailed_sensor_chart(selected_recording, detailed_sensor, active_events, context_seconds, full_timeline),
            width="stretch",
        )

    with st.expander("Annotation details"):
        show_event_info(active_events[0])
