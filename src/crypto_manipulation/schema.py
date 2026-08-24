"""Input schema and validation for event-level analysis."""

from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = [
    "market_cap_usd",
    "pre_event_volume_usd",
    "coin_age_days",
    "hype_intensity",
    "urgency_index",
    "instruction_density",
]

REQUIRED_COLUMNS = [
    "event_id",
    "channel_id",
    "event_time",
    *FEATURE_COLUMNS,
    "return_10m",
    "success",
]


def validate_events(events: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an event-level dataframe.

    A copy is returned so downstream code never mutates the caller's data.
    Channel identifiers are treated as grouping labels, not model features.
    """

    missing = sorted(set(REQUIRED_COLUMNS) - set(events.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    if events.empty:
        raise ValueError("The event dataset is empty")

    cleaned = events.loc[:, REQUIRED_COLUMNS].copy()
    cleaned["event_time"] = pd.to_datetime(cleaned["event_time"], utc=True, errors="raise")

    numeric_columns = [*FEATURE_COLUMNS, "return_10m", "success"]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="raise")

    if cleaned["event_id"].duplicated().any():
        raise ValueError("event_id must be unique")
    if cleaned["channel_id"].isna().any():
        raise ValueError("channel_id cannot contain null values")
    if not set(cleaned["success"].unique()).issubset({0, 1}):
        raise ValueError("success must contain only 0 and 1")
    if cleaned[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("model features cannot contain null values")
    if (cleaned[["market_cap_usd", "pre_event_volume_usd", "coin_age_days"]] < 0).any().any():
        raise ValueError("market and asset features cannot be negative")

    cleaned["success"] = cleaned["success"].astype("int8")
    cleaned["channel_id"] = cleaned["channel_id"].astype(str)
    cleaned["event_id"] = cleaned["event_id"].astype(str)
    return cleaned

