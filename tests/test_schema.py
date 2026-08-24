import pandas as pd
import pytest

from crypto_manipulation import make_synthetic_events, validate_events


def test_synthetic_events_validate():
    events = make_synthetic_events(n_events=80, n_channels=8, seed=7)
    validated = validate_events(events)
    assert len(validated) == 80
    assert set(validated["success"].unique()).issubset({0, 1})


def test_missing_column_is_rejected():
    events = make_synthetic_events(n_events=80, n_channels=8).drop(columns=["urgency_index"])
    with pytest.raises(ValueError, match="urgency_index"):
        validate_events(events)


def test_duplicate_event_id_is_rejected():
    events = make_synthetic_events(n_events=80, n_channels=8)
    events.loc[1, "event_id"] = events.loc[0, "event_id"]
    with pytest.raises(ValueError, match="unique"):
        validate_events(events)

