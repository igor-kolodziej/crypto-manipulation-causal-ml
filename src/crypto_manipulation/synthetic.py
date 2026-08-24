"""Generate privacy-safe event fixtures for demos and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_events(
    n_events: int = 600,
    n_channels: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a deterministic synthetic dataset with channel-level structure."""

    if n_events < 40:
        raise ValueError("n_events must be at least 40")
    if n_channels < 4 or n_channels >= n_events:
        raise ValueError("n_channels must be between 4 and n_events - 1")

    rng = np.random.default_rng(seed)
    channel_number = rng.integers(0, n_channels, size=n_events)
    channel_effect = rng.normal(0, 0.25, size=n_channels)[channel_number]

    market_cap = np.exp(rng.normal(np.log(25_000_000), 1.0, size=n_events))
    volume = np.exp(rng.normal(np.log(900_000), 1.1, size=n_events))
    coin_age = rng.gamma(shape=2.3, scale=420, size=n_events)
    hype = rng.beta(2.2, 2.0, size=n_events)
    urgency = np.clip(0.55 * hype + rng.normal(0.2, 0.18, size=n_events), 0, 1)
    instructions = np.clip(rng.beta(2.0, 3.0, size=n_events) + 0.15 * urgency, 0, 1)

    small_cap = np.clip(1 - np.log10(market_cap) / 10, 0, 1)
    return_10m = (
        0.055 * urgency * small_cap
        + 0.020 * hype
        + 0.010 * instructions
        + 0.008 * channel_effect
        + rng.normal(0, 0.038, size=n_events)
    )

    return pd.DataFrame(
        {
            "event_id": [f"synthetic-{i:05d}" for i in range(n_events)],
            "channel_id": [f"channel-{i:02d}" for i in channel_number],
            "event_time": pd.date_range("2024-01-01", periods=n_events, freq="6h", tz="UTC"),
            "market_cap_usd": market_cap.round(2),
            "pre_event_volume_usd": volume.round(2),
            "coin_age_days": coin_age.round(1),
            "hype_intensity": hype,
            "urgency_index": urgency,
            "instruction_density": instructions,
            "return_10m": return_10m,
            "success": (return_10m > 0).astype("int8"),
        }
    )

