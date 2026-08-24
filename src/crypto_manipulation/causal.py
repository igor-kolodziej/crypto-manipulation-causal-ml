"""Optional EconML estimators used by the public demonstration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

from .schema import FEATURE_COLUMNS, validate_events


def _analysis_arrays(events: pd.DataFrame, treatment: str):
    validated = validate_events(events)
    if treatment not in {"hype_intensity", "urgency_index", "instruction_density"}:
        raise ValueError("treatment must be a documented signal feature")
    controls = [column for column in FEATURE_COLUMNS if column != treatment]
    return (
        validated["return_10m"].to_numpy(),
        validated[treatment].to_numpy(),
        validated[controls].to_numpy(),
    )


def estimate_linear_dml(events: pd.DataFrame, treatment: str = "urgency_index") -> dict[str, float]:
    """Estimate an average treatment effect with EconML LinearDML."""

    try:
        from econml.dml import LinearDML
    except ImportError as error:
        raise RuntimeError('Install optional dependencies with: pip install -e ".[causal]"') from error

    outcome, treatment_values, controls = _analysis_arrays(events, treatment)
    estimator = LinearDML(
        model_y=RandomForestRegressor(n_estimators=120, min_samples_leaf=8, random_state=42),
        model_t=RandomForestRegressor(n_estimators=120, min_samples_leaf=8, random_state=42),
        model_final=LassoCV(cv=5),
        cv=3,
        random_state=42,
    )
    estimator.fit(outcome, treatment_values, X=controls)
    effect = np.asarray(estimator.effect(controls))
    return {"mean_effect": float(effect.mean()), "effect_std": float(effect.std())}


def estimate_causal_forest(events: pd.DataFrame, treatment: str = "urgency_index") -> dict[str, float]:
    """Estimate heterogeneous effects with EconML CausalForestDML."""

    try:
        from econml.dml import CausalForestDML
    except ImportError as error:
        raise RuntimeError('Install optional dependencies with: pip install -e ".[causal]"') from error

    outcome, treatment_values, controls = _analysis_arrays(events, treatment)
    estimator = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=120, min_samples_leaf=8, random_state=42),
        model_t=RandomForestRegressor(n_estimators=120, min_samples_leaf=8, random_state=42),
        n_estimators=200,
        min_samples_leaf=8,
        cv=3,
        random_state=42,
    )
    estimator.fit(outcome, treatment_values, X=controls)
    effect = np.asarray(estimator.effect(controls))
    return {
        "mean_effect": float(effect.mean()),
        "effect_std": float(effect.std()),
        "effect_p10": float(np.quantile(effect, 0.10)),
        "effect_p90": float(np.quantile(effect, 0.90)),
    }

