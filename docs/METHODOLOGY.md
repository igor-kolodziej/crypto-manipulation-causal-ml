# Methodology

## Predictive task

The binary target is whether the 10-minute logarithmic return after the reveal is positive. Only information available by reveal time is eligible as a feature. Evaluation uses a group-aware holdout: every channel appears in either training or testing, never both. This tests transfer to unseen communication environments and prevents channel identity from leaking across the split.

The thesis benchmark used XGBoost, threshold analysis, and a broader feature set. The public default uses scikit-learn gradient boosting so the demonstration remains compact; the distinction is explicit in every reported output.

## Causal task

The continuous outcome is the 10-minute logarithmic return. Signal features such as hype intensity, urgency index, and instruction density are analyzed as treatments. Market capitalization, pre-event volume, coin age, and the remaining signal features act as observed controls.

`LinearDML` estimates an average conditional relationship after nuisance-model residualization. `CausalForestDML` explores heterogeneous effects. These estimators depend on identification assumptions that cannot be proven from the observed data, including conditional exchangeability and adequate overlap. Estimates should therefore be interpreted as assumption-dependent evidence, not proof that a message feature caused a return.

## Reproducibility

The original training data and language-model checkpoint are private. The public tests cover schema enforcement, deterministic fixtures, grouped holdout integrity, metric bounds, and estimator entry points.
