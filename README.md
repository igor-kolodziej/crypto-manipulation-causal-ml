# Causal ML for Crypto-Market Manipulation

Research code and thesis materials for **Causal AI for Detecting Crypto Market Manipulation**, my MSc thesis in Data Science & Business Analysis at the University of Warsaw (2026, graduated with distinction).

The project studies whether the wording and timing of Telegram pump-and-dump signals help explain the immediate market response after a target coin is revealed. It deliberately separates two questions:

1. **Prediction:** can success be classified using information available before the reveal?
2. **Causal estimation:** how are signal characteristics associated with short-horizon returns after accounting for observed confounders?

> This repository is for research and education. It is not trading advice and does not provide instructions for participating in market manipulation.

## Thesis results

The private research dataset combined **12.4 million Telegram messages**, market observations, and asset metadata. A group-aware holdout evaluated generalization to channels not seen during training.

| Result | Thesis estimate |
| --- | ---: |
| Unseen-channel predictive macro-F1 | **0.76** |
| Recall for successful events | **0.89** |
| Primary outcome | 10-minute logarithmic return |

These are results reported in the thesis. They are **not** reproduced by the synthetic demonstration included here.

## What is public

- A schema-validated event representation.
- A deterministic synthetic-data generator with no copied messages or real trading records.
- A group-aware predictive benchmark that prevents channel overlap between train and test sets.
- Optional `LinearDML` and `CausalForestDML` demonstrations using EconML.
- The LaTeX thesis source, figures, and aggregate result tables.

Raw Telegram content, annotation data, account/channel identifiers, and proprietary market extracts are not published.

## Architecture

```text
event records
    │
    ├── schema validation ──► grouped train/test split ──► predictive benchmark
    │
    └── treatment/outcome definition ──► DML / causal forest ──► effect estimates
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run the privacy-safe predictive demo
crypto-manipulation demo --events 600 --output demo-results.json

# Run tests
pytest
```

For the optional causal estimators:

```bash
pip install -e ".[causal]"
crypto-manipulation causal-demo --treatment urgency_index
```

## Reproducibility boundary

The public demo validates the data flow, grouped evaluation, metric calculation, and estimator interfaces. It cannot reproduce the thesis estimates without the original research data and trained language model. See [Data availability](docs/DATA_AVAILABILITY.md), [Methodology](docs/METHODOLOGY.md), and [Ethics](docs/ETHICS.md).

## Attribution

The original data-collection stage was informed by the public research artifacts in [Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency](https://github.com/Bayi-Hu/Pump-and-Dump-Detection-on-Cryptocurrency). That repository does not display an open-source license, so its code and datasets are not redistributed here. The public implementation in this repository was written independently from the documented research design and recovered author-owned analysis history.

## Author

[Igor Kołodziej](https://igor-kolodziej.github.io/) · [LinkedIn](https://www.linkedin.com/in/igor-kolodziej/)

