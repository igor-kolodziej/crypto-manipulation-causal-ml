from crypto_manipulation import make_synthetic_events
from crypto_manipulation.predictive import make_grouped_holdout, run_grouped_benchmark


def test_grouped_holdout_has_no_channel_overlap():
    events = make_synthetic_events(n_events=160, n_channels=12, seed=11)
    split = make_grouped_holdout(events, seed=11)
    train_channels = set(events.iloc[split.train_index]["channel_id"])
    test_channels = set(events.iloc[split.test_index]["channel_id"])
    assert train_channels.isdisjoint(test_channels)


def test_benchmark_returns_bounded_metrics():
    events = make_synthetic_events(n_events=180, n_channels=14, seed=3)
    metrics = run_grouped_benchmark(events, seed=3)
    for name in ["accuracy", "macro_f1", "success_precision", "success_recall"]:
        assert 0 <= metrics[name] <= 1
    assert metrics["train_channels"] + metrics["test_channels"] <= 14

