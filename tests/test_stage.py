"""Unit tests for determine_stage()."""
from tests.conftest import make_events
from app.predictor.stage import determine_stage


def test_empty_records_returns_bayesian():
    assert determine_stage([]) == "bayesian"


def test_few_records_returns_bayesian():
    # 5 eventos en 30 días — por debajo de MIN_RECORDS_FOR_WMA (10)
    events = make_events(5, span_days=30)
    assert determine_stage(events) == "bayesian"


def test_enough_records_short_span_returns_bayesian():
    # 10 eventos en 10 días — no alcanza MIN_DAYS_FOR_WMA (28)
    events = make_events(10, span_days=10)
    assert determine_stage(events) == "bayesian"


def test_enough_records_medium_span_returns_moving_average():
    # 10 eventos en 40 días — entre WMA y GBM threshold
    events = make_events(10, span_days=40)
    assert determine_stage(events) == "moving_average"


def test_enough_records_long_span_returns_gbm():
    # 10 eventos en 60 días — supera MIN_DAYS_FOR_GBM (56)
    events = make_events(10, span_days=60)
    assert determine_stage(events) == "gbm"


def test_single_record_returns_bayesian():
    events = make_events(1, span_days=0)
    assert determine_stage(events) == "bayesian"
