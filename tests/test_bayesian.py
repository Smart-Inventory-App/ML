"""Unit tests for bayesian.predict()."""
from app.predictor import bayesian
from app.core.config import settings
from tests.conftest import make_event


def test_empty_records_returns_prior_mean():
    result = bayesian.predict([])
    expected = settings.BAYESIAN_PRIOR_ALPHA / (settings.BAYESIAN_PRIOR_ALPHA + settings.BAYESIAN_PRIOR_BETA)
    assert abs(result - expected) < 1e-9


def test_with_records_increases_estimate():
    # Con consumo observado la estimación sube respecto al prior
    prior = bayesian.predict([])
    events = [make_event(quantity=5), make_event(quantity=3)]
    posterior = bayesian.predict(events)
    assert posterior > prior


def test_posterior_formula():
    alpha = settings.BAYESIAN_PRIOR_ALPHA
    beta = settings.BAYESIAN_PRIOR_BETA
    events = [make_event(quantity=4), make_event(quantity=2)]
    observed_units = 6
    observed_days = 2
    expected = (alpha + observed_units) / (alpha + beta + observed_days)
    assert abs(bayesian.predict(events) - expected) < 1e-9


def test_higher_consumption_yields_higher_rate():
    low = bayesian.predict([make_event(quantity=1)])
    high = bayesian.predict([make_event(quantity=10)])
    assert high > low
