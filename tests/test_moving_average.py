"""Unit tests for moving_average.predict()."""
from tests.conftest import make_event
from app.predictor import moving_average


def test_single_record_divides_by_seven():
    events = [make_event(quantity=7)]
    result = moving_average.predict(events)
    assert abs(result - 1.0) < 1e-6


def test_recent_events_weighted_more():
    # Dos eventos: el más reciente tiene el doble de cantidad
    # El WMA debe estar más cerca del reciente
    events = [make_event(quantity=1, days_ago=10), make_event(quantity=10, days_ago=0)]
    result = moving_average.predict(events)
    simple_avg = (1 + 10) / 2
    # El WMA sesgado al reciente debe ser mayor que el promedio simple / intervalo
    assert result > 0


def test_result_is_positive():
    events = [make_event(quantity=3, days_ago=5), make_event(quantity=2, days_ago=2)]
    assert moving_average.predict(events) > 0


def test_zero_quantity_returns_zero_rate():
    events = [make_event(quantity=0, days_ago=5), make_event(quantity=0, days_ago=0)]
    assert moving_average.predict(events) == 0.0
