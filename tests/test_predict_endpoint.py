"""Integration tests for POST /models/predict-product."""
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings
from tests.conftest import make_events

client = TestClient(app)
HEADERS = {"X-Internal-Token": settings.INTERNAL_SERVICE_TOKEN}
PRODUCT = {"id": 1, "stock": 20, "stock_minimo": 5, "categoria": "Alimentos", "nombre": "Arroz"}


def post_predict(events, product=None):
    return client.post(
        "/models/predict-product",
        json={"events": events, "product": product or PRODUCT},
        headers=HEADERS,
    )


def test_predict_no_token_returns_403():
    r = client.post("/models/predict-product", json={"events": [], "product": PRODUCT})
    assert r.status_code == 403


def test_predict_empty_events_returns_bayesian():
    r = post_predict([])
    assert r.status_code == 200
    data = r.json()
    assert data["stage"] == "bayesian"
    assert data["daily_rate"] > 0
    assert "days_until_stockout" in data


def test_predict_few_events_returns_bayesian():
    events = make_events(5, span_days=30)
    r = post_predict(events)
    assert r.status_code == 200
    assert r.json()["stage"] == "bayesian"


def test_predict_medium_span_returns_moving_average():
    events = make_events(10, span_days=40)
    r = post_predict(events)
    assert r.status_code == 200
    assert r.json()["stage"] == "moving_average"


def test_predict_days_until_stockout_with_zero_stock():
    # stock=0 → 0 días hasta agotarse (no None, ya está agotado)
    product = {**PRODUCT, "stock": 0}
    r = post_predict([], product=product)
    assert r.status_code == 200
    assert r.json()["days_until_stockout"] == 0


def test_predict_days_until_stockout_calculation():
    r = post_predict(make_events(5, span_days=10))
    data = r.json()
    if data["daily_rate"] > 0:
        expected = round(PRODUCT["stock"] / data["daily_rate"])
        assert data["days_until_stockout"] == expected


def test_predict_filters_restocked_events():
    from tests.conftest import make_event
    # Mezcla consumed y restocked — el stage debe calcularse solo con consumed
    events = make_events(5, span_days=10)
    events.append({**make_event(quantity=10, days_ago=1), "event_type": "restocked"})
    r = post_predict(events)
    assert r.status_code == 200
    # 5 consumed events → bayesian (< MIN_RECORDS_FOR_WMA)
    assert r.json()["stage"] == "bayesian"
