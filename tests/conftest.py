from datetime import datetime, timezone, timedelta

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def make_event(quantity: int, days_ago: int = 0, event_type: str = "consumed") -> dict:
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    return {"quantity": quantity, "event_type": event_type, "consumed_at": ts, "product_id": 1}


def make_events(n: int, span_days: int) -> list[dict]:
    """n events spread over span_days."""
    events = []
    for i in range(n):
        days_ago = span_days - int(i * span_days / max(n - 1, 1))
        events.append(make_event(quantity=2, days_ago=days_ago))
    return events
