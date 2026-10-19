"""Tests del endpoint público de predicción y sus alertas de vencimiento."""
from datetime import date, datetime, timedelta, timezone

import pytest
from jose import jwt

from app.core.config import settings
from app.routes import predict as predict_route
from tests.conftest import make_events


def hoy_utc() -> date:
    """La fecha que usa el servicio.

    `_expiry_alert` calcula los días restantes contra `datetime.now(timezone.utc)`,
    no contra la fecha local. Usar `date.today()` acá hace que los tests fallen
    en cualquier huso al oeste de Greenwich cuando ya cambió el día en UTC.
    """
    return datetime.now(timezone.utc).date()


def auth_header(sub: str = "user-1") -> dict:
    token = jwt.encode({"sub": sub}, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    return {"Authorization": f"Bearer {token}"}


PRODUCTO = {"id": 1, "nombre": "Arroz", "stock": 20, "stock_minimo": 5, "categoria": "Alimentos"}


def fake_backend(monkeypatch, records, product):
    async def _hist(product_id, token):
        return records

    async def _prod(product_id, token):
        return product

    monkeypatch.setattr(predict_route, "get_consumption_history", _hist)
    monkeypatch.setattr(predict_route, "get_product", _prod)


# --- _parse_expiry ---

def test_parse_expiry_sin_fecha():
    assert predict_route._parse_expiry({}) is None


def test_parse_expiry_acepta_objeto_date():
    hoy = hoy_utc()
    assert predict_route._parse_expiry({"vencimiento": hoy}) == hoy


def test_parse_expiry_acepta_iso():
    assert predict_route._parse_expiry({"vencimiento": "2030-05-01"}) == date(2030, 5, 1)


def test_parse_expiry_texto_invalido_es_none():
    assert predict_route._parse_expiry({"vencimiento": "ayer"}) is None


# --- _expiry_alert ---

def test_sin_vencimiento_no_hay_alerta():
    a = predict_route._expiry_alert(10, 1.0, None)
    assert a["expiry_alert"] is False
    assert a["days_until_expiry"] is None


def test_producto_vencido_pone_todo_el_stock_en_riesgo():
    a = predict_route._expiry_alert(10, 1.0, hoy_utc() - timedelta(days=3))
    assert a["expiry_alert"] is True
    assert a["units_at_risk"] == 10
    assert a["days_until_expiry"] < 0


def test_consumo_lento_deja_unidades_en_riesgo():
    a = predict_route._expiry_alert(100, 1.0, hoy_utc() + timedelta(days=10))
    assert a["expiry_alert"] is True
    assert a["units_at_risk"] == pytest.approx(90.0)


def test_consumo_rapido_no_alerta():
    a = predict_route._expiry_alert(5, 10.0, hoy_utc() + timedelta(days=10))
    assert a["expiry_alert"] is False
    assert a["units_at_risk"] == 0


# --- endpoint ---

def test_sin_credenciales_es_401(client):
    assert client.get("/predict/1").status_code == 401


def test_token_invalido_es_401(client):
    r = client.get("/predict/1", headers={"Authorization": "Bearer basura"})
    assert r.status_code == 401


def test_token_sin_sub_es_401(client):
    token = jwt.encode({"foo": "bar"}, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    r = client.get("/predict/1", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 401


def test_backend_caido_es_502(client, monkeypatch):
    async def explota(*a, **kw):
        raise RuntimeError("sin backend")

    monkeypatch.setattr(predict_route, "get_consumption_history", explota)
    r = client.get("/predict/1", headers=auth_header())
    assert r.status_code == 502


def test_prediccion_devuelve_el_resumen(client, monkeypatch):
    fake_backend(monkeypatch, make_events(20, 90), PRODUCTO)
    body = client.get("/predict/1", headers=auth_header()).json()

    assert body["product_id"] == 1
    assert body["product_name"] == "Arroz"
    assert body["current_stock"] == 20
    assert body["model_stage"] == "gbm"
    assert body["data_points"] == 20
    assert body["prediction_days"] == 7
    assert body["predicted_daily_consumption"] > 0


def test_prediccion_acepta_horizonte_propio(client, monkeypatch):
    fake_backend(monkeypatch, make_events(20, 90), PRODUCTO)
    body = client.get("/predict/1?days=30", headers=auth_header()).json()
    assert body["prediction_days"] == 30
    assert body["predicted_consumption"] == pytest.approx(
        body["predicted_daily_consumption"] * 30, rel=0.01
    )


def test_prediccion_cuenta_eventos_vencidos_aparte(client, monkeypatch):
    consumidos = make_events(20, 90)
    vencidos = [{**e, "event_type": "expired"} for e in make_events(3, 10)]
    fake_backend(monkeypatch, consumidos + vencidos, PRODUCTO)

    body = client.get("/predict/1", headers=auth_header()).json()
    assert body["data_points"] == 20
    assert body["expired_events"] == 3


def test_prediccion_sin_historial_usa_bayesiano(client, monkeypatch):
    fake_backend(monkeypatch, [], PRODUCTO)
    body = client.get("/predict/1", headers=auth_header()).json()
    assert body["model_stage"] == "bayesian"


def test_prediccion_con_historial_medio_usa_media_movil(client, monkeypatch):
    fake_backend(monkeypatch, make_events(15, 40), PRODUCTO)
    body = client.get("/predict/1", headers=auth_header()).json()
    assert body["model_stage"] == "moving_average"
    assert body["predicted_daily_consumption"] > 0


def test_prediccion_incluye_alerta_de_vencimiento(client, monkeypatch):
    producto = {**PRODUCTO, "stock": 500,
                "vencimiento": (hoy_utc() + timedelta(days=5)).isoformat()}
    fake_backend(monkeypatch, make_events(20, 90), producto)

    body = client.get("/predict/1", headers=auth_header()).json()
    assert body["expiry_alert"] is True
    assert body["days_until_expiry"] == 5
    assert body["units_at_risk"] > 0


@pytest.mark.parametrize("tipo", ["sold", "rented"])
def test_prediccion_cuenta_ventas_y_alquileres_como_consumo(client, monkeypatch, tipo):
    consumidos = make_events(10, 90)
    vendidos = [{**e, "event_type": tipo} for e in make_events(10, 45)]
    fake_backend(monkeypatch, consumidos + vendidos, PRODUCTO)

    body = client.get("/predict/1", headers=auth_header()).json()

    assert body["data_points"] == 20
    assert body["expired_events"] == 0
