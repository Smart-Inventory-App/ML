"""Tests del cliente HTTP al backend, el dataset de prueba y el arranque del servicio."""
from datetime import date, datetime

import httpx
import pytest

from app.services import backend_client, test_dataset


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=None)


class FakeClient:
    """Reemplaza a httpx.AsyncClient registrando las llamadas."""

    registro: list[tuple[str, dict]] = []
    payload = {}
    status = 200

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, headers=None):
        FakeClient.registro.append((url, headers or {}))
        return FakeResponse(FakeClient.payload, FakeClient.status)


@pytest.fixture
def fake_http(monkeypatch):
    FakeClient.registro = []
    FakeClient.payload = {}
    FakeClient.status = 200
    monkeypatch.setattr(backend_client.httpx, "AsyncClient", FakeClient)
    return FakeClient


# --- backend_client ---

@pytest.mark.anyio
async def test_get_all_consumption_usa_token_interno(fake_http):
    fake_http.payload = [{"id": 1}]
    assert await backend_client.get_all_consumption() == [{"id": 1}]
    url, headers = fake_http.registro[0]
    assert url.endswith("/internal/consumption")
    assert "X-Internal-Token" in headers


@pytest.mark.anyio
async def test_get_all_products_usa_token_interno(fake_http):
    fake_http.payload = [{"id": 7}]
    assert await backend_client.get_all_products() == [{"id": 7}]
    assert fake_http.registro[0][0].endswith("/internal/products")


@pytest.mark.anyio
async def test_get_consumption_history_usa_bearer(fake_http):
    fake_http.payload = [{"quantity": 2}]
    await backend_client.get_consumption_history(5, "tok")
    url, headers = fake_http.registro[0]
    assert url.endswith("/products/5/consumption")
    assert headers["Authorization"] == "Bearer tok"


@pytest.mark.anyio
async def test_get_product_usa_bearer(fake_http):
    fake_http.payload = {"id": 5}
    assert await backend_client.get_product(5, "tok") == {"id": 5}
    assert fake_http.registro[0][0].endswith("/products/5")


@pytest.mark.anyio
async def test_error_http_se_propaga(fake_http):
    fake_http.status = 500
    with pytest.raises(httpx.HTTPStatusError):
        await backend_client.get_all_products()


# --- test_dataset ---

def test_productos_del_dataset_son_comercio():
    prods = test_dataset.load_products()
    assert prods
    assert all(p["user_type"] == "comercio" for p in prods)
    assert all(isinstance(p["stock"], int) for p in prods)


def test_eventos_terminan_hoy():
    eventos = test_dataset.load_events()
    assert eventos
    ultimo = max(datetime.fromisoformat(e["consumed_at"]).date() for e in eventos)
    assert ultimo == date.today()


def test_eventos_tienen_la_forma_del_backend():
    e = test_dataset.load_events()[0]
    assert set(e) == {"product_id", "quantity", "event_type", "consumed_at"}
    assert e["event_type"] == "consumed"


def test_dataset_vacio_devuelve_lista_vacia(tmp_path, monkeypatch):
    import csv
    import gzip

    vacio = tmp_path / "events.csv.gz"
    with gzip.open(vacio, "wt", newline="") as f:
        csv.writer(f).writerow(["date", "product_id", "quantity"])

    monkeypatch.setattr(test_dataset, "EVENTS_FILE", vacio)
    assert test_dataset.load_events() == []


# --- arranque ---

def test_arranque_carga_el_modelo_activo_de_cada_segmento(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.predictor import gbm
    from app.services import model_store
    from tests.test_model_store import Modelo

    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(model_store, "VERSIONS_FILE", str(tmp_path / "versions.json"))
    gbm._models.clear()

    v = model_store.save_new_version(model=Modelo(), mae=1.0, mape=1.0, n_samples=5,
                                     n_products=5, segment="comercio")
    model_store.activate_version(v["id"])

    from app.main import app

    with TestClient(app) as c:
        assert c.get("/health").json()["models"]["comercio"]["active_version"] == v["id"]
    assert gbm.is_trained("comercio")
    gbm._models.clear()


def test_health_reporta_ambos_segmentos(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert set(body["models"]) == {"individuo", "comercio"}
    for estado in body["models"].values():
        assert "loaded" in estado and "active_version" in estado
