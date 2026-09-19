"""Tests de los endpoints de entrenamiento, activación y predicción interna."""
import pytest

from app.core.config import settings
from app.predictor import gbm
from app.routes import train as train_route
from app.services import model_store
from tests.conftest import make_events

HEADERS = {"X-Internal-Token": settings.INTERNAL_SERVICE_TOKEN}


@pytest.fixture(autouse=True)
def entorno(tmp_path, monkeypatch):
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(model_store, "VERSIONS_FILE", str(tmp_path / "versions.json"))
    gbm._models.clear()
    yield
    gbm._models.clear()


def dataset(n_productos=8, user_type="individuo", qty=2):
    prods = [
        {"id": i, "stock": 20, "stock_minimo": 5, "categoria": "Alimentos",
         "nombre": f"P{i}", "user_type": user_type}
        for i in range(1, n_productos + 1)
    ]
    eventos = []
    for p in prods:
        for e in make_events(20, 90):
            eventos.append({**e, "product_id": p["id"], "quantity": qty})
    return eventos, prods


def fake_backend(monkeypatch, eventos, prods):
    async def _ev():
        return eventos

    async def _pr():
        return prods

    monkeypatch.setattr(train_route.backend_client, "get_all_consumption", _ev)
    monkeypatch.setattr(train_route.backend_client, "get_all_products", _pr)


# --- autenticación de servicio ---

def test_listar_sin_token_prohibido(client):
    assert client.get("/models/").status_code == 403


def test_entrenar_sin_token_prohibido(client):
    assert client.post("/models/train").status_code == 403


def test_token_invalido_prohibido(client):
    r = client.get("/models/", headers={"X-Internal-Token": "mal"})
    assert r.status_code == 403


def test_listar_vacio(client):
    assert client.get("/models/", headers=HEADERS).json() == []


# --- _check_metrics ---

def test_mape_alto_rechaza_cualquier_segmento():
    assert train_route._check_metrics("individuo", 0.1, 1.0, 99.0) is not None
    assert train_route._check_metrics("comercio", 0.1, 1.0, 99.0) is not None


def test_individuo_usa_mae_absoluto():
    assert train_route._check_metrics("individuo", 5.0, 1.0, 5.0) is not None
    assert train_route._check_metrics("individuo", 1.0, 99.0, 5.0) is None


def test_comercio_usa_error_relativo():
    assert train_route._check_metrics("comercio", 99.0, 3.0, 5.0) is None
    assert train_route._check_metrics("comercio", 1.0, 99.0, 5.0) is not None


# --- entrenamiento ---

def test_entrenar_guarda_version(client, monkeypatch):
    fake_backend(monkeypatch, *dataset())
    r = client.post("/models/train", json={"model_type": "gbm"}, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "saved"
    assert body["segment"] == "individuo"
    assert body["version"]["id"] == "v1"


def test_entrenar_datos_insuficientes_es_422(client, monkeypatch):
    fake_backend(monkeypatch, *dataset(n_productos=2))
    r = client.post("/models/train", headers=HEADERS)
    assert r.status_code == 422


def test_entrenar_con_backend_caido_es_502(client, monkeypatch):
    async def explota():
        raise RuntimeError("sin conexión")

    monkeypatch.setattr(train_route.backend_client, "get_all_consumption", explota)
    r = client.post("/models/train", headers=HEADERS)
    assert r.status_code == 502


def test_entrenar_rechaza_metricas_malas(client, monkeypatch):
    fake_backend(monkeypatch, *dataset())

    def metricas_malas(*a, **kw):
        return {"model": None, "scaler": None, "mae": 50.0, "mae_relative": 80.0,
                "mape": 90.0, "n_samples": 8, "n_products": 8,
                "feature_importances": {}, "model_type": "gbm"}

    monkeypatch.setattr(train_route.gbm, "train", metricas_malas)
    r = client.post("/models/train", headers=HEADERS)
    body = r.json()
    assert body["status"] == "rejected"
    assert "MAPE" in body["reason"]
    assert model_store.list_versions() == []


def test_entrenar_rechaza_mae_alto_en_individuos(client, monkeypatch):
    fake_backend(monkeypatch, *dataset())

    def mae_alto(*a, **kw):
        return {"model": None, "scaler": None, "mae": 50.0, "mae_relative": 1.0,
                "mape": 5.0, "n_samples": 8, "n_products": 8,
                "feature_importances": {}, "model_type": "gbm"}

    monkeypatch.setattr(train_route.gbm, "train", mae_alto)
    body = client.post("/models/train", headers=HEADERS).json()
    assert body["status"] == "rejected"
    assert "u/sem" in body["reason"]


def test_entrenar_con_dataset_de_prueba(client, monkeypatch):
    eventos, prods = dataset(user_type="comercio")
    monkeypatch.setattr(train_route.test_dataset, "load_events", lambda: eventos)
    monkeypatch.setattr(train_route.test_dataset, "load_products", lambda: prods)

    r = client.post("/models/train",
                    json={"data_source": "test", "segment": "comercio"}, headers=HEADERS)
    body = r.json()
    assert body["status"] == "saved"
    assert body["version"]["data_source"] == "test"


def test_dataset_de_prueba_no_aplica_a_individuos(client):
    r = client.post("/models/train",
                    json={"data_source": "test", "segment": "individuo"}, headers=HEADERS)
    assert r.status_code == 422
    assert "comercial" in r.json()["detail"]


def test_entrenar_comercio_usa_criterio_relativo(client, monkeypatch):
    # consumo alto: el MAE absoluto sería inaceptable para un individuo
    fake_backend(monkeypatch, *dataset(user_type="comercio", qty=300))
    r = client.post("/models/train",
                    json={"segment": "comercio"}, headers=HEADERS)
    assert r.json()["status"] == "saved"


# --- importancias ---

def test_importancias_de_version_inexistente_es_404(client):
    assert client.get("/models/v99/feature-importances", headers=HEADERS).status_code == 404


def test_importancias_de_version_entrenada(client, monkeypatch):
    fake_backend(monkeypatch, *dataset())
    client.post("/models/train", headers=HEADERS)
    r = client.get("/models/v1/feature-importances", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["model_type"] == "gbm"
    assert r.json()["feature_importances"]


# --- activación ---

def test_activar_inexistente_es_404(client):
    assert client.post("/models/v99/activate", headers=HEADERS).status_code == 404


def test_activar_carga_el_modelo_en_su_segmento(client, monkeypatch):
    fake_backend(monkeypatch, *dataset())
    client.post("/models/train", headers=HEADERS)

    r = client.post("/models/v1/activate", headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["status"] == "activated"
    assert gbm.is_trained("individuo")


# --- predict-product ---

PRODUCTO = {"id": 1, "stock": 20, "stock_minimo": 5, "categoria": "Alimentos", "nombre": "Arroz"}


def test_predict_sin_eventos_usa_bayesiano(client):
    r = client.post("/models/predict-product",
                    json={"events": [], "product": PRODUCTO}, headers=HEADERS)
    body = r.json()
    assert body["stage"] == "bayesian"
    assert body["daily_rate"] > 0
    assert body["days_until_stockout"] is not None


def test_predict_informa_el_segmento(client):
    r = client.post("/models/predict-product",
                    json={"events": [], "product": PRODUCTO, "segment": "comercio"},
                    headers=HEADERS)
    assert r.json()["segment"] == "comercio"


def test_predict_ignora_eventos_vencidos(client):
    eventos = [{**e, "event_type": "expired"} for e in make_events(20, 90)]
    r = client.post("/models/predict-product",
                    json={"events": eventos, "product": PRODUCTO}, headers=HEADERS)
    assert r.json()["stage"] == "bayesian"


def test_predict_con_historial_largo_usa_gbm(client):
    r = client.post("/models/predict-product",
                    json={"events": make_events(20, 90), "product": PRODUCTO}, headers=HEADERS)
    assert r.json()["stage"] == "gbm"


def test_predict_con_historial_medio_usa_media_movil(client):
    r = client.post("/models/predict-product",
                    json={"events": make_events(15, 40), "product": PRODUCTO}, headers=HEADERS)
    assert r.json()["stage"] == "moving_average"


def test_predict_sin_stock_no_calcula_agotamiento(client):
    producto = {**PRODUCTO, "stock": 0}
    r = client.post("/models/predict-product",
                    json={"events": [], "product": producto}, headers=HEADERS)
    assert r.json()["days_until_stockout"] == 0
