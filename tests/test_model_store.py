"""Tests del versionado de modelos, con una versión activa por segmento."""
import json
import os

import pytest

from app.services import model_store


@pytest.fixture(autouse=True)
def store_aislado(tmp_path, monkeypatch):
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(model_store, "VERSIONS_FILE", str(tmp_path / "versions.json"))
    yield tmp_path


class Modelo:
    def predict(self, X):
        return [1.0]


class Scaler:
    def transform(self, X):
        return X


def guardar(**kwargs) -> dict:
    base = dict(model=Modelo(), mae=1.0, mape=10.0, n_samples=5, n_products=5)
    return model_store.save_new_version(**{**base, **kwargs})


def test_lista_vacia_si_no_hay_archivo():
    assert model_store.list_versions() == []


def test_sin_versiones_no_hay_activa():
    assert model_store.get_active_version() is None
    assert model_store.load_active_model() is None


def test_guardar_asigna_ids_incrementales():
    assert guardar()["id"] == "v1"
    assert guardar()["id"] == "v2"


def test_guardar_nace_inactiva_y_persiste_el_pkl(store_aislado):
    v = guardar()
    assert v["status"] == "inactive"
    assert os.path.exists(store_aislado / v["model_file"])


def test_guardar_registra_metadatos():
    v = guardar(model_type="svr", data_source="test", segment="comercio", mae_relative=4.2)
    assert v["model_type"] == "svr"
    assert v["data_source"] == "test"
    assert v["segment"] == "comercio"
    assert v["mae_relative"] == 4.2


def test_guardar_redondea_metricas():
    v = guardar(mae=1.234567, mape=9.87654, mae_relative=3.14159)
    assert v["mae"] == 1.2346
    assert v["mape"] == 9.88
    assert v["mae_relative"] == 3.14


def test_activar_version_inexistente_devuelve_none():
    assert model_store.activate_version("v99") is None


def test_activar_marca_activa():
    v = guardar()
    assert model_store.activate_version(v["id"])["status"] == "active"
    assert model_store.get_active_version()["id"] == v["id"]


def test_activar_desactiva_la_anterior_del_mismo_segmento():
    a = guardar(segment="individuo")
    b = guardar(segment="individuo")
    model_store.activate_version(a["id"])
    model_store.activate_version(b["id"])

    estados = {v["id"]: v["status"] for v in model_store.list_versions()}
    assert estados[a["id"]] == "inactive"
    assert estados[b["id"]] == "active"


def test_activar_en_un_segmento_no_toca_el_otro():
    ind = guardar(segment="individuo")
    com = guardar(segment="comercio")
    model_store.activate_version(ind["id"])
    model_store.activate_version(com["id"])

    assert model_store.get_active_version("individuo")["id"] == ind["id"]
    assert model_store.get_active_version("comercio")["id"] == com["id"]


def test_version_sin_segmento_cuenta_como_individuo(store_aislado):
    v = guardar()
    del v["segment"]
    (store_aislado / "versions.json").write_text(json.dumps([{**v, "status": "active"}]))
    assert model_store.get_active_version("individuo")["id"] == v["id"]
    assert model_store.get_active_version("comercio") is None


def test_cargar_modelo_activo_devuelve_modelo_y_scaler():
    v = guardar(scaler=Scaler())
    model_store.activate_version(v["id"])
    modelo, scaler = model_store.load_active_model()
    assert modelo is not None and scaler is not None


def test_cargar_sin_archivo_en_disco_devuelve_none(store_aislado):
    v = guardar()
    model_store.activate_version(v["id"])
    os.remove(store_aislado / v["model_file"])
    assert model_store.load_active_model() is None


def test_cargar_formato_viejo_sin_scaler(store_aislado):
    import joblib

    v = guardar()
    model_store.activate_version(v["id"])
    # las versiones antiguas guardaban el modelo pelado, no un dict
    joblib.dump(Modelo(), store_aislado / v["model_file"])

    modelo, scaler = model_store.load_active_model()
    assert modelo is not None
    assert scaler is None
