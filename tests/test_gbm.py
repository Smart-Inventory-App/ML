"""Tests del predictor GBM: features, entrenamiento y segmentación."""
import pytest

from app.predictor import gbm
from tests.conftest import make_event, make_events


@pytest.fixture(autouse=True)
def clean_models():
    gbm._models.clear()
    yield
    gbm._models.clear()


def products(n: int, user_type: str = "individuo", categoria: str = "Alimentos") -> list[dict]:
    return [
        {"id": i, "stock": 20 + i, "stock_minimo": 5, "categoria": categoria,
         "nombre": f"P{i}", "user_type": user_type}
        for i in range(1, n + 1)
    ]


def events_for(product_ids, n: int = 20, span_days: int = 90) -> list[dict]:
    out = []
    for pid in product_ids:
        for e in make_events(n, span_days):
            out.append({**e, "product_id": pid})
    return out


# --- _compute_daily_rate ---

def test_daily_rate_sin_registros_es_cero():
    assert gbm._compute_daily_rate([]) == 0.0


def test_daily_rate_con_un_registro_asume_una_semana():
    assert gbm._compute_daily_rate([make_event(14)]) == pytest.approx(2.0)


def test_daily_rate_mismo_dia_no_divide_por_cero():
    ev = [make_event(3, days_ago=0), make_event(5, days_ago=0)]
    assert gbm._compute_daily_rate(ev) == 8


def test_daily_rate_reparte_sobre_el_span():
    ev = [make_event(5, days_ago=10), make_event(5, days_ago=0)]
    assert gbm._compute_daily_rate(ev) == pytest.approx(1.0)


# --- _build_features ---

def test_features_tiene_el_largo_declarado():
    feats = gbm._build_features(make_events(12, 60), products(1)[0])
    # FEATURE_NAMES incluye categoria, que train() agrega aparte
    assert len(feats) == len(gbm.FEATURE_NAMES) - 1


def test_features_sin_registros_no_rompe():
    feats = gbm._build_features([], {"stock": 10, "stock_minimo": 2})
    assert feats[0] == 0.0 and feats[1] == 0.0
    assert feats[3] == 999


def test_features_un_solo_registro_deja_frecuencia_en_cero():
    feats = gbm._build_features([make_event(4, days_ago=1)], products(1)[0])
    assert feats[2] == 0.0


def test_features_stock_ratio_cero_si_no_hay_minimo():
    feats = gbm._build_features(make_events(12, 60), {"stock": 10, "stock_minimo": 0})
    assert feats[5] == 0.0


def test_features_stock_ratio_se_calcula():
    feats = gbm._build_features(make_events(12, 60), {"stock": 10, "stock_minimo": 4})
    assert feats[5] == pytest.approx(2.5)


def test_features_one_hot_de_dia_y_mes_suman_uno():
    feats = gbm._build_features(make_events(12, 60), products(1)[0])
    assert sum(feats[6:13]) == 1.0
    assert sum(feats[13:25]) == 1.0


# --- set_model / is_trained / predict ---

def test_sin_modelo_predict_cae_a_media_movil():
    ev = make_events(12, 40)
    assert gbm.predict(ev, products(1)[0]) == pytest.approx(gbm.moving_average.predict(ev))


def test_is_trained_es_por_segmento():
    gbm.set_model(object(), segment="comercio")
    assert gbm.is_trained("comercio")
    assert not gbm.is_trained("individuo")


def test_predict_usa_el_modelo_del_segmento():
    class Fake:
        def predict(self, X):
            return [7.5]

    gbm.set_model(Fake(), scaler=None, segment="comercio")
    ev = make_events(12, 60)
    assert gbm.predict(ev, products(1)[0], segment="comercio") == pytest.approx(7.5)
    # el otro segmento sigue sin modelo
    assert gbm.predict(ev, products(1)[0], segment="individuo") != pytest.approx(7.5)


def test_predict_aplica_el_scaler():
    registro = {}

    class Fake:
        def predict(self, X):
            return [1.0]

    class Scaler:
        def transform(self, X):
            registro["llamado"] = True
            return X

    gbm.set_model(Fake(), scaler=Scaler(), segment="individuo")
    gbm.predict(make_events(12, 60), products(1)[0])
    assert registro.get("llamado")


# --- _make_regressor ---

@pytest.mark.parametrize("tipo", ["gbm", "xgboost", "lightgbm", "svr"])
def test_make_regressor_soporta_los_cuatro(tipo):
    assert gbm._make_regressor(tipo) is not None


def test_make_regressor_desconocido_cae_a_gbm():
    from sklearn.ensemble import GradientBoostingRegressor
    assert isinstance(gbm._make_regressor("inexistente"), GradientBoostingRegressor)


# --- train ---

def test_train_con_pocos_productos_falla():
    prods = products(3)
    with pytest.raises(ValueError, match="Datos insuficientes"):
        gbm.train(events_for([p["id"] for p in prods]), prods)


def test_train_descarta_productos_con_poco_historial():
    prods = products(6)
    # 5 eventos por producto: por debajo de MIN_RECORDS_FOR_WMA
    with pytest.raises(ValueError, match="Datos insuficientes"):
        gbm.train(events_for([p["id"] for p in prods], n=5), prods)


def test_train_descarta_historial_corto():
    prods = products(6)
    with pytest.raises(ValueError, match="Datos insuficientes"):
        gbm.train(events_for([p["id"] for p in prods], n=20, span_days=30), prods)


def test_train_devuelve_metricas_y_modelo():
    prods = products(8)
    res = gbm.train(events_for([p["id"] for p in prods]), prods)
    assert res["n_samples"] == 8
    assert res["n_products"] == 8
    assert res["model_type"] == "gbm"
    assert res["model"] is not None and res["scaler"] is not None
    for k in ("mae", "mae_relative", "mape"):
        assert isinstance(res[k], float)


def test_train_ignora_eventos_no_consumidos():
    prods = products(8)
    ev = events_for([p["id"] for p in prods])
    basura = [{**e, "event_type": "expired"} for e in ev]
    res = gbm.train(ev + basura, prods)
    assert res["n_products"] == 8


def test_train_filtra_por_segmento():
    dom = products(8, user_type="individuo")
    com = [{**p, "id": p["id"] + 100, "user_type": "comercio"} for p in products(8)]
    todos = dom + com
    ev = events_for([p["id"] for p in todos])

    res = gbm.train(ev, todos, segment="comercio")
    assert res["n_samples"] == 8


def test_train_sin_segmento_usa_todos_los_productos():
    dom = products(6, user_type="individuo")
    com = [{**p, "id": p["id"] + 100, "user_type": "comercio"} for p in products(6)]
    todos = dom + com
    res = gbm.train(events_for([p["id"] for p in todos]), todos)
    assert res["n_samples"] == 12


def test_train_segmento_sin_productos_falla():
    prods = products(8, user_type="individuo")
    with pytest.raises(ValueError):
        gbm.train(events_for([p["id"] for p in prods]), prods, segment="comercio")


def test_train_producto_sin_user_type_cuenta_como_individuo():
    prods = [{k: v for k, v in p.items() if k != "user_type"} for p in products(8)]
    res = gbm.train(events_for([p["id"] for p in prods]), prods, segment="individuo")
    assert res["n_samples"] == 8


def test_train_reporta_importancias():
    prods = products(8)
    res = gbm.train(events_for([p["id"] for p in prods]), prods)
    assert set(res["feature_importances"]) <= set(gbm.FEATURE_NAMES)
    assert len(res["feature_importances"]) == len(gbm.FEATURE_NAMES)


def test_train_svr_no_reporta_importancias():
    prods = products(8)
    res = gbm.train(events_for([p["id"] for p in prods]), prods, model_type="svr")
    assert res["feature_importances"] == {}


def test_train_categoria_desconocida_no_rompe():
    prods = products(8)
    prods[0]["categoria"] = None
    res = gbm.train(events_for([p["id"] for p in prods]), prods)
    assert res["n_samples"] == 8


def test_train_categoria_fuera_del_encoder_no_rompe(monkeypatch):
    """Un producto sin eventos queda fuera del fit del encoder pero igual se codifica."""
    prods = products(8)
    eventos = events_for([p["id"] for p in prods])
    # el encoder se entrena con las categorías de all_products; forzamos un fallo
    # de transform para ejercitar el fallback a 0.0
    import sklearn.preprocessing as pre

    original = pre.LabelEncoder.transform

    def transform_roto(self, values):
        raise ValueError("categoría no vista")

    monkeypatch.setattr(pre.LabelEncoder, "transform", transform_roto)
    res = gbm.train(eventos, prods)
    monkeypatch.setattr(pre.LabelEncoder, "transform", original)
    assert res["n_samples"] == 8
