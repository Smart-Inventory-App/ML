"""
Stage 3 — Gradient Boosting / XGBoost / LightGBM / SVR.
Requiere >= 8 semanas de historial. Hasta que haya modelo entrenado, delega a moving_average.
"""
from app.predictor import moving_average

# Solo se normalizan las features numéricas continuas; las one-hot no se tocan.
_NUMERIC_FEATURES = 6   # rolling_mean_7, rolling_mean_14, freq_per_week, days_since_last, stock, stock_ratio

FEATURE_NAMES = [
    "rolling_mean_7",
    "rolling_mean_14",
    "freq_per_week",
    "days_since_last",
    "stock",
    "stock_ratio",
    *[f"dow_{i}" for i in range(7)],
    *[f"mon_{i}" for i in range(1, 13)],
    "categoria",
]

_model = None
_scaler = None


def set_model(model, scaler=None) -> None:
    global _model, _scaler
    _model = model
    _scaler = scaler


def is_trained() -> bool:
    return _model is not None


def predict(records: list[dict], product: dict) -> float:
    if not is_trained():
        return moving_average.predict(records)
    import numpy as np
    features = _build_features(records, product)
    X = np.array([features])
    if _scaler is not None:
        X[:, :_NUMERIC_FEATURES] = _scaler.transform(X[:, :_NUMERIC_FEATURES])
    return float(_model.predict(X)[0])


def _build_features(records: list[dict], product: dict) -> list[float]:
    from datetime import datetime, timezone

    quantities = [r["quantity"] for r in records]
    now = datetime.now(timezone.utc)

    def parse_dt(s: str) -> datetime:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))

    rolling_mean_7 = sum(quantities[-7:]) / len(quantities[-7:]) if quantities else 0.0
    rolling_mean_14 = sum(quantities[-14:]) / len(quantities[-14:]) if quantities else 0.0

    if len(records) >= 2:
        span_days = (parse_dt(records[-1]["consumed_at"]) - parse_dt(records[0]["consumed_at"])).days or 1
        freq_per_week = len(records) / span_days * 7
    else:
        freq_per_week = 0.0

    days_since_last = (now - parse_dt(records[-1]["consumed_at"])).days if records else 999

    last_dt = parse_dt(records[-1]["consumed_at"]) if records else now
    day_of_week = last_dt.weekday()
    month = last_dt.month

    stock = product.get("stock", 0)
    stock_minimo = product.get("stock_minimo") or 0
    stock_ratio = stock / stock_minimo if stock_minimo > 0 else 0.0

    dow = [1.0 if day_of_week == i else 0.0 for i in range(7)]
    mon = [1.0 if month == i else 0.0 for i in range(1, 13)]

    return [rolling_mean_7, rolling_mean_14, freq_per_week, days_since_last, stock, stock_ratio] + dow + mon


def _compute_daily_rate(records: list[dict]) -> float:
    from datetime import datetime

    if len(records) < 2:
        return sum(r["quantity"] for r in records) / 7.0

    def parse_dt(s: str) -> datetime:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))

    span_days = (parse_dt(records[-1]["consumed_at"]) - parse_dt(records[0]["consumed_at"])).days
    if span_days == 0:
        return sum(r["quantity"] for r in records)
    return sum(r["quantity"] for r in records) / span_days


def _make_regressor(model_type: str):
    """Instancia el regresor según model_type."""
    try:
        if model_type == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, verbosity=0)
        if model_type == "lightgbm":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, verbose=-1)
        if model_type == "svr":
            from sklearn.svm import SVR
            return SVR(kernel="rbf", C=1.0, epsilon=0.1)
    except ImportError as e:
        raise ValueError(f"Librería no disponible en este entorno: {e}")
    # default: gbm
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)


def train(all_events: list[dict], all_products: list[dict], model_type: str = "gbm") -> dict:
    """
    Entrena el modelo seleccionado con todos los eventos de consumo disponibles.
    Aplica StandardScaler sobre features numéricas y calcula importancia de features.
    Retorna métricas {mae, mape, n_samples, n_products, feature_importances, model_type} o lanza ValueError.
    """
    import numpy as np
    from sklearn.metrics import mean_absolute_error
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from app.core.config import settings
    from datetime import datetime

    from collections import defaultdict
    by_product: dict[int, list[dict]] = defaultdict(list)
    for e in all_events:
        if e.get("event_type", "consumed") == "consumed":
            by_product[e["product_id"]].append(e)

    product_map = {p["id"]: p for p in all_products}

    all_cats = list({p.get("categoria", "") or "" for p in all_products})
    le = LabelEncoder()
    le.fit(all_cats)

    def encode_cat(cat: str) -> float:
        try:
            return float(le.transform([cat or ""])[0])
        except Exception:
            return 0.0

    def parse_dt(s: str) -> datetime:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))

    rows_X, rows_y = [], []

    for pid, records in by_product.items():
        records = sorted(records, key=lambda r: r["consumed_at"])
        product = product_map.get(pid, {})

        oldest = parse_dt(records[0]["consumed_at"])
        newest = parse_dt(records[-1]["consumed_at"])
        days_span = (newest - oldest).days

        if len(records) < settings.MIN_RECORDS_FOR_WMA or days_span < settings.MIN_DAYS_FOR_GBM:
            continue

        features = _build_features(records, product)
        cat_enc = encode_cat(product.get("categoria", "") or "")
        features.append(cat_enc)

        target = _compute_daily_rate(records)
        rows_X.append(features)
        rows_y.append(target)

    if len(rows_X) < 5:
        raise ValueError(f"Datos insuficientes para entrenar: {len(rows_X)} productos calificados (mínimo 5)")

    X = np.array(rows_X)
    y = np.array(rows_y)

    # Normalizar solo features numéricas continuas (primeras _NUMERIC_FEATURES columnas)
    scaler = StandardScaler()
    X[:, :_NUMERIC_FEATURES] = scaler.fit_transform(X[:, :_NUMERIC_FEATURES])

    # Split 80/20
    split = max(1, int(len(X) * 0.8))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = _make_regressor(model_type)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae_daily = float(mean_absolute_error(y_test, y_pred))
    mae_weekly = mae_daily * 7

    # Error relativo ponderado por volumen (WAPE). El MAE absoluto no es
    # comparable entre escalas: 15 u/sem es 4% para un comercio que mueve 366
    # u/sem y un disparate para una heladera que consume 3.
    denom = float(np.sum(np.abs(y_test)))
    mae_relative = float(np.sum(np.abs(y_test - y_pred)) / denom * 100) if denom > 0 else 999.0

    mask = y_test > 0
    mape = float(np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100) if mask.any() else 999.0

    # Importancia de features — SVR no la soporta nativamente
    if hasattr(model, "feature_importances_"):
        feature_importances = {
            name: round(float(imp), 6)
            for name, imp in sorted(
                zip(FEATURE_NAMES, model.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )
        }
    else:
        feature_importances = {}

    return {
        "model": model,
        "scaler": scaler,
        "mae": round(mae_weekly, 4),
        "mae_relative": round(mae_relative, 2),
        "mape": round(mape, 2),
        "n_samples": len(rows_X),
        "n_products": len(by_product),
        "feature_importances": feature_importances,
        "model_type": model_type,
    }
