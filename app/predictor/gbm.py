"""
Stage 3 — Gradient Boosting Regressor.
Requiere >= 8 semanas de historial. Hasta que haya modelo entrenado, delega a moving_average.
"""
from app.predictor import moving_average

_model = None


def set_model(model) -> None:
    global _model
    _model = model


def is_trained() -> bool:
    return _model is not None


def predict(records: list[dict], product: dict) -> float:
    if not is_trained():
        return moving_average.predict(records)
    features = _build_features(records, product)
    return float(_model.predict([features])[0])


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

    # one-hot encoding día de semana (0-6) y mes (1-12)
    dow = [1.0 if day_of_week == i else 0.0 for i in range(7)]
    mon = [1.0 if month == i else 0.0 for i in range(1, 13)]

    return [rolling_mean_7, rolling_mean_14, freq_per_week, days_since_last, stock] + dow + mon


def _compute_daily_rate(records: list[dict]) -> float:
    """Tasa real de consumo (unidades/día) para un conjunto de registros."""
    from datetime import datetime

    if len(records) < 2:
        return sum(r["quantity"] for r in records) / 7.0

    def parse_dt(s: str) -> datetime:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))

    span_days = (parse_dt(records[-1]["consumed_at"]) - parse_dt(records[0]["consumed_at"])).days
    if span_days == 0:
        return sum(r["quantity"] for r in records)
    return sum(r["quantity"] for r in records) / span_days


def train(all_events: list[dict], all_products: list[dict]) -> dict:
    """
    Entrena el GBM con todos los eventos de consumo disponibles.
    Retorna métricas {mae, mape, n_samples, n_products} o lanza ValueError si no hay suficientes datos.
    """
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.preprocessing import LabelEncoder
    from app.core.config import settings
    from datetime import datetime

    # Agrupar eventos por producto (solo "consumed")
    from collections import defaultdict
    by_product: dict[int, list[dict]] = defaultdict(list)
    for e in all_events:
        if e.get("event_type", "consumed") == "consumed":
            by_product[e["product_id"]].append(e)

    # Mapa de producto_id → producto
    product_map = {p["id"]: p for p in all_products}

    # Categorías para encoding
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

    # Split 80/20
    split = max(1, int(len(X) * 0.8))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae_daily = float(mean_absolute_error(y_test, y_pred))
    mae_weekly = mae_daily * 7

    # MAPE — evitar división por cero
    mask = y_test > 0
    mape = float(np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100) if mask.any() else 999.0

    return {
        "model": model,
        "mae": round(mae_weekly, 4),
        "mape": round(mape, 2),
        "n_samples": len(rows_X),
        "n_products": len(by_product),
    }
