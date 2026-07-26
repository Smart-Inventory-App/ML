"""
Stage 3 — Gradient Boosting Regressor.
este requiere suficiente historial (>= 8 semanas) para poder entrenarlo asi que por ahora delega a moving_average hasta tener el modelo entrenado.
"""
from app.predictor import moving_average

# Placeholder para el modelo entrenado (se carga desde disco en producción)
_model = None


def is_trained() -> bool:
    return _model is not None

"""
-> Si el modelo está entrenado, usa GBM si no, cae en moving_average.
"""
def predict(records: list[dict], product: dict) -> float:
    
    if not is_trained():
        return moving_average.predict(records)

    features = _build_features(records, product)
    return float(_model.predict([features])[0])

"""
    Features según la propuesta:
    - rolling mean últimos N registros
    - frecuencia de consumo (eventos/semana)
    - tiempo desde último consumo (días)
    - stock actual / stock mínimo
    """

def _build_features(records: list[dict], product: dict) -> list[float]:
    
    from datetime import datetime, timezone

    quantities = [r["quantity"] for r in records]
    now = datetime.now(timezone.utc)

    def parse_dt(s: str) -> datetime:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))

    rolling_mean_7 = sum(quantities[-7:]) / len(quantities[-7:]) if quantities else 0
    rolling_mean_14 = sum(quantities[-14:]) / len(quantities[-14:]) if quantities else 0

    if len(records) >= 2:
        span_days = (parse_dt(records[-1]["consumed_at"]) - parse_dt(records[0]["consumed_at"])).days or 1
        freq_per_week = len(records) / span_days * 7
    else:
        freq_per_week = 0

    days_since_last = (now - parse_dt(records[-1]["consumed_at"])).days if records else 999

    stock = product.get("stock", 0)

    return [rolling_mean_7, rolling_mean_14, freq_per_week, days_since_last, stock]
