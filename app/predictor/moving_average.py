import math

"""
    Stage 2 — Weighted moving average.
    Los registros más recientes tienen mayor peso (caen exponencial)
    Returns: consumo diario estimado (unidades/día).
"""
def predict(records: list[dict]) -> float:
 
    n = len(records)
    decay = 0.9

    weights = [decay ** (n - 1 - i) for i in range(n)]
    total_weight = sum(weights)

    weighted_sum = sum(r["quantity"] * w for r, w in zip(records, weights))
    weighted_avg_quantity = weighted_sum / total_weight

    if n < 2:
        return weighted_avg_quantity / 7  # asumir 1 evento por semana

    from datetime import datetime

    def parse_dt(s: str) -> datetime:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))

    intervals_days = []
    for i in range(1, n):
        delta = parse_dt(records[i]["consumed_at"]) - parse_dt(records[i - 1]["consumed_at"])
        intervals_days.append(max(delta.total_seconds() / 86400, 0.1))

    avg_interval = sum(intervals_days) / len(intervals_days)
    return weighted_avg_quantity / avg_interval
