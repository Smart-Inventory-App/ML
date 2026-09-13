"""
Dataset de prueba: historial de consumo simulado a partir de dos datasets
públicos de Kaggle (ver dataset/README.md). Se usa como fuente alternativa de
entrenamiento para demos y para evaluar el modelo de forma reproducible.

Devuelve los datos con la misma forma que `backend_client`, así el
entrenamiento no distingue entre esta fuente y los datos reales.
"""
import csv
import gzip
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "test_dataset"
EVENTS_FILE = DATA_DIR / "events.csv.gz"
PRODUCTS_FILE = DATA_DIR / "products.csv"


def _to_timestamp(d: date) -> str:
    return datetime.combine(d, time(12, 0), tzinfo=timezone.utc).isoformat()


def load_events() -> list[dict]:
    """
    Eventos de consumo con las fechas ancladas: el último día del dataset
    se mapea a hoy. Sin esto `days_since_last` valdría miles de días al
    entrenar y cero en producción, y el modelo aprendería un régimen que
    nunca va a ver.
    """
    rows: list[tuple[date, str, int]] = []
    with gzip.open(EVENTS_FILE, "rt", newline="") as f:
        for r in csv.DictReader(f):
            rows.append((
                date.fromisoformat(r["date"]),
                r["product_id"],
                int(r["quantity"]),
            ))

    if not rows:
        return []

    shift = date.today() - max(d for d, _, _ in rows)

    return [
        {
            "product_id": pid,
            "quantity": qty,
            "event_type": "consumed",
            "consumed_at": _to_timestamp(d + shift),
        }
        for d, pid, qty in rows
    ]


def load_products() -> list[dict]:
    with open(PRODUCTS_FILE, newline="") as f:
        return [
            {
                "id": r["id"],
                "nombre": r["nombre"],
                "categoria": r["categoria"],
                "stock": int(r["stock"]),
                "stock_minimo": int(r["stock_minimo"]),
            }
            for r in csv.DictReader(f)
        ]
