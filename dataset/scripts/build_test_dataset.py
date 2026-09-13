"""
Empaqueta el dataset simulado en los dos archivos que consume el servicio ML
como fuente de entrenamiento de prueba.

Entrada (generada por cross_reference.py + simulate_stock.py):
    data/processed/events.csv          — un registro de consumo por producto/día
    data/processed/product_params.csv  — parámetros por producto

Salida (en ML/app/data/test_dataset/):
    events.csv.gz   — date, product_id, quantity
    products.csv    — id, nombre, categoria, stock, stock_minimo

El stock de cada producto es el del último día simulado, para que la relación
stock/historial sea la misma que en producción: el stock es "ahora" y los
eventos son lo que pasó hasta ahora.

Uso:
    python3 scripts/build_test_dataset.py
"""
import csv
import gzip
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PROCESSED = BASE / "data" / "processed"
OUT = BASE.parent / "app" / "data" / "test_dataset"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Último stock observado por producto (el CSV viene ordenado por fecha)
    last_stock: dict[str, str] = {}
    n_events = 0

    with open(PROCESSED / "events.csv") as fin, \
         gzip.open(OUT / "events.csv.gz", "wt", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["date", "product_id", "quantity"])
        for row in csv.DictReader(fin):
            writer.writerow([row["date"], row["product_id"], row["quantity"]])
            last_stock[row["product_id"]] = row["stock"]
            n_events += 1

    with open(PROCESSED / "product_params.csv") as fin, \
         open(OUT / "products.csv", "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow(["id", "nombre", "categoria", "stock", "stock_minimo"])
        n_products = 0
        for row in csv.DictReader(fin):
            pid = row["product_id"]
            if pid not in last_stock:
                continue
            writer.writerow([
                pid,
                f"Producto {pid}",
                row["category"],
                int(float(last_stock[pid])),
                row["stock_minimo"],
            ])
            n_products += 1

    ev_mb = os.path.getsize(OUT / "events.csv.gz") / 1024 / 1024
    print(f"✓ {OUT}")
    print(f"  events.csv.gz  {n_events:,} eventos — {ev_mb:.2f} MB")
    print(f"  products.csv   {n_products:,} productos")


if __name__ == "__main__":
    main()
