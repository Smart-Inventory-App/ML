"""
Fase 4: Simulación de stock temporal por producto.
Stock_t = Stock_(t-1) - Demanda_t + Reposicion_t
La reposición ocurre lead_time_days después de cruzar stock_minimo.
El stock se registra ANTES del consumo para evitar fuga de información.
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED = Path(__file__).parent.parent / "data" / "processed"
SEED = 42
rng = np.random.default_rng(SEED)


def simulate_product(dates: pd.Series, demands: np.ndarray, stock_initial: int,
                     stock_minimo: int, lead_time: int) -> np.ndarray:
    """
    Simula el stock diario para un producto.
    Devuelve array con el stock disponible AL INICIO de cada día (antes del consumo).
    """
    n = len(demands)
    stock = np.empty(n, dtype=float)
    current = float(stock_initial)
    pending_restock: list[tuple[int, float]] = []  # (día de llegada, cantidad)

    for t in range(n):
        # Aplicar reposiciones programadas para hoy
        arrivals = [(d, q) for d, q in pending_restock if d == t]
        for _, qty in arrivals:
            current += qty
        pending_restock = [(d, q) for d, q in pending_restock if d != t]

        # Registrar stock ANTES del consumo
        stock[t] = max(current, 0.0)

        # Consumo del día
        current = max(current - demands[t], 0.0)

        # Si bajamos del stock mínimo y no hay reposición en camino, programar una
        has_pending = any(d > t for d, _ in pending_restock)
        if current <= stock_minimo and not has_pending:
            restock_qty = stock_initial - current  # reponer hasta stock inicial
            arrival_day = t + max(1, lead_time)
            if arrival_day < n:
                pending_restock.append((arrival_day, restock_qty))

    return stock


def run():
    demand = pd.read_csv(PROCESSED / "demand_normalized.csv", parse_dates=["date"])
    params = pd.read_csv(PROCESSED / "product_params.csv")
    params_idx = params.set_index("product_id")

    demand = demand.sort_values(["product_id", "date"]).reset_index(drop=True)

    all_rows = []
    products = demand["product_id"].unique()

    print(f"Simulando stock para {len(products)} productos...")
    for i, pid in enumerate(products):
        if i % 100 == 0:
            print(f"  {i}/{len(products)}")

        mask = demand["product_id"] == pid
        grp = demand.loc[mask].copy()

        p = params_idx.loc[pid]
        stock_arr = simulate_product(
            dates=grp["date"],
            demands=grp["quantity"].to_numpy(dtype=float),
            stock_initial=int(p["stock_initial"]),
            stock_minimo=int(p["stock_minimo"]),
            lead_time=int(p["lead_time_days"]),
        )
        grp["stock"] = stock_arr
        grp["stock_minimo"] = int(p["stock_minimo"])
        all_rows.append(grp)

    events = pd.concat(all_rows, ignore_index=True)
    events = events[["date", "product_id", "quantity", "category", "stock", "stock_minimo"]]

    # Validación básica de causalidad temporal (sin fuga)
    negative_stock = (events["stock"] < 0).sum()
    zero_stock_min = (events["stock_minimo"] <= 0).sum()
    print(f"\nValidación:")
    print(f"  Filas con stock negativo (debe ser 0): {negative_stock}")
    print(f"  Filas con stock_minimo <= 0 (debe ser 0): {zero_stock_min}")

    out = PROCESSED / "events.csv"
    events.to_csv(out, index=False)
    print(f"\nevents.csv guardado: {len(events):,} filas")
    print(events.describe().round(2))
    return events


if __name__ == "__main__":
    run()
