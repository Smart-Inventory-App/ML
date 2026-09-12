"""
Fase 3: Cross-reference entre product_stats y logistics_normalized.
Para cada producto asigna stock_minimo (mediana de reorder_point de vecinos)
y lead_time_days (mediana de vecinos), usando matching por categoría y demanda.
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED = Path(__file__).parent.parent / "data" / "processed"
K = 10   # vecinos más cercanos
EPS = 1e-6
SEED = 42


def match_product(row: pd.Series, logistics_cat: pd.DataFrame) -> dict:
    """Selecciona k vecinos del dataset de logística por distancia relativa de demanda."""
    if logistics_cat.empty:
        return {"stock_minimo": np.nan, "lead_time_days": np.nan}

    mu = row["mean_daily_demand"]
    dist = (logistics_cat["daily_demand"] - mu).abs() / (mu + EPS)
    neighbors = logistics_cat.nsmallest(K, "dist_tmp") if "dist_tmp" in logistics_cat else logistics_cat.assign(dist_tmp=dist).nsmallest(K, "dist_tmp")

    return {
        "stock_minimo": neighbors["reorder_point"].median(),
        "lead_time_days": neighbors["lead_time_days"].median(),
        "stock_level_ref": neighbors["stock_level"].median(),
    }


def run():
    product_stats = pd.read_csv(PROCESSED / "product_stats.csv")
    logistics = pd.read_csv(PROCESSED / "logistics_normalized.csv")

    # Agrupar logística por categoría para filtrar rápido
    logistics_by_cat = {cat: grp.reset_index(drop=True) for cat, grp in logistics.groupby("category")}

    records = []
    for _, row in product_stats.iterrows():
        cat = row["category"]
        lcat = logistics_by_cat.get(cat, pd.DataFrame())

        mu = row["mean_daily_demand"]

        if not lcat.empty:
            lcat = lcat.copy()
            lcat["dist_tmp"] = (lcat["daily_demand"] - mu).abs() / (mu + EPS)
            neighbors = lcat.nsmallest(K, "dist_tmp")
        else:
            neighbors = pd.DataFrame()

        if neighbors.empty:
            stock_minimo = logistics["reorder_point"].median()
            lead_time = logistics["lead_time_days"].median()
            stock_level_ref = logistics["stock_level"].median()
        else:
            stock_minimo = neighbors["reorder_point"].median()
            lead_time = neighbors["lead_time_days"].median()
            stock_level_ref = neighbors["stock_level"].median()

        records.append({
            "product_id": row["product_id"],
            "category": cat,
            "mean_daily_demand": mu,
            "std_daily_demand": row["std_daily_demand"],
            "stock_minimo": round(stock_minimo),
            "lead_time_days": round(lead_time),
            "stock_initial": round(stock_level_ref),
        })

    params = pd.DataFrame(records)
    out = PROCESSED / "product_params.csv"
    params.to_csv(out, index=False)

    print(f"product_params.csv guardado: {len(params)} productos")
    print("\nResumen por categoría:")
    print(
        params.groupby("category")[["stock_minimo", "lead_time_days", "stock_initial"]]
        .agg(["mean", "median"])
        .round(1)
        .to_string()
    )
    print("\nPrimeras filas:")
    print(params.head(10).to_string(index=False))
    return params


if __name__ == "__main__":
    run()
