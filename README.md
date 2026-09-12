# ML Service

Servicio de predicción de demanda e inventario. Implementa un pipeline de 3 etapas (Bayesiano → Moving Average → GBM/XGBoost/LightGBM/SVR) según la cantidad de historial disponible por producto.

---

## Dataset de prueba

El directorio `dataset/` contiene el pipeline completo para construir y validar el dataset de prueba usando dos fuentes públicas de Kaggle:

- **Store Item Demand Forecasting** — 913k registros diarios, 500 productos, 2013–2017
- **Logistics Warehouse Dataset** (CC0) — 3204 registros de parámetros de inventario

### Notebooks en Google Colab

| # | Notebook | Descripción |
|---|----------|-------------|
| 1 | [EDA y Normalización](https://drive.google.com/file/d/1ftM9NA4jAWD4U0LiYEir7iW_mM0EjH2k/view?usp=sharing) | Limpieza, construcción de product_id, cross-reference y simulación de stock |
| 2 | [Validación de Features](https://drive.google.com/file/d/1pXgXYcUiyV7tFzUmRHn5eBiCx7rzPh5M/view?usp=sharing) | Rolling means, freq_per_week, stock_ratio — verificación de causalidad temporal |
| 3 | [Entrenamiento y Evaluación](https://drive.google.com/file/d/1ku6knnQIBbo-6DECqTsCn84ONP46XPMC/view?usp=sharing) | GBM con split temporal, MAE/MAPE, feature importances |

### Cómo ejecutar

**Prerequisitos**

1. Tener una cuenta de Google con acceso a Google Drive y Google Colab
2. Descargar los dos datasets de Kaggle:
   - [Store Item Demand Forecasting](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data) → bajar `train.csv` y renombrarlo `stored_item_demand.csv`
   - [Logistics Warehouse Dataset](https://www.kaggle.com/datasets/ziya07/logistics-warehouse-dataset) → bajar `logistics_warehouse.csv`

**Estructura de carpetas en Drive**

Crear esta estructura en Google Drive antes de ejecutar:

```
Mi unidad/
└── tpp-ml-dataset/
    └── data/
        ├── raw/
        │   ├── stored_item_demand.csv
        │   ├── logistics_warehouse.csv
        │   └── metadata.json
        └── processed/        ← se genera automáticamente
```

El archivo `metadata.json` está en `dataset/data/raw/metadata.json` de este repo.

**Orden de ejecución**

Ejecutar las notebooks **en orden**, usando "Ejecutar todo" (Runtime → Run all) en cada una:

1. **Notebook 01** — genera `demand_normalized.csv`, `product_stats.csv`, `logistics_normalized.csv`, `product_params.csv` y `events.csv` en `data/processed/`
2. **Notebook 02** — genera `events_with_features.csv` con las features del pipeline ML
3. **Notebook 03** — entrena el modelo GBM y genera `evaluation_report.json` con MAE y MAPE

Cada notebook monta Google Drive automáticamente al ejecutarse — va a aparecer un popup pidiendo permiso, hay que aceptarlo.

### Resultados obtenidos

| Modelo | MAE | MAPE | Estado |
|--------|-----|------|--------|
| GBM (seed=42) | 6.43 | 13.2% | ✅ Aprobado (umbral: 25%) |

Las features más importantes son `rolling_mean_7` (55%) y `rolling_mean_14` (37%).

---

## Ejecución local

```bash
# Instalar dependencias
cd dataset
python3 -m venv .venv
.venv/bin/pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Ejecutar scripts
.venv/bin/python scripts/cross_reference.py
.venv/bin/python scripts/simulate_stock.py

# Abrir notebooks
.venv/bin/jupyter notebook notebooks/
```

Los CSVs grandes (`events.csv`, `events_with_features.csv`, `stored_item_demand.csv`) están en `.gitignore` y deben generarse localmente ejecutando los scripts en orden.
