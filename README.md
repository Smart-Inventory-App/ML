# Machine Learning Service — Smart Inventory Platform

[![codecov](https://codecov.io/gh/Smart-Inventory-App/ML/branch/main/graph/badge.svg?token=MBCA7OZASE)](https://codecov.io/gh/Smart-Inventory-App/ML)

Servicio de predicción de consumo para Smart Inventory. Estima la tasa diaria de consumo de cada producto y los días restantes hasta agotar stock, que es lo que consume el job de alertas de `App_backend`.

---

## Cómo Predice

El modelo que se usa depende del historial disponible para ese producto:

| Etapa | Cuándo se aplica | Método |
|-------|------------------|--------|
| `bayesian` | menos de 10 eventos o menos de 4 semanas | Prior Beta conjugado (α=1, β=6) |
| `moving_average` | ≥ 10 eventos y entre 4 y 8 semanas | Media móvil ponderada (decay 0.9) |
| `gbm` | ≥ 10 eventos y ≥ 8 semanas | Modelo entrenado |

Los umbrales están en `app/core/config.py`. Para la etapa `gbm` se puede usar GBM (sklearn), XGBoost, LightGBM o SVR; se elige al entrenar desde el BackOffice.

## Segmentos

El servicio entrena y sirve **un modelo por segmento de usuario**. El consumo doméstico ronda 1–5 unidades semanales por producto y el comercial varios cientos, así que un modelo único quedaría dominado por el de mayor volumen.

| Segmento | Criterio de aceptación |
|----------|------------------------|
| `individuo` | MAE ≤ 2 u/sem y MAPE ≤ 25% |
| `comercio` | WAPE ≤ 25% y MAPE ≤ 25% |

El MAE absoluto de 2 u/sem es el umbral de la propuesta y aplica al segmento doméstico, que es el alcance del MVP. En comercios se evalúa con WAPE —error absoluto ponderado por volumen— porque el MAE no transfiere entre escalas: 15 u/sem es un 4% de error para un comercio y un disparate para una heladera.

Cada segmento mantiene su versión activa de forma independiente.

## Endpoints

Todos los de `/models` requieren el header `X-Internal-Token`.

| Método | Ruta | Uso |
|--------|------|-----|
| `GET` | `/health` | Estado y versión activa de cada segmento |
| `GET` | `/predict/{product_id}` | Predicción para un producto (requiere JWT de usuario) |
| `GET` | `/models/` | Lista de versiones entrenadas |
| `POST` | `/models/train` | Entrena y evalúa una versión nueva |
| `POST` | `/models/{id}/activate` | Activa una versión en su segmento |
| `GET` | `/models/{id}/feature-importances` | Importancia de features de una versión |
| `POST` | `/models/predict-product` | Predicción interna que usa el job de alertas |

`POST /models/train` acepta:

```json
{
  "model_type": "gbm",
  "data_source": "real",
  "segment": "individuo"
}
```

## Ejecución Local

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/uvicorn app.main:app --reload --port 8003
```

Variables relevantes: `APP_BACKEND_URL`, `INTERNAL_SERVICE_TOKEN`, `JWT_SECRET`,
`MODEL_DIR`.

## Tests

```bash
venv/bin/python -m pytest
```

`pytest.ini` exige un mínimo de 90% de cobertura; la corrida falla si baja.

---

## Dataset de Prueba

Además de los datos reales, el servicio puede entrenar con un dataset simulado (útil para demos y para evaluar el modelo de forma reproducible) eligiendo `data_source: "test"` desde el BackOffice.

Vive en `app/data/test_dataset/`: 913k eventos de 500 productos en 3 MB comprimidos. Es de **escala comercial**, así que solo aplica al segmento `comercio`.

Al cargarlo, las fechas se desplazan para que el último evento caiga en el día de ejecución. Sin ese anclaje `days_since_last` valdría unos 3.200 días al entrenar y cerca de 0 en producción, y el modelo aprendería un régimen que nunca va a encontrar al predecir.

Para regenerar los archivos a partir del pipeline:

```bash
cd dataset
python3 scripts/build_test_dataset.py
```

---

## Pipeline del dataset

El directorio `dataset/` contiene la construcción y validación del dataset a partir de dos fuentes públicas de Kaggle:

- **Store Item Demand Forecasting** — 913k registros diarios, 500 productos, 2013–2017
- **Logistics Warehouse Dataset** (CC0) — 3204 registros de parámetros de inventario

### Notebooks en Google Colab

| # | Notebook | Contenido |
|---|----------|-----------|
| 1 | [EDA y Normalización](https://drive.google.com/file/d/1ftM9NA4jAWD4U0LiYEir7iW_mM0EjH2k/view?usp=sharing) | Limpieza, cross-reference entre fuentes y simulación de stock |
| 2 | [Validación de Features](https://drive.google.com/file/d/1pXgXYcUiyV7tFzUmRHn5eBiCx7rzPh5M/view?usp=sharing) | Rolling means, freq_per_week, stock_ratio y verificación de causalidad |
| 3 | [Entrenamiento y Evaluación](https://drive.google.com/file/d/1ku6knnQIBbo-6DECqTsCn84ONP46XPMC/view?usp=sharing) | Entrenamiento, baselines y métricas |

### Requisitos

Descargar los dos datasets de Kaggle:

- [Store Item Demand Forecasting](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data) → `train.csv`, renombrado a `stored_item_demand.csv`
- [Logistics Warehouse Dataset](https://www.kaggle.com/datasets/ziya07/logistics-warehouse-dataset) → `logistics_warehouse.csv`

Y armar esta estructura en Google Drive:

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

`metadata.json` está en `dataset/data/raw/` de este repo.

### Orden de Ejecución

Las notebooks se ejecutan **en orden**, con Runtime → Run all. Cada una monta Drive al arrancar y pide permiso.

1. **Notebook 01** genera `demand_normalized.csv`, `product_stats.csv`, `logistics_normalized.csv`, `product_params.csv` y `events.csv`
2. **Notebook 02** genera `events_with_features.csv` con las features del pipeline
3. **Notebook 03** entrena, compara contra baselines y genera `evaluation_report.json`

### Resultados

Notebook 03, sobre el dataset simulado:

| Modelo | MAE | RMSE | MAPE |
|--------|----:|-----:|-----:|
| Último valor | 11,24 | 15,38 | 22,63% |
| Media móvil 7d | 8,98 | 11,85 | 17,65% |
| Media móvil 14d | 8,99 | 11,90 | 17,67% |
| **GBM** | **6,43** | **8,38** | **13,23%** |

El GBM mejora un 28% en MAE sobre el mejor baseline y es el único que baja del 15% de MAPE. Las features dominantes son `rolling_mean_7` (56%) y `rolling_mean_14` (37%).

> **La notebook y el servicio no resuelven el mismo problema.** La notebook
> predice la cantidad consumida en un día concreto, con una fila por
> producto-día y split temporal. El servicio predice la tasa diaria promedio del
> producto, con una fila por producto. Las features también difieren: la
> notebook usa `dow` y `month` como enteros y el servicio los codifica one-hot.
> Sus métricas no son comparables entre sí.

### Ejecución Local del Pipeline

```bash
cd dataset
python3 -m venv .venv
.venv/bin/pip install pandas numpy matplotlib seaborn scikit-learn jupyter

.venv/bin/python scripts/cross_reference.py
.venv/bin/python scripts/simulate_stock.py
.venv/bin/jupyter notebook notebooks/
```

Los CSVs grandes (`events.csv`, `events_with_features.csv`, `stored_item_demand.csv`) están en `.gitignore` y se generan localmente.
