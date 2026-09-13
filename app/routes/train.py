from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from app.core.config import settings
from app.predictor import gbm
from app.services import backend_client, model_store, test_dataset

router = APIRouter(prefix="/models", tags=["models"])

_token_header = APIKeyHeader(name="X-Internal-Token", auto_error=False)


def _verify_service_token(token: str | None = Security(_token_header)):
    if token != settings.INTERNAL_SERVICE_TOKEN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Token de servicio inválido")


@router.get("/", dependencies=[Depends(_verify_service_token)])
def list_models():
    return model_store.list_versions()


class TrainRequest(BaseModel):
    model_type: Literal["gbm", "xgboost", "lightgbm", "svr"] = "gbm"
    data_source: Literal["real", "test"] = "real"


@router.post("/train", dependencies=[Depends(_verify_service_token)])
async def trigger_training(body: TrainRequest = TrainRequest()):
    try:
        if body.data_source == "test":
            all_events = test_dataset.load_events()
            all_products = test_dataset.load_products()
        else:
            all_events = await backend_client.get_all_consumption()
            all_products = await backend_client.get_all_products()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Error obteniendo datos: {e}")

    try:
        result = gbm.train(all_events, all_products, model_type=body.model_type)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    mae = result["mae"]
    mae_relative = result["mae_relative"]
    mape = result["mape"]

    rejected = mae_relative > settings.MAX_MAE_RELATIVE or mape > settings.MAX_MAPE
    if rejected:
        return {
            "status": "rejected",
            "reason": (
                f"Métricas no aceptables — MAE relativo={mae_relative}% "
                f"(máx {settings.MAX_MAE_RELATIVE}%), MAPE={mape}% (máx {settings.MAX_MAPE}%)"
            ),
            "mae": mae,
            "mae_relative": mae_relative,
            "mape": mape,
            "n_samples": result["n_samples"],
            "model_type": body.model_type,
            "feature_importances": result["feature_importances"],
        }

    version = model_store.save_new_version(
        model=result["model"],
        scaler=result["scaler"],
        mae=mae,
        mae_relative=mae_relative,
        mape=mape,
        n_samples=result["n_samples"],
        n_products=result["n_products"],
        feature_importances=result["feature_importances"],
        model_type=body.model_type,
        data_source=body.data_source,
    )

    return {
        "status": "saved",
        "version": version,
        "model_type": body.model_type,
        "feature_importances": result["feature_importances"],
    }


@router.get("/{version_id}/feature-importances", dependencies=[Depends(_verify_service_token)])
def get_feature_importances(version_id: str):
    versions = model_store.list_versions()
    version = next((v for v in versions if v["id"] == version_id), None)
    if version is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Versión no encontrada")
    return {
        "version_id": version_id,
        "model_type": version.get("model_type", "gbm"),
        "feature_importances": version.get("feature_importances", {}),
    }


class PredictProductRequest(BaseModel):
    events: list[dict]
    product: dict


@router.post("/predict-product", dependencies=[Depends(_verify_service_token)])
def predict_product_internal(body: PredictProductRequest):
    """
    Predicción interna: recibe eventos y datos del producto, devuelve tasa diaria
    y días estimados hasta que se agote el stock. Usado por App_backend para alertas.
    """
    from app.predictor import bayesian, moving_average
    from app.predictor.stage import determine_stage

    consumed = [e for e in body.events if e.get("event_type", "consumed") == "consumed"]
    stage = determine_stage(consumed)

    if stage == "bayesian":
        daily_rate = bayesian.predict(consumed)
    elif stage == "moving_average":
        daily_rate = moving_average.predict(consumed)
    else:
        daily_rate = gbm.predict(consumed, body.product)

    stock = body.product.get("stock", 0)
    days_until_stockout = round(stock / daily_rate) if daily_rate > 0 else None

    return {
        "daily_rate": round(daily_rate, 4),
        "days_until_stockout": days_until_stockout,
        "stage": stage,
    }


@router.post("/{version_id}/activate", dependencies=[Depends(_verify_service_token)])
def activate_model(version_id: str):
    version = model_store.activate_version(version_id)
    if version is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Versión no encontrada")

    result = model_store.load_active_model()
    if result:
        model, scaler = result
        gbm.set_model(model, scaler)

    return {"status": "activated", "version": version}
