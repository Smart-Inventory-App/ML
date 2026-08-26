from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from app.core.config import settings
from app.predictor import gbm
from app.services import backend_client, model_store

router = APIRouter(prefix="/models", tags=["models"])

_token_header = APIKeyHeader(name="X-Internal-Token", auto_error=False)


def _verify_service_token(token: str | None = Security(_token_header)):
    if token != settings.INTERNAL_SERVICE_TOKEN:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Token de servicio inválido")


@router.get("/", dependencies=[Depends(_verify_service_token)])
def list_models():
    return model_store.list_versions()


@router.post("/train", dependencies=[Depends(_verify_service_token)])
async def trigger_training():
    try:
        all_events = await backend_client.get_all_consumption()
        all_products = await backend_client.get_all_products()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Error obteniendo datos: {e}")

    try:
        result = gbm.train(all_events, all_products)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))

    mae = result["mae"]
    mape = result["mape"]

    rejected = mae > 2.0 or mape > 25.0
    if rejected:
        return {
            "status": "rejected",
            "reason": f"Métricas no aceptables — MAE={mae} (máx 2.0), MAPE={mape}% (máx 25%)",
            "mae": mae,
            "mape": mape,
            "n_samples": result["n_samples"],
        }

    version = model_store.save_new_version(
        model=result["model"],
        mae=mae,
        mape=mape,
        n_samples=result["n_samples"],
        n_products=result["n_products"],
    )

    return {"status": "saved", "version": version}


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

    model = model_store.load_active_model()
    if model:
        gbm.set_model(model)

    return {"status": "activated", "version": version}
