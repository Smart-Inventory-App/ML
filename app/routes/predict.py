from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.core.security import get_current_user
from app.predictor import bayesian, gbm, moving_average
from app.predictor.stage import determine_stage
from app.services.backend_client import get_consumption_history, get_product

router = APIRouter(prefix="/predict", tags=["predict"])


@router.get("/{product_id}")
async def predict_consumption(
    product_id: int,
    request: Request,
    days: int = 7,
    current_user: dict = Depends(get_current_user),
):
    token = request.headers["authorization"].split(" ")[1]

    try:
        records = await get_consumption_history(product_id, token)
        product = await get_product(product_id, token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="No se pudo obtener datos del backend",
        )

    stage = determine_stage(records)

    if stage == "bayesian":
        daily_rate = bayesian.predict(records)
    elif stage == "moving_average":
        daily_rate = moving_average.predict(records)
    else:
        daily_rate = gbm.predict(records, product)

    return {
        "product_id": product_id,
        "product_name": product.get("nombre"),
        "predicted_daily_consumption": round(daily_rate, 3),
        "predicted_consumption": round(daily_rate * days, 2),
        "prediction_days": days,
        "model_stage": stage,
        "data_points": len(records),
    }
