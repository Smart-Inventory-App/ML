from datetime import date, datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.core.security import get_current_user
from app.predictor import bayesian, gbm, moving_average
from app.predictor.stage import determine_stage
from app.services.backend_client import get_consumption_history, get_product

router = APIRouter(prefix="/predict", tags=["predict"])


def _parse_expiry(product: dict) -> date | None:
    raw = product.get("vencimiento")
    if raw is None:
        return None
    if isinstance(raw, date):
        return raw
    try:
        return date.fromisoformat(str(raw))
    except ValueError:
        return None


def _expiry_alert(stock: int, daily_rate: float, expiry: date | None) -> dict:
    """
    Compara el stock actual con el consumo predicho hasta el vencimiento.
    Retorna campos de alerta listos para incluir en la respuesta.
    """
    if expiry is None:
        return {"expiry_alert": False, "days_until_expiry": None, "units_at_risk": None}

    today = datetime.now(timezone.utc).date()
    days_left = (expiry - today).days

    if days_left <= 0:
        # ya venció
        return {
            "expiry_alert": True,
            "days_until_expiry": days_left,
            "units_at_risk": stock,
        }

    predicted_until_expiry = daily_rate * days_left
    units_at_risk = max(0.0, stock - predicted_until_expiry)

    return {
        "expiry_alert": units_at_risk > 0,
        "days_until_expiry": days_left,
        "units_at_risk": round(units_at_risk, 2),
    }


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

    consumed_records = [r for r in records if r.get("event_type", "consumed") == "consumed"]

    stage = determine_stage(consumed_records)

    if stage == "bayesian":
        daily_rate = bayesian.predict(consumed_records)
    elif stage == "moving_average":
        daily_rate = moving_average.predict(consumed_records)
    else:
        daily_rate = gbm.predict(consumed_records, product)

    stock = product.get("stock", 0)
    expiry = _parse_expiry(product)
    alert = _expiry_alert(stock, daily_rate, expiry)

    # días hasta que se agote el stock al ritmo predicho
    days_until_stockout = round(stock / daily_rate) if daily_rate > 0 else None

    return {
        "product_id": product_id,
        "product_name": product.get("nombre"),
        "current_stock": stock,
        "predicted_daily_consumption": round(daily_rate, 3),
        "predicted_consumption": round(daily_rate * days, 2),
        "prediction_days": days,
        "days_until_stockout": days_until_stockout,
        "model_stage": stage,
        "data_points": len(consumed_records),
        "expired_events": len(records) - len(consumed_records),
        **alert,
    }
