from datetime import datetime
from typing import Literal

from app.core.config import settings

Stage = Literal["bayesian", "moving_average", "gbm"]


def determine_stage(records: list[dict]) -> Stage:
    n = len(records)

    if n == 0:
        return "bayesian"

    def parse_dt(s: str) -> datetime:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))

    oldest = parse_dt(records[0]["consumed_at"])
    newest = parse_dt(records[-1]["consumed_at"])
    days_of_data = (newest - oldest).days

    if n < settings.MIN_RECORDS_FOR_WMA or days_of_data < settings.MIN_DAYS_FOR_WMA:
        return "bayesian"

    if days_of_data >= settings.MIN_DAYS_FOR_GBM:
        return "gbm"

    return "moving_average"
