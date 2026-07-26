from typing import Any

import httpx

from app.core.config import settings


async def get_consumption_history(product_id: int, token: str) -> list[dict]:
    url = f"{settings.APP_BACKEND_URL}/products/{product_id}/consumption"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        return r.json()


async def get_product(product_id: int, token: str) -> dict[str, Any]:
    url = f"{settings.APP_BACKEND_URL}/products/{product_id}"
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        return r.json()
