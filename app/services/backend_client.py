from typing import Any

import httpx

from app.core.config import settings

_INTERNAL_HEADERS = {"X-Internal-Token": settings.INTERNAL_SERVICE_TOKEN}


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


async def get_all_consumption() -> list[dict]:
    """Endpoint interno de App_backend — devuelve todos los eventos de consumo."""
    url = f"{settings.APP_BACKEND_URL}/internal/consumption"
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(url, headers=_INTERNAL_HEADERS)
        r.raise_for_status()
        return r.json()


async def get_all_products() -> list[dict]:
    """Endpoint interno de App_backend — devuelve todos los productos."""
    url = f"{settings.APP_BACKEND_URL}/internal/products"
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(url, headers=_INTERNAL_HEADERS)
        r.raise_for_status()
        return r.json()
