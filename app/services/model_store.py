"""
Gestión de versiones del modelo GBM.
Persiste metadata en versions.json y el modelo en archivos .pkl dentro de MODEL_DIR.
"""
import json
import os
from datetime import datetime, timezone
from typing import Optional

import joblib

MODEL_DIR = os.getenv("MODEL_DIR", "/app/model")
VERSIONS_FILE = os.path.join(MODEL_DIR, "versions.json")


def _load_versions() -> list[dict]:
    if not os.path.exists(VERSIONS_FILE):
        return []
    with open(VERSIONS_FILE) as f:
        return json.load(f)


def _save_versions(versions: list[dict]) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(VERSIONS_FILE, "w") as f:
        json.dump(versions, f, indent=2, default=str)


def list_versions() -> list[dict]:
    return _load_versions()


def get_active_version() -> Optional[dict]:
    for v in _load_versions():
        if v.get("status") == "active":
            return v
    return None


def load_active_model():
    v = get_active_version()
    if v is None:
        return None
    path = os.path.join(MODEL_DIR, v["model_file"])
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def save_new_version(model, mae: float, mape: float, n_samples: int, n_products: int) -> dict:
    versions = _load_versions()

    version_id = f"v{len(versions) + 1}"
    model_file = f"gbm_{version_id}.pkl"

    new_version = {
        "id": version_id,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "mae": round(mae, 4),
        "mape": round(mape, 2),
        "n_samples": n_samples,
        "n_products": n_products,
        "status": "inactive",
        "model_file": model_file,
    }

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, model_file))

    versions.append(new_version)
    _save_versions(versions)
    return new_version


def activate_version(version_id: str) -> Optional[dict]:
    versions = _load_versions()
    target = None
    for v in versions:
        if v["id"] == version_id:
            v["status"] = "active"
            target = v
        elif v["status"] == "active":
            v["status"] = "inactive"
    if target is None:
        return None
    _save_versions(versions)
    return target
