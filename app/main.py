from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.predictor import gbm
from app.routes.predict import router as predict_router
from app.routes.train import router as train_router
from app.services import model_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = model_store.load_active_model()
    if model:
        gbm.set_model(model)
    yield


app = FastAPI(title="ML Service — Smart Inventory", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)
app.include_router(train_router)


@app.get("/health")
def health():
    from app.services.model_store import get_active_version
    active = get_active_version()
    return {
        "status": "ok",
        "model_loaded": gbm.is_trained(),
        "active_version": active["id"] if active else None,
    }
