from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    JWT_SECRET: str = "change-this-secret-in-production"
    JWT_ALGORITHM: str = "HS256"

    APP_BACKEND_URL: str = "http://localhost:8000"
    INTERNAL_SERVICE_TOKEN: str = "change-this-internal-token"

    # Stage thresholds
    MIN_RECORDS_FOR_WMA: int = 10
    MIN_DAYS_FOR_WMA: int = 28      # 4 semanas
    MIN_DAYS_FOR_GBM: int = 56      # 8 semanas

    # numeros para la Bayesiana !!!encontrar una manera de traer consumos de manera externa!!!
    BAYESIAN_PRIOR_ALPHA: float = 1.0
    BAYESIAN_PRIOR_BETA: float = 6.0

    # Criterio de aceptación. Ambas métricas son relativas para que el mismo
    # umbral sirva en escala doméstica (unidades/semana) y comercial (cientos).
    MAX_MAE_RELATIVE: float = 25.0  # WAPE — error ponderado por volumen
    MAX_MAPE: float = 25.0


settings = Settings()
