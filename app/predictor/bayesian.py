from app.core.config import settings


def predict(records: list[dict]) -> float:
    """
    Estima consumo diario usando una distribución Beta conjugada.
    Prior global: alpha unidades consumidas, beta días sin consumo.
    Returns: consumo diario estimado (unidades/día).
    """
    alpha = settings.BAYESIAN_PRIOR_ALPHA
    beta = settings.BAYESIAN_PRIOR_BETA

    observed_units = sum(r["quantity"] for r in records)
    observed_days = len(records)

    posterior_mean = (alpha + observed_units) / (alpha + beta + observed_days)
    return posterior_mean
