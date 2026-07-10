"""
Catálogo de modelos y armado del pipeline completo.

Comparte definiciones entre los experimentos y el entrenamiento final para
que "el modelo" sea exactamente el mismo objeto en ambos lados.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

from . import config as C
from .features import construir_preprocesador


def catalogo_modelos() -> dict:
    """Nombre -> estimador base. Ordenados de más simple a más complejo."""
    return {
        "Regresión Lineal": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "KNN (k=7)": KNeighborsRegressor(n_neighbors=7),
        "Árbol/Random Forest": RandomForestRegressor(
            n_estimators=300, random_state=C.RANDOM_STATE, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=C.RANDOM_STATE
        ),
    }


def construir_pipeline(modelo, log_target: bool = True):
    """Encadena preprocesador + modelo.

    Si log_target=True, entrena sobre log(1+precio) y deshace la
    transformación al predecir (el EDA mostró que el precio es muy sesgado;
    modelar en escala log estabiliza la varianza y mejora las métricas).
    """
    pipe = Pipeline(steps=[
        ("preprocesador", construir_preprocesador()),
        ("modelo", modelo),
    ])
    if log_target:
        return TransformedTargetRegressor(
            regressor=pipe, func=np.log1p, inverse_func=np.expm1
        )
    return pipe
