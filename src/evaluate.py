"""Métricas de evaluación para regresión (compartidas por todo el proyecto)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def metricas(y_true, y_pred) -> dict:
    """Devuelve MAE, RMSE, R² y MAPE (%) — las métricas estándar de regresión.

    - MAE : error absoluto medio, en soles (fácil de interpretar).
    - RMSE: penaliza más los errores grandes, en soles.
    - R²  : proporción de varianza explicada (0-1, más alto mejor).
    - MAPE: error porcentual medio (independiente de la escala).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def formatear(m: dict) -> str:
    return (
        f"MAE={m['MAE']:.0f}  RMSE={m['RMSE']:.0f}  "
        f"R2={m['R2']:.3f}  MAPE={m['MAPE']:.1f}%"
    )
