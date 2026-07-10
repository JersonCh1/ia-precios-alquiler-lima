"""
Construcción de features: el preprocesador de Scikit-learn.

Encapsula TODO el preprocesamiento en un ColumnTransformer para que se
aplique de forma idéntica en entrenamiento y en la app (sin fuga de datos:
los imputadores/escaladores se ajustan solo con el train).
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C


def cargar_modelado() -> tuple[pd.DataFrame, pd.Series]:
    """Devuelve (X, y) listos para entrenar desde el CSV procesado."""
    df = pd.read_csv(C.DATA_PROCESSED)
    X = df[C.FEATURES].copy()
    y = df[C.TARGET].copy()
    return X, y


def construir_preprocesador() -> ColumnTransformer:
    """ColumnTransformer con dos ramas:

    - Numéricas: imputación por mediana (robusta a outliers y a los 183
      nulos de 'antiguedad') + estandarización (necesaria para modelos
      lineales; inocua para árboles).
    - Categóricas: imputación por la moda + One-Hot con handle_unknown para
      tolerar distritos no vistos en producción.
    """
    num = Pipeline(steps=[
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])
    cat = Pipeline(steps=[
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("num", num, C.NUM_FEATURES),
        ("cat", cat, C.CAT_FEATURES),
    ])
