"""
Entrenamiento del modelo final.

1. Carga el dataset limpio.
2. Separa train/test (80/20).
3. Selecciona el mejor modelo del catálogo por validación cruzada (RMSE).
4. Reentrena ese modelo con todo el train y lo evalúa en el test hold-out.
5. Guarda el pipeline entrenado + metadatos en models/.

Uso:
    python -m src.train
"""
from __future__ import annotations

import joblib
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from . import config as C
from .evaluate import formatear, metricas
from .features import cargar_modelado
from .models import catalogo_modelos, construir_pipeline


def seleccionar_mejor(X_train, y_train) -> tuple[str, object]:
    """Elige el modelo con menor RMSE en validación cruzada de 5 folds."""
    kf = KFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    resultados = {}
    print("Selección de modelo por validación cruzada (5-fold):")
    for nombre, base in catalogo_modelos().items():
        pipe = construir_pipeline(base, log_target=True)
        # scoring RMSE (negativo porque sklearn maximiza).
        scores = cross_val_score(
            pipe, X_train, y_train, cv=kf,
            scoring="neg_root_mean_squared_error", n_jobs=-1,
        )
        rmse = -scores.mean()
        resultados[nombre] = rmse
        print(f"  {nombre:24s} RMSE_cv = {rmse:8.1f}  (±{scores.std():.0f})")
    mejor = min(resultados, key=resultados.get)
    print(f"\nMejor modelo: {mejor}")
    return mejor, catalogo_modelos()[mejor]


def main() -> None:
    X, y = cargar_modelado()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    nombre, base = seleccionar_mejor(X_train, y_train)

    # Reentrenar el ganador con todo el train.
    modelo = construir_pipeline(base, log_target=True)
    modelo.fit(X_train, y_train)

    # Evaluación honesta en el conjunto de prueba nunca visto.
    pred = modelo.predict(X_test)
    m = metricas(y_test, pred)
    print(f"\n=== Desempeño en TEST (hold-out) ===\n{formatear(m)}")

    # Guardar modelo + metadatos.
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    artefacto = {
        "modelo": modelo,
        "nombre": nombre,
        "features": C.FEATURES,
        "num_features": C.NUM_FEATURES,
        "cat_features": C.CAT_FEATURES,
        "metricas_test": m,
        "usd_to_pen": C.USD_TO_PEN,
        # opciones para los desplegables de la app
        "distritos": sorted(X["distrito"].unique().tolist()),
        "tipos": sorted(X["tipo_vivienda"].unique().tolist()),
    }
    joblib.dump(artefacto, C.MODEL_PATH)
    print(f"\nModelo guardado en {C.MODEL_PATH}")


if __name__ == "__main__":
    main()
