"""
Los 3 experimentos comparativos del proyecto.

  EXP 1 — Comparación de modelos: 5 algoritmos, mismas features y split.
  EXP 2 — Efecto de la transformación log del objetivo y ablación de la
          feature 'distrito' (¿cuánto aporta la ubicación?).
  EXP 3 — Ajuste de hiperparámetros (GridSearchCV) del mejor modelo +
          importancia de features + gráfico predicho vs real.

Cada experimento imprime una tabla, la guarda en reports/tables/ (CSV + MD)
y produce figuras en reports/figures/.

Uso:
    python -m src.experiments
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split

from . import config as C
from .evaluate import metricas
from .features import cargar_modelado
from .models import catalogo_modelos, construir_pipeline

sns.set_theme(style="whitegrid", palette="deep")


def _split():
    X, y = cargar_modelado()
    return train_test_split(X, y, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE)


def _guardar_tabla(df: pd.DataFrame, nombre: str, titulo: str) -> None:
    C.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.TABLES_DIR / f"{nombre}.csv", index=False)
    with open(C.TABLES_DIR / f"{nombre}.md", "w", encoding="utf-8") as f:
        f.write(f"# {titulo}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"\n{titulo}\n{df.to_string(index=False)}")


def _guardar_fig(fig, nombre: str) -> None:
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(C.FIGURES_DIR / nombre, dpi=120, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# EXPERIMENTO 1 — Comparación de modelos
# --------------------------------------------------------------------------- #
def exp1_modelos(X_train, X_test, y_train, y_test) -> str:
    print("\n" + "=" * 70 + "\nEXPERIMENTO 1 — Comparación de modelos\n" + "=" * 70)
    kf = KFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    filas = []
    for nombre, base in catalogo_modelos().items():
        pipe = construir_pipeline(base, log_target=True)
        rmse_cv = -cross_val_score(
            pipe, X_train, y_train, cv=kf,
            scoring="neg_root_mean_squared_error", n_jobs=-1,
        ).mean()
        pipe.fit(X_train, y_train)
        m = metricas(y_test, pipe.predict(X_test))
        filas.append({
            "Modelo": nombre,
            "RMSE_cv": round(rmse_cv, 0),
            "MAE_test": round(m["MAE"], 0),
            "RMSE_test": round(m["RMSE"], 0),
            "R2_test": round(m["R2"], 3),
            "MAPE_test": round(m["MAPE"], 1),
        })
    tabla = pd.DataFrame(filas).sort_values("RMSE_test")
    _guardar_tabla(tabla, "exp1_modelos", "EXP 1 — Comparación de modelos")

    # Gráfico: R² y RMSE por modelo.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.barplot(data=tabla, y="Modelo", x="R2_test", ax=axes[0], color="#2a6f97")
    axes[0].set_title("R² en test (más alto = mejor)")
    sns.barplot(data=tabla, y="Modelo", x="RMSE_test", ax=axes[1], color="#e07a5f")
    axes[1].set_title("RMSE en test (más bajo = mejor)")
    fig.suptitle("Experimento 1 — Comparación de modelos")
    _guardar_fig(fig, "exp1_modelos.png")

    mejor = tabla.iloc[0]["Modelo"]
    print(f"\n-> Mejor por RMSE_test: {mejor}")
    return mejor


# --------------------------------------------------------------------------- #
# EXPERIMENTO 2 — Transformación log + ablación de 'distrito'
# --------------------------------------------------------------------------- #
def exp2_configuraciones(X_train, X_test, y_train, y_test) -> None:
    print("\n" + "=" * 70 + "\nEXPERIMENTO 2 — Log-target y ablación de features\n" + "=" * 70)
    base = lambda: RandomForestRegressor(
        n_estimators=300, random_state=C.RANDOM_STATE, n_jobs=-1
    )
    configs = [
        ("Sin log + todas las features", True, C.FEATURES),
        ("Con log + todas las features", False, C.FEATURES),
        ("Con log + SIN distrito (ablación)", False, [f for f in C.FEATURES if f != "distrito"]),
        ("Con log + solo numéricas", False, C.NUM_FEATURES),
    ]
    filas = []
    for etiqueta, sin_log, feats in configs:
        log = not sin_log
        # Reconstruimos un preprocesador restringido a 'feats'.
        Xtr, Xte = X_train[feats], X_test[feats]
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        num = [f for f in feats if f in C.NUM_FEATURES]
        cat = [f for f in feats if f in C.CAT_FEATURES]
        trans = []
        if num:
            trans.append(("num", Pipeline([("i", SimpleImputer(strategy="median")),
                                           ("s", StandardScaler())]), num))
        if cat:
            trans.append(("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                                           ("o", OneHotEncoder(handle_unknown="ignore"))]), cat))
        pre = ColumnTransformer(trans)
        pipe = Pipeline([("pre", pre), ("modelo", base())])
        if log:
            from sklearn.compose import TransformedTargetRegressor
            pipe = TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
        pipe.fit(Xtr, y_train)
        m = metricas(y_test, pipe.predict(Xte))
        filas.append({
            "Configuración": etiqueta,
            "MAE_test": round(m["MAE"], 0),
            "RMSE_test": round(m["RMSE"], 0),
            "R2_test": round(m["R2"], 3),
            "MAPE_test": round(m["MAPE"], 1),
        })
    tabla = pd.DataFrame(filas)
    _guardar_tabla(tabla, "exp2_configuraciones",
                   "EXP 2 — Transformación log y ablación de features")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(data=tabla, y="Configuración", x="R2_test", ax=ax, color="#2a6f97")
    ax.set_title("Experimento 2 — R² en test por configuración")
    ax.set_xlabel("R² (más alto = mejor)")
    ax.set_ylabel("")
    _guardar_fig(fig, "exp2_configuraciones.png")


# --------------------------------------------------------------------------- #
# EXPERIMENTO 3 — Tuning de hiperparámetros + interpretabilidad
# --------------------------------------------------------------------------- #
def exp3_tuning(X_train, X_test, y_train, y_test) -> None:
    print("\n" + "=" * 70 + "\nEXPERIMENTO 3 — Tuning (GridSearchCV) + importancia\n" + "=" * 70)
    # Modelo por defecto (referencia).
    base_def = RandomForestRegressor(
        n_estimators=300, random_state=C.RANDOM_STATE, n_jobs=-1
    )
    pipe_def = construir_pipeline(base_def, log_target=True)
    pipe_def.fit(X_train, y_train)
    m_def = metricas(y_test, pipe_def.predict(X_test))

    # Búsqueda en grilla sobre los hiperparámetros clave del bosque.
    grid = {
        "regressor__modelo__n_estimators": [300, 600],
        "regressor__modelo__max_depth": [None, 12, 20],
        "regressor__modelo__min_samples_leaf": [1, 2, 4],
        "regressor__modelo__max_features": ["sqrt", 1.0],
    }
    pipe = construir_pipeline(
        RandomForestRegressor(random_state=C.RANDOM_STATE, n_jobs=-1),
        log_target=True,
    )
    kf = KFold(n_splits=5, shuffle=True, random_state=C.RANDOM_STATE)
    busqueda = GridSearchCV(
        pipe, grid, cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    busqueda.fit(X_train, y_train)
    print(f"Mejores hiperparámetros: {busqueda.best_params_}")

    m_tuned = metricas(y_test, busqueda.predict(X_test))
    tabla = pd.DataFrame([
        {"Configuración": "Random Forest (default)", **{k: round(v, 3) if k == "R2" else round(v, 0) for k, v in m_def.items()}},
        {"Configuración": "Random Forest (tuned)", **{k: round(v, 3) if k == "R2" else round(v, 0) for k, v in m_tuned.items()}},
    ])
    _guardar_tabla(tabla, "exp3_tuning", "EXP 3 — Default vs Tuned")

    # Importancia de features (mapeando nombres tras el One-Hot).
    mejor = busqueda.best_estimator_.regressor_
    pre = mejor.named_steps["preprocesador"]
    rf = mejor.named_steps["modelo"]
    nombres = pre.get_feature_names_out()
    imp = pd.Series(rf.feature_importances_, index=nombres).sort_values(ascending=False)
    top = imp.head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=top.values, y=[n.split("__")[-1] for n in top.index], ax=ax, color="#2a6f97")
    ax.set_title("Experimento 3 — Importancia de features (top 15)")
    ax.set_xlabel("Importancia (Random Forest)")
    _guardar_fig(fig, "exp3_importancia.png")

    # Predicho vs real del modelo afinado.
    pred = busqueda.predict(X_test)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, pred, alpha=0.5, color="#2a6f97")
    lim = [0, max(y_test.max(), pred.max())]
    ax.plot(lim, lim, "--", color="red", label="predicción perfecta")
    ax.set_xlabel("Precio real (soles)")
    ax.set_ylabel("Precio predicho (soles)")
    ax.set_title("Modelo afinado — Predicho vs Real (test)")
    ax.legend()
    _guardar_fig(fig, "exp3_pred_vs_real.png")

    print("\nTop features:")
    print(top.round(3).to_string())


def main() -> None:
    X_train, X_test, y_train, y_test = _split()
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")
    exp1_modelos(X_train, X_test, y_train, y_test)
    exp2_configuraciones(X_train, X_test, y_train, y_test)
    exp3_tuning(X_train, X_test, y_train, y_test)
    print("\nExperimentos completos. Tablas en reports/tables/, figuras en reports/figures/")


if __name__ == "__main__":
    main()
