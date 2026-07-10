"""
Análisis Exploratorio de Datos (EDA).

Genera estadísticos y figuras que justifican las decisiones de
preprocesamiento y modelado. Todas las figuras se guardan en
reports/figures/ para incrustarlas en el README.

Uso:
    python -m src.eda
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: corre en servidores/CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config as C

sns.set_theme(style="whitegrid", palette="deep")


def _guardar(fig, nombre: str) -> None:
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ruta = C.FIGURES_DIR / nombre
    fig.savefig(ruta, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura -> {ruta.name}")


def cargar() -> pd.DataFrame:
    if not C.DATA_PROCESSED.exists():
        raise FileNotFoundError(
            "Falta el dataset limpio. Ejecuta primero: python -m src.data_prep"
        )
    return pd.read_csv(C.DATA_PROCESSED)


def resumen(df: pd.DataFrame) -> None:
    print("\n=== RESUMEN GENERAL ===")
    print(f"Filas: {len(df)}  |  Columnas: {df.shape[1]}")
    print("\nEstadísticos numéricos:")
    print(df[C.NUM_FEATURES + [C.TARGET]].describe().round(1).to_string())
    print("\nProporción de moneda original de los avisos:")
    print((df["moneda_original"].value_counts(normalize=True) * 100).round(1))


def graficos(df: pd.DataFrame) -> None:
    print("\n=== GENERANDO FIGURAS ===")

    # 1) Distribución del precio (cruda vs log) --------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[C.TARGET], bins=40, kde=True, ax=axes[0], color="#2a6f97")
    axes[0].set_title("Precio de alquiler (soles) — sesgado a la derecha")
    axes[0].set_xlabel("Soles / mes")
    sns.histplot(np.log1p(df[C.TARGET]), bins=40, kde=True, ax=axes[1], color="#e07a5f")
    axes[1].set_title("log(1+precio) — casi normal")
    axes[1].set_xlabel("log(soles)")
    fig.suptitle("Distribución del objetivo: motiva la transformación logarítmica")
    _guardar(fig, "01_distribucion_precio.png")

    # 2) Precio por tipo de vivienda ------------------------------------- #
    fig, ax = plt.subplots(figsize=(8, 4))
    orden = df.groupby("tipo_vivienda")[C.TARGET].median().sort_values().index
    sns.boxplot(data=df, x="tipo_vivienda", y=C.TARGET, order=orden, ax=ax)
    ax.set_title("Precio por tipo de vivienda")
    ax.set_xlabel("")
    ax.set_ylabel("Soles / mes")
    _guardar(fig, "02_precio_por_tipo.png")

    # 3) Top distritos por precio mediano -------------------------------- #
    top = (
        df.groupby("distrito")[C.TARGET]
        .median()
        .sort_values(ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=top.values, y=top.index, ax=ax, color="#2a6f97")
    ax.set_title("Top 15 distritos por alquiler mediano")
    ax.set_xlabel("Soles / mes (mediana)")
    _guardar(fig, "03_precio_por_distrito.png")

    # 4) Área vs precio --------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=df, x="area_m2", y=C.TARGET, hue="dormitorios",
        palette="viridis", alpha=0.6, ax=ax,
    )
    ax.set_title("Área vs precio (color = dormitorios)")
    ax.set_xlabel("Área (m²)")
    ax.set_ylabel("Soles / mes")
    _guardar(fig, "04_area_vs_precio.png")

    # 5) Correlación entre numéricas ------------------------------------- #
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df[C.NUM_FEATURES + [C.TARGET]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlación de Pearson")
    _guardar(fig, "05_correlacion.png")

    print("\nCorrelación de cada feature con el precio:")
    print(corr[C.TARGET].drop(C.TARGET).sort_values(ascending=False).round(3))


if __name__ == "__main__":
    df = cargar()
    resumen(df)
    graficos(df)
    print("\nEDA completo. Figuras en reports/figures/")
