"""
Ejecuta el pipeline completo de principio a fin, en orden.

    python run_pipeline.py

Equivale a correr, en secuencia:
    src.data_prep -> src.eda -> src.train -> src.experiments
"""
from __future__ import annotations

from src import data_prep, eda, experiments, train


def main() -> None:
    print("\n" + "#" * 70)
    print("# PASO 1/4 — Limpieza y preparación de datos")
    print("#" * 70)
    data_prep.construir_dataset()

    print("\n" + "#" * 70)
    print("# PASO 2/4 — Análisis exploratorio (EDA)")
    print("#" * 70)
    df = eda.cargar()
    eda.resumen(df)
    eda.graficos(df)

    print("\n" + "#" * 70)
    print("# PASO 3/4 — Entrenamiento y selección de modelo")
    print("#" * 70)
    train.main()

    print("\n" + "#" * 70)
    print("# PASO 4/4 — Experimentos comparativos")
    print("#" * 70)
    experiments.main()

    print("\n[OK] Pipeline completo. Lanza la app con:  streamlit run app/app.py")


if __name__ == "__main__":
    main()
