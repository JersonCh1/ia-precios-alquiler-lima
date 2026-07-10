# Cómo este proyecto cumple la rúbrica (20 pts)

Guía rápida para la defensa: dónde está cada requisito y qué decir.

| Criterio (pts) | ¿Cómo lo cumple el proyecto? | Evidencia |
|---|---|:--:|
| **Aplicación funcional (5)** | App **Streamlit** que carga el modelo entrenado y estima el alquiler en tiempo real, con rango de confianza y equivalente en USD. Corre de principio a fin. | `app/app.py`, `reports/figures/app_streamlit.png` |
| **Datos y pipeline (4)** | Datos **reales** de Properati Lima (Zenodo, DOI, CC-BY-4.0, 1,079 filas). Pipeline documentado: descarga → limpieza → EDA → preprocesamiento → modelo → evaluación. | `README.md` §2–§6, `src/` |
| **Técnica de IA (4)** | **Regresión** con Scikit-learn (Random Forest + objetivo log vía `TransformedTargetRegressor`), preprocesamiento con `ColumnTransformer`, selección por validación cruzada. Bien fundamentada. | `README.md` §4, `src/models.py`, `src/train.py` |
| **Experimentos y resultados (4)** | **3 experimentos** con tablas y gráficos: (1) comparación de 5 modelos, (2) log-target + ablación de features, (3) tuning GridSearchCV + importancia. Con discusión honesta. | `README.md` §7, `reports/tables/`, `reports/figures/` |
| **Presentación (3)** | Demo en vivo de la app + recorrido por los hallazgos. Guion abajo. | esta guía |

---

## Guion de 5 minutos (sugerido)

1. **(30 s) Problema.** El mercado de alquiler en Lima publica precios en soles y dólares sin estandarizar; no hay referencia objetiva. Predecimos el alquiler mensual con ML.
2. **(45 s) Datos.** Properati Lima, Zenodo (DOI, CC-BY-4.0), 1,079 avisos reales. Mostrar la mezcla de monedas y cómo se limpió (867 filas finales).
3. **(60 s) Pipeline + técnica.** Diagrama del README: limpieza → EDA → `ColumnTransformer` → modelos. Por qué regresión y por qué log del precio (EDA: sesgo).
4. **(90 s) Experimentos.** Los 3 con sus tablas:
   - Árboles ganan; los lineales dan R² negativo.
   - El log mejora; la ablación revela que el One-Hot de distrito sobreajusta con pocos datos.
   - El tuning no mueve la aguja → el techo lo pone la data.
5. **(75 s) Demo en vivo.** `streamlit run app/app.py`: Miraflores 90 m² ≈ S/3,200 vs Comas 60 m² ≈ S/1,300. Explicar el rango de confianza.
6. **(cierre) Conclusión + honestidad.** R² ≈ 0.40: modesto pero real; el preprocesamiento y el tamaño de muestra pesan más que el algoritmo. Trabajo futuro: zonas socioeconómicas, más features.

---

## Roles del equipo (plantilla — completar)

| Integrante | Rol sugerido |
|---|---|
| … | Datos y limpieza (`data_prep.py`) |
| … | EDA y visualización (`eda.py`, notebook) |
| … | Modelado y experimentos (`train.py`, `experiments.py`) |
| … | App y despliegue (`app.py`) |
| … | Documentación y presentación (`README.md`, defensa) |
