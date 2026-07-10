# Proyecto Final — Inteligencia Artificial

**Ingeniería de Software · Universidad La Salle**
**Docente:** Dr. Vicente Enrique Machaca Arceda
**Curso:** Inteligencia Artificial

---

## Objetivo

1. **Aplicación funcional** — integra una técnica de IA vista en clase, aplicada de principio a fin.
2. **Problema real** — resuelto con datos reales y verificables, no simulados.
3. **Prioridad práctica** — que el pipeline funcione completo, por encima de la complejidad matemática.

## Modalidad

- Equipos de hasta 5 integrantes, con roles definidos.
- Una técnica principal (combinable con otras si el problema lo justifica).
- Se puede partir de un paper o repo, siempre que se declaren las mejoras propias.

## Técnicas posibles

- **Búsqueda y optimización:** BFS, DFS, UCS, A*, heurísticas, hill climbing, simulated annealing, algoritmos genéticos.
- **Machine Learning (Scikit-learn):** regresión, clustering, árboles, random forest. Casos: precios, segmentación, fraude, riesgo.
- **Deep Learning (PyTorch/Keras):** MLP, CNN, RNN. Casos: imágenes, texto, series de tiempo.
- **Agentes:** búsqueda, entornos/juegos, LLM con herramientas.

> **Técnica elegida por este equipo:** Machine Learning — **regresión** con Scikit-learn.

## Requisitos de los datos

- Fuente verificable (Kaggle, UCI, datos abiertos, Hugging Face, APIs públicas).
- Documentar fuente, fecha de obtención, licencia y tamaño.
- Se valora usar datos de Perú o recolectados por el equipo.

## Pipeline exigido

1. **Datos:** obtención, EDA (estadísticas, distribuciones, faltantes), preprocesamiento (limpieza, codificación, normalización, split train/val/test).
2. **Modelo y evaluación:** técnica de IA + hiperparámetros, métricas apropiadas + validación k-fold, salida interpretable.

## Experimentos y métricas

- Mínimo **3 experimentos** comparando configuraciones/modelos/instancias.
- Métricas por tarea (MAE/F1/silueta/nodos expandidos/recompensa, según corresponda).
- Reporte con tablas y gráficos comparativos, con discusión.

## La aplicación

- Formato libre (app web, API, notebook, bot o CLI).
- Funcional (se ejecuta de principio a fin, con README claro).
- Reproducible (código, dependencias y datos o enlace incluidos).

## Entregables

- Repositorio Git/Colab reproducible con README e instrucciones.
- Presentación en vivo (máx. 5 min) con demo funcionando.
- Todos los integrantes con cámara activa, archivos sin comprimir.

## Rúbrica de evaluación (total: 20 puntos)

| Criterio | Puntos |
|---|:--:|
| Aplicación funcional: se ejecuta de principio a fin, interfaz clara, pipeline integrado | 5 |
| Datos y pipeline: datos reales justificados, pipeline documentado en el README | 4 |
| Técnica de IA: aplicada correctamente, con las librerías correspondientes, bien fundamentada | 4 |
| Experimentos y resultados: mínimo 3, con tablas y gráficos, discutidos | 4 |
| Presentación: máximo 5 min, demo funcionando, dominio del tema | 3 |
