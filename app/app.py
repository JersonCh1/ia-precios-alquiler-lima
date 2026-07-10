"""
Aplicación web (Streamlit) — Estimador de alquiler en Lima.

Carga el pipeline entrenado (models/modelo_alquiler.joblib) y predice el
alquiler mensual de un inmueble a partir de sus características.

Uso:
    streamlit run app/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Permite importar 'src' cuando Streamlit ejecuta este archivo directamente.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402

st.set_page_config(page_title="Estimador de alquiler — Lima", page_icon="🏠", layout="centered")


@st.cache_resource
def cargar_modelo():
    if not C.MODEL_PATH.exists():
        return None
    return joblib.load(C.MODEL_PATH)


artefacto = cargar_modelo()

st.title("🏠 Estimador de alquiler en Lima")
st.caption(
    "Predice el alquiler mensual (soles) de un inmueble en Lima Metropolitana "
    "usando un modelo de Machine Learning entrenado con datos reales de Properati."
)

if artefacto is None:
    st.error(
        "No se encontró el modelo entrenado. Ejecuta primero:\n\n"
        "```\npython -m src.data_prep\npython -m src.train\n```"
    )
    st.stop()

modelo = artefacto["modelo"]
distritos = artefacto["distritos"]
tipos = artefacto["tipos"]
mae = artefacto["metricas_test"]["MAE"]

# --------------------------------------------------------------------------- #
# Formulario de entrada
# --------------------------------------------------------------------------- #
st.subheader("Características del inmueble")
col1, col2 = st.columns(2)
with col1:
    distrito = st.selectbox("Distrito", distritos,
                            index=distritos.index("Miraflores") if "Miraflores" in distritos else 0)
    tipo = st.selectbox("Tipo de vivienda", tipos,
                        index=tipos.index("Apartamento") if "Apartamento" in tipos else 0)
    area = st.slider("Área (m²)", 15, 500, 90, step=5)
with col2:
    dormitorios = st.number_input("Dormitorios", min_value=0, max_value=10, value=2)
    banos = st.number_input("Baños", min_value=0, max_value=10, value=2)
    antiguedad = st.slider("Antigüedad (años)", 0, 100, 10)

entrada = pd.DataFrame([{
    "area_m2": float(area),
    "dormitorios": float(dormitorios),
    "banos": float(banos),
    "antiguedad": float(antiguedad),
    "distrito": distrito,
    "tipo_vivienda": tipo,
}])

if st.button("Estimar alquiler", type="primary", use_container_width=True):
    pred = float(modelo.predict(entrada)[0])
    bajo, alto = max(0, pred - mae), pred + mae
    st.success(f"### Alquiler estimado: **S/ {pred:,.0f} / mes**")
    st.write(
        f"Rango probable (± error medio del modelo): "
        f"**S/ {bajo:,.0f} — S/ {alto:,.0f}**"
    )
    usd = pred / artefacto["usd_to_pen"]
    st.caption(f"≈ US$ {usd:,.0f} / mes  (tipo de cambio {artefacto['usd_to_pen']} S/US$)")

# --------------------------------------------------------------------------- #
# Ficha del modelo
# --------------------------------------------------------------------------- #
with st.expander("ℹ️ Detalles del modelo"):
    m = artefacto["metricas_test"]
    st.write(f"**Algoritmo:** {artefacto['nombre']} (con objetivo en escala log).")
    st.write("**Desempeño en el conjunto de prueba (hold-out):**")
    st.table(pd.DataFrame([{
        "MAE (S/)": f"{m['MAE']:,.0f}",
        "RMSE (S/)": f"{m['RMSE']:,.0f}",
        "R²": f"{m['R2']:.3f}",
        "MAPE": f"{m['MAPE']:.1f}%",
    }]))
    st.caption(
        "Fuente de datos: Properati Lima (Zenodo 10.5281/zenodo.7846211, "
        "CC-BY-4.0). El modelo es una herramienta orientativa, no una tasación oficial."
    )
