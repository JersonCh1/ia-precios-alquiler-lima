"""
Configuración central del proyecto.

Todas las rutas, constantes y decisiones "mágicas" viven aquí para que el
pipeline sea reproducible y fácil de auditar. Ningún otro módulo debe
hardcodear rutas ni números sueltos.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Rutas del proyecto (relativas a la raíz del repo, sin importar desde dónde
# se ejecute el script).
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw" / "properati_lima.csv"
DATA_PROCESSED = ROOT / "data" / "processed" / "lima_alquiler_limpio.csv"

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "modelo_alquiler.joblib"

FIGURES_DIR = ROOT / "reports" / "figures"
TABLES_DIR = ROOT / "reports" / "tables"

# URL de descarga directa y estable (Zenodo, DOI 10.5281/zenodo.7846211).
DATA_URL = "https://zenodo.org/records/7846211/files/dataset.csv?download=1"
DATA_ENCODING = "latin-1"  # el CSV original NO es UTF-8 (verificado)

# --------------------------------------------------------------------------- #
# Reproducibilidad
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.20

# --------------------------------------------------------------------------- #
# Reglas de negocio / limpieza
# --------------------------------------------------------------------------- #
# El dataset mezcla precios en USD y en soles. Unificamos TODO a soles (PEN).
# Tipo de cambio venta promedio de abril 2023 (fecha de publicación del
# dataset), fuente BCRP / SBS ≈ 3.75 soles por dólar.
USD_TO_PEN = 3.75

# La fuente incluye avisos de otras ciudades (Arequipa, Piura, etc.). Este
# proyecto modela SOLO Lima Metropolitana + Callao para que los precios sean
# comparables. Estos son los tokens de "departamento/provincia" que aceptamos.
CIUDADES_LIMA = {"Lima", "Provincia de Lima", "Callao", "Gobierno Regional de Lima"}

# Filtros de saneamiento (quitan outliers imposibles / errores de tipeo).
MIN_AREA_M2 = 10        # menos de 10 m² no es una vivienda real
MAX_AREA_M2 = 2000      # más de 2000 m² es error de digitación para alquiler
MIN_PRECIO_PEN = 300    # alquiler mensual mínimo plausible en soles
MAX_PRECIO_PEN = 60000  # tope para descartar precios de venta mal etiquetados

# --------------------------------------------------------------------------- #
# Definición del problema
# --------------------------------------------------------------------------- #
TARGET = "precio_pen"  # variable objetivo: alquiler mensual en soles

# Features finales que usa el modelo (tras el parsing).
NUM_FEATURES = ["area_m2", "dormitorios", "banos", "antiguedad"]
CAT_FEATURES = ["distrito", "tipo_vivienda"]
FEATURES = NUM_FEATURES + CAT_FEATURES
