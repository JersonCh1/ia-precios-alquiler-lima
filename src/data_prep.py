"""
Limpieza y parsing del dataset crudo de Properati.

El CSV original guarda casi todo como texto sucio:
    price   -> "USD900", "S/.3,100"  (monedas mezcladas)
    bedroom -> "2 dormitorios"
    area    -> "103 m²"
    location-> "Ur. Santa Cruz, Miraflores, Lima, Lima "

Este módulo convierte todo eso a columnas numéricas/categóricas limpias,
unifica la moneda a soles, filtra a Lima Metropolitana y descarta outliers
imposibles. Cada decisión está comentada y justificada.

Uso:
    python -m src.data_prep      # genera data/processed/lima_alquiler_limpio.csv
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config as C

# Año de referencia = año de publicación del dataset (Properati, abril 2023).
ANIO_REFERENCIA = 2023


# --------------------------------------------------------------------------- #
# Parsers de campos individuales
# --------------------------------------------------------------------------- #
def parse_precio(texto: str) -> tuple[float, str]:
    """'USD900' -> (900, 'USD'); 'S/.3,100' -> (3100, 'PEN').

    Devuelve (nan, '') si no se puede interpretar.
    """
    if not isinstance(texto, str):
        return np.nan, ""
    t = texto.strip()
    moneda = "USD" if "USD" in t or "$" in t else ("PEN" if "S/" in t else "")
    # Quitamos separadores de miles y nos quedamos con los dígitos + decimal.
    limpio = t.replace(",", "").replace("S/.", "").replace("S/", "")
    limpio = limpio.replace("USD", "").replace("$", "").strip()
    m = re.search(r"\d+(\.\d+)?", limpio)
    if not m:
        return np.nan, ""
    return float(m.group()), moneda


def primer_numero(texto: str) -> float:
    """Extrae el primer número de un texto ('2 dormitorios' -> 2.0)."""
    if not isinstance(texto, str):
        return np.nan
    m = re.search(r"\d+(\.\d+)?", texto.replace(",", ""))
    return float(m.group()) if m else np.nan


def parse_ubicacion(location: str) -> tuple[str, str]:
    """Devuelve (distrito, departamento) a partir del campo location.

    Formatos observados (separados por coma):
        [barrio, distrito, provincia, departamento]  -> 4 tokens
        [distrito, provincia, departamento]          -> 3 tokens
    El departamento es siempre el último token; el distrito es el token que
    está justo antes de la provincia (antepenúltimo). Con 2 tokens usamos el
    primero como distrito.
    """
    if not isinstance(location, str):
        return "", ""
    tokens = [x.strip() for x in location.split(",") if x.strip()]
    if not tokens:
        return "", ""
    departamento = tokens[-1]
    if len(tokens) >= 3:
        distrito = tokens[-3]
    else:
        distrito = tokens[0]
    return distrito, departamento


# --------------------------------------------------------------------------- #
# Pipeline de limpieza completo
# --------------------------------------------------------------------------- #
def cargar_crudo() -> pd.DataFrame:
    """Lee el CSV crudo con el encoding correcto (latin-1)."""
    if not C.DATA_RAW.exists():
        raise FileNotFoundError(
            f"No se encontró {C.DATA_RAW}. Ejecuta 'python -m src.data' para "
            "descargar el dataset desde Zenodo."
        )
    return pd.read_csv(C.DATA_RAW, encoding=C.DATA_ENCODING)


def limpiar(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma el DataFrame crudo en uno limpio y modelable."""
    out = pd.DataFrame(index=df.index)

    # --- Precio: parsear y unificar a soles -------------------------------- #
    precio_moneda = df["price"].apply(parse_precio)
    valor = precio_moneda.apply(lambda x: x[0])
    moneda = precio_moneda.apply(lambda x: x[1])
    # Convertimos USD -> PEN; los que ya están en PEN quedan igual.
    out["precio_pen"] = np.where(
        moneda.eq("USD"), valor * C.USD_TO_PEN, valor
    )
    out["moneda_original"] = moneda

    # --- Features numéricas ------------------------------------------------ #
    out["area_m2"] = df["area"].apply(primer_numero)
    out["dormitorios"] = df["bedroom"].apply(primer_numero)
    out["banos"] = df["bathroom"].apply(primer_numero)

    anio = pd.to_numeric(df["year_contruction"], errors="coerce")
    # Antigüedad en años; recortada a un rango físicamente posible.
    out["antiguedad"] = (ANIO_REFERENCIA - anio).clip(lower=0, upper=120)

    # --- Ubicación --------------------------------------------------------- #
    ubic = df["location"].apply(parse_ubicacion)
    out["distrito"] = ubic.apply(lambda x: x[0])
    out["departamento"] = ubic.apply(lambda x: x[1])

    # --- Tipo de vivienda -------------------------------------------------- #
    out["tipo_vivienda"] = df["housing_type"].astype(str).str.strip()

    return out


def filtrar(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra a Lima Metropolitana y descarta outliers imposibles."""
    n0 = len(df)

    # 1) Solo Lima Metropolitana + Callao (precios comparables).
    df = df[df["departamento"].isin(C.CIUDADES_LIMA)].copy()
    n_lima = len(df)

    # 2) Solo viviendas residenciales (fuera oficinas / locales comerciales).
    residenciales = {"Apartamento", "Casa", "Casa de playa", "Habitación"}
    df = df[df["tipo_vivienda"].isin(residenciales)].copy()

    # 3) El objetivo y las features clave no pueden ser nulos.
    df = df.dropna(subset=["precio_pen", "area_m2", "dormitorios", "banos"])

    # 4) Rangos plausibles (quitan errores de digitación y precios de venta).
    df = df[
        df["precio_pen"].between(C.MIN_PRECIO_PEN, C.MAX_PRECIO_PEN)
        & df["area_m2"].between(C.MIN_AREA_M2, C.MAX_AREA_M2)
        & df["dormitorios"].between(0, 15)
        & df["banos"].between(0, 15)
    ].copy()

    print(
        f"Filtrado: {n0} filas crudas -> {n_lima} en Lima -> "
        f"{len(df)} tras saneamiento."
    )
    return df


def agrupar_distritos_raros(df: pd.DataFrame, min_freq: int = 8) -> pd.DataFrame:
    """Distritos con muy pocos avisos se agrupan como 'Otros'.

    Evita que el modelo memorice categorías con 1-2 ejemplos (sobreajuste) y
    reduce la explosión de columnas del One-Hot.
    """
    conteo = df["distrito"].value_counts()
    frecuentes = conteo[conteo >= min_freq].index
    df["distrito"] = df["distrito"].where(df["distrito"].isin(frecuentes), "Otros")
    return df


def construir_dataset(guardar: bool = True) -> pd.DataFrame:
    """Ejecuta todo el pipeline de limpieza y (opcionalmente) lo guarda."""
    crudo = cargar_crudo()
    limpio = limpiar(crudo)
    limpio = filtrar(limpio)
    limpio = agrupar_distritos_raros(limpio)

    columnas = [C.TARGET, *C.FEATURES, "moneda_original"]
    limpio = limpio[columnas].reset_index(drop=True)

    if guardar:
        C.DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
        limpio.to_csv(C.DATA_PROCESSED, index=False, encoding="utf-8")
        print(f"Guardado: {C.DATA_PROCESSED}  ({len(limpio)} filas)")
    return limpio


if __name__ == "__main__":
    df = construir_dataset()
    print("\n--- Vista del dataset limpio ---")
    print(df.head(8).to_string())
    print("\n--- Distritos ---")
    print(df["distrito"].value_counts())
    print("\n--- Tipos de vivienda ---")
    print(df["tipo_vivienda"].value_counts())
    print("\n--- Resumen del objetivo (precio_pen) ---")
    print(df["precio_pen"].describe().round(1))
    print("\n--- Nulos restantes ---")
    print(df.isna().sum())
