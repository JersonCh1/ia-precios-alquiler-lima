"""
Descarga del dataset crudo desde Zenodo (fuente reproducible).

El CSV ya viene versionado en data/raw/ (licencia CC-BY-4.0 permite
redistribución con atribución), pero este script permite regenerarlo desde
la fuente oficial.

Uso:
    python -m src.data
"""
from __future__ import annotations

import urllib.request

from . import config as C


def descargar() -> None:
    C.DATA_RAW.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando desde {C.DATA_URL} ...")
    urllib.request.urlretrieve(C.DATA_URL, C.DATA_RAW)
    print(f"Guardado en {C.DATA_RAW}  ({C.DATA_RAW.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    descargar()
