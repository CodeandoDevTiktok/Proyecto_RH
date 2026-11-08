"""
API REST del modelo p09-reingreso-30d (probabilidad de reingreso hospitalario < 30 días).

Archivos esperados en la misma carpeta (o rutas por variables de entorno):
    modelo_campeon.joblib   MODELO_PATH   pipeline scikit-learn (CalibratedClassifierCV + HistGradientBoosting)
    ficha_modelo.json       FICHA_PATH    metadatos: columnas, métricas, veredicto
    ejemplo_entrada.json    EJEMPLO_PATH  (opcional) ejemplo en formato {"columns": [...], "data": [[...]]}

Instalación (la versión de scikit-learn DEBE ser la del entrenamiento):
    pip install "scikit-learn==1.6.1" joblib pandas numpy fastapi "uvicorn[standard]"

Ejecución:
    uvicorn api_reingreso:app --host 0.0.0.0 --port 8000        # o: python api_reingreso.py
Documentación interactiva (probar desde el navegador):
    http://localhost:8000/docs

Pruebas rápidas:
    curl http://localhost:8000/salud
    curl http://localhost:8000/probar-ejemplo
    curl -X POST http://localhost:8000/predecir -H "Content-Type: application/json" -d @ejemplo_entrada.json
    curl -X POST "http://localhost:8000/priorizar?k_pct=0.10" -H "Content-Type: application/json" -d @ejemplo_entrada.json

AVISO: la ficha del modelo indica veredicto "NO CUMPLE". Es un prototipo académico:
no debe usarse para decidir sobre pacientes reales.
"""

from __future__ import annotations

import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

BASE = Path(__file__).resolve().parent.parent / "Modelo"
MODELO_PATH = Path(os.environ.get("MODELO_PATH", BASE / "modelo_campeon.joblib"))
FICHA_PATH = Path(os.environ.get("FICHA_PATH", BASE / "ficha_modelo.json"))
EJEMPLO_PATH = Path(os.environ.get("EJEMPLO_PATH", BASE / "ejemplo_entrada.json"))
SKLEARN_ENTRENAMIENTO = "1.6.1"
K_PCT_DEFECTO = 0.10  # la ficha indica contactar el top 10 % de las altas

# ----------------------------------------------------------------------------- carga
if sklearn.__version__ != SKLEARN_ENTRENAMIENTO:
    warnings.warn(
        f"scikit-learn {sklearn.__version__} instalado; el modelo se entrenó con {SKLEARN_ENTRENAMIENTO}. "
        f"Instala scikit-learn=={SKLEARN_ENTRENAMIENTO} para evitar errores o predicciones distintas."
    )
try:
    MODELO = joblib.load(MODELO_PATH)
except AttributeError as e:
    raise RuntimeError(
        f"No se pudo cargar {MODELO_PATH.name} ({e}). Casi siempre es una versión distinta de "
        f"scikit-learn: instala scikit-learn=={SKLEARN_ENTRENAMIENTO}."
    ) from e

FICHA: Dict[str, Any] = (
    json.loads(FICHA_PATH.read_text(encoding="utf-8")) if FICHA_PATH.exists() else {}
)
EJEMPLO: Optional[Dict[str, Any]] = (
    json.loads(EJEMPLO_PATH.read_text(encoding="utf-8"))
    if EJEMPLO_PATH.exists()
    else None
)

if FICHA.get("columnas_entrada"):
    COLUMNAS = [c["name"] for c in FICHA["columnas_entrada"]]
    TIPOS = {c["name"]: c["type"] for c in FICHA["columnas_entrada"]}
else:  # sin ficha: se usan las columnas del propio modelo
    COLUMNAS = list(getattr(MODELO, "feature_names_in_", []))
    TIPOS = {c: "string" for c in COLUMNAS}
NUMERICAS = [
    c for c in COLUMNAS if TIPOS.get(c) in ("double", "float", "integer", "long")
]
CATEGORICAS = [c for c in COLUMNAS if c not in NUMERICAS]
IDX_POSITIVA = (
    int(np.where(np.asarray(MODELO.classes_) == 1)[0][0])
    if 1 in list(MODELO.classes_)
    else -1
)
ADVERTENCIA = FICHA.get("uso", "Prototipo académico.")
INICIO = time.time()


# ----------------------------------------------------------------------------- esquemas
class EntradaSplit(BaseModel):
    """Formato 'split' de pandas (el mismo de ejemplo_entrada.json)."""

    columns: List[str] = Field(
        ..., description="Nombres de las columnas, en cualquier orden"
    )
    data: List[List[Any]] = Field(
        ..., description="Filas: una lista de valores por paciente/alta"
    )
    ids: Optional[List[str]] = Field(
        None, description="Identificadores opcionales de cada fila"
    )

    model_config = {
        "json_schema_extra": {"example": EJEMPLO or {"columns": COLUMNAS, "data": []}}
    }


class EntradaRegistros(BaseModel):
    """Lista de registros: [{columna: valor, ...}, ...]."""

    registros: List[Dict[str, Any]]
    ids: Optional[List[str]] = None


# ----------------------------------------------------------------------------- utilidades
def preparar(df: pd.DataFrame) -> pd.DataFrame:
    faltan = [c for c in COLUMNAS if c not in df.columns]
    if faltan:
        raise HTTPException(
            422, {"error": "Faltan columnas obligatorias", "columnas_faltantes": faltan}
        )
    X = df[COLUMNAS].copy()  # orden del entrenamiento; columnas extra se ignoran
    for c in NUMERICAS:
        conv = pd.to_numeric(X[c], errors="coerce")
        malos = X[c].notna() & conv.isna()
        if malos.any():
            raise HTTPException(
                422,
                {
                    "error": f"La columna '{c}' debe ser numérica",
                    "filas": [int(i) for i in np.where(malos)[0][:20]],
                },
            )
        X[c] = conv.astype(float)
    for c in CATEGORICAS:
        X[c] = X[c].where(X[c].isna(), X[c].astype(str))
    return X


def puntuar(
    df: pd.DataFrame, ids: Optional[List[str]], k_pct: Optional[float]
) -> Dict[str, Any]:
    if len(df) == 0:
        raise HTTPException(422, "No se enviaron filas.")
    X = preparar(df)
    try:
        proba = MODELO.predict_proba(X)[:, IDX_POSITIVA]
    except Exception as e:
        raise HTTPException(
            422, f"El modelo no pudo procesar la entrada: {type(e).__name__}: {e}"
        )
    ids = ids if ids and len(ids) == len(df) else [str(i) for i in range(len(df))]
    orden = np.argsort(-proba)
    rank = np.empty(len(proba), int)
    rank[orden] = np.arange(1, len(proba) + 1)
    res = [
        {
            "id": ids[i],
            "probabilidad_reingreso_30d": round(float(proba[i]), 6),
            "ranking": int(rank[i]),
        }
        for i in range(len(proba))
    ]
    salida = {
        "modelo": FICHA.get("modelo"),
        "version": FICHA.get("version"),
        "n": len(res),
        "advertencia": ADVERTENCIA,
    }
    if k_pct is not None:
        k = max(1, math.ceil(k_pct * len(res)))
        for r in res:
            r["contactar"] = r["ranking"] <= k
        res.sort(key=lambda r: r["ranking"])
        salida.update(
            {
                "k_pct": k_pct,
                "k": k,
                "criterio": f"contactar las {k} altas con mayor probabilidad ({k_pct:.0%} del lote)",
            }
        )
    salida["resultados"] = res
    return salida


# ----------------------------------------------------------------------------- API
app = FastAPI(
    title="API de reingreso hospitalario < 30 días (p09)",
    description=f"Modelo {FICHA.get('modelo', '')} · {FICHA.get('algoritmo', '')} calibrado "
    f"({FICHA.get('calibracion', '')}). **{ADVERTENCIA}**",
    version=str(FICHA.get("version", "1")),
)


@app.get("/", tags=["general"])
def inicio():
    return {
        "mensaje": "API de reingreso < 30 días",
        "documentacion": "/docs",
        "endpoints": [
            "/salud",
            "/modelo",
            "/esquema",
            "/ejemplo",
            "/probar-ejemplo",
            "/predecir",
            "/predecir/registros",
            "/priorizar",
        ],
        "advertencia": ADVERTENCIA,
    }


@app.get("/salud", tags=["general"])
def salud():
    return {
        "estado": "ok",
        "modelo_cargado": True,
        "tipo": type(MODELO).__name__,
        "sklearn": sklearn.__version__,
        "sklearn_entrenamiento": SKLEARN_ENTRENAMIENTO,
        "version_correcta": sklearn.__version__ == SKLEARN_ENTRENAMIENTO,
        "n_columnas": len(COLUMNAS),
        "segundos_activo": round(time.time() - INICIO, 1),
    }


@app.get("/modelo", tags=["general"])
def ficha():
    claves = [
        "modelo",
        "alias",
        "version",
        "run_id",
        "mlflow",
        "algoritmo",
        "calibracion",
        "salida",
        "metricas_test",
        "etiquetas_version",
        "veredicto",
        "uso",
    ]
    return {k: FICHA.get(k) for k in claves if k in FICHA}


@app.get("/esquema", tags=["general"])
def esquema():
    return {
        "columnas": [
            {"nombre": c, "tipo": TIPOS.get(c), "obligatoria": True} for c in COLUMNAS
        ],
        "numericas": NUMERICAS,
        "categoricas": CATEGORICAS,
        "formato_entrada": {"columns": ["..."], "data": [["..."]], "ids": "opcional"},
    }


@app.get("/ejemplo", tags=["pruebas"])
def ejemplo():
    if EJEMPLO is None:
        raise HTTPException(404, "No se encontró ejemplo_entrada.json")
    return EJEMPLO


@app.get("/probar-ejemplo", tags=["pruebas"])
def probar_ejemplo():
    if EJEMPLO is None:
        raise HTTPException(404, "No se encontró ejemplo_entrada.json")
    return puntuar(
        pd.DataFrame(EJEMPLO["data"], columns=EJEMPLO["columns"]), None, K_PCT_DEFECTO
    )


@app.post("/predecir", tags=["predicción"])
def predecir(entrada: EntradaSplit):
    """Probabilidad de reingreso < 30 días por fila (formato split, como ejemplo_entrada.json)."""
    return puntuar(
        pd.DataFrame(entrada.data, columns=entrada.columns), entrada.ids, None
    )


@app.post("/predecir/registros", tags=["predicción"])
def predecir_registros(entrada: EntradaRegistros):
    """Igual que /predecir, pero con una lista de diccionarios {columna: valor}."""
    return puntuar(pd.DataFrame(entrada.registros), entrada.ids, None)


@app.post("/priorizar", tags=["predicción"])
def priorizar(
    entrada: EntradaSplit,
    k_pct: float = Query(
        K_PCT_DEFECTO, gt=0, le=1, description="Fracción del lote a contactar"
    ),
):
    """Ordena las altas de mayor a menor riesgo y marca el top K (por defecto el 10 %)."""
    return puntuar(
        pd.DataFrame(entrada.data, columns=entrada.columns), entrada.ids, k_pct
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
