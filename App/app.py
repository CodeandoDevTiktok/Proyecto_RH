"""
Front Flask para la API p09-reingreso-30d: lista de verificación de alta.

Una sola pantalla: el equipo de alta registra los datos del episodio y obtiene
el riesgo de reingreso < 30 días con el seguimiento sugerido.

Instalación:
    pip install flask requests

Ejecución (con la API ya levantada en el puerto 8000):
    python app.py                          # abrir http://localhost:5000
    API_URL=http://otra-maquina:8000 python app.py

AVISO: la ficha del modelo indica veredicto "NO CUMPLE". Prototipo académico.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests
from flask import Flask, flash, render_template, request

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = float(os.environ.get("API_TIMEOUT", 30))
PREVALENCIA = 0.1136  # test_prevalencia de la ficha; se actualiza si la API responde

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-reingreso-p09")

# ------------------------------------------------------------------ catálogos (valor del modelo, etiqueta en español)
# El valor (izquierda) es lo que se envía a la API y NO debe traducirse.
# Ajusta estas listas si tu preprocesamiento usa otras categorías.
MEDICAMENTOS = [
    ("metformin", "Metformina"),
    ("repaglinide", "Repaglinida"),
    ("nateglinide", "Nateglinida"),
    ("chlorpropamide", "Clorpropamida"),
    ("glimepiride", "Glimepirida"),
    ("acetohexamide", "Acetohexamida"),
    ("glipizide", "Glipizida"),
    ("glyburide", "Gliburida"),
    ("tolbutamide", "Tolbutamida"),
    ("pioglitazone", "Pioglitazona"),
    ("rosiglitazone", "Rosiglitazona"),
    ("acarbose", "Acarbosa"),
    ("miglitol", "Miglitol"),
    ("troglitazone", "Troglitazona"),
    ("tolazamide", "Tolazamida"),
    ("insulin", "Insulina"),
    ("glyburide-metformin", "Gliburida + metformina"),
    ("glipizide-metformin", "Glipizida + metformina"),
    ("glimepiride-pioglitazone", "Glimepirida + pioglitazona"),
    ("metformin-rosiglitazone", "Metformina + rosiglitazona"),
    ("metformin-pioglitazone", "Metformina + pioglitazona"),
]
OPC_MED = [("No", "No recibe"), ("Steady", "Sin cambio de dosis"),
           ("Up", "Dosis aumentada"), ("Down", "Dosis reducida")]

GRUPOS_DIAG = [
    ("circulatorio", "Circulatorio"), ("respiratorio", "Respiratorio"),
    ("digestivo", "Digestivo"), ("diabetes", "Diabetes"), ("lesiones", "Lesiones"),
    ("musculoesqueletico", "Musculoesquelético"), ("genitourinario", "Genitourinario"),
    ("neoplasias", "Neoplasias"), ("infecciosas", "Infecciosas"), ("otros", "Otros"),
]

OPCIONES = {
    "gender": [("Female", "Femenino"), ("Male", "Masculino")],
    "payer_code": [
        ("Desconocido", "Desconocido"), ("MC", "Medicare"), ("MD", "Medicaid"),
        ("HM", "HMO (plan gestionado)"), ("SP", "Pago propio"), ("BC", "Blue Cross / Blue Shield"),
        ("UN", "UnitedHealthcare"), ("CP", "Otra aseguradora (CP)"), ("CM", "Otra aseguradora (CM)"),
        ("DM", "Otra aseguradora (DM)"), ("OG", "Otro programa público"), ("PO", "Otra aseguradora (PO)"),
        ("SI", "Autoasegurado"), ("WC", "Seguro de accidentes de trabajo"), ("CH", "Seguro militar (CH)"),
        ("OT", "Otro"), ("MP", "Otra aseguradora (MP)"), ("FR", "Otra aseguradora (FR)"),
    ],
    "max_glu_serum": [("No medido", "No medido"), ("Norm", "Normal"),
                      (">200", "Mayor a 200 mg/dL"), (">300", "Mayor a 300 mg/dL")],
    "A1Cresult": [("No medido", "No medido"), ("Norm", "Normal"),
                  (">7", "Mayor a 7 %"), (">8", "Mayor a 8 %")],
    "change": [("No", "No"), ("Ch", "Sí")],
    "diabetesMed": [("No", "No"), ("Yes", "Sí")],
    "diag_1_grupo": GRUPOS_DIAG,
    "diag_2_grupo": GRUPOS_DIAG,
    "diag_3_grupo": GRUPOS_DIAG,
    "especialidad_grupo": [
        ("Desconocido", "Desconocida"), ("InternalMedicine", "Medicina interna"),
        ("Family/GeneralPractice", "Medicina familiar / general"), ("Cardiology", "Cardiología"),
        ("Surgery", "Cirugía"), ("Emergency/Trauma", "Emergencias / trauma"), ("Otros", "Otras"),
    ],
    "admission_type_grupo": [("emergency", "Emergencia"), ("urgent", "Urgente"),
                             ("elective", "Programada (electiva)"), ("otros", "Otro")],
    "admission_source_grupo": [("emergencia", "Servicio de emergencia"),
                               ("traslado", "Traslado de otro centro"),
                               ("referencia", "Referencia médica"), ("otros", "Otro")],
}

NUMERICOS = [
    ("time_in_hospital", "Días de estancia", 1, 14),
    ("num_lab_procedures", "N.º de exámenes de laboratorio", 0, 132),
    ("num_procedures", "N.º de procedimientos", 0, 6),
    ("num_medications", "N.º de medicamentos", 0, 81),
    ("number_outpatient", "Consultas ambulatorias (último año)", 0, 42),
    ("number_emergency", "Visitas a emergencia (último año)", 0, 76),
    ("number_inpatient", "Hospitalizaciones (último año)", 0, 21),
    ("number_diagnoses", "N.º de diagnósticos", 1, 16),
    ("edad_num", "Edad (años)", 5, 95),
]


# ------------------------------------------------------------------ cliente de la API
class ErrorAPI(Exception):
    pass


def llamar(metodo: str, ruta: str, **kw) -> Any:
    try:
        r = requests.request(metodo, f"{API_URL}{ruta}", timeout=TIMEOUT, **kw)
    except requests.RequestException as e:
        raise ErrorAPI(f"No se pudo conectar con la API en {API_URL} ({type(e).__name__}).")
    if r.status_code >= 400:
        try:
            det = r.json().get("detail", r.text)
        except ValueError:
            det = r.text
        if isinstance(det, (dict, list)):
            det = json.dumps(det, ensure_ascii=False)
        raise ErrorAPI(f"La API respondió con el código {r.status_code}: {det}")
    return r.json()


def prevalencia() -> float:
    try:
        return float(llamar("GET", "/modelo")["metricas_test"]["test_prevalencia"])
    except (ErrorAPI, TypeError, KeyError, ValueError):
        return PREVALENCIA


# ------------------------------------------------------------------ reglas del front
def nivel_riesgo(p: float, prev: float) -> Dict[str, str]:
    """Nivel orientativo relativo a la prevalencia (no es un umbral validado clínicamente)."""
    if p >= 2 * prev:
        return {"nivel": "Alto", "clase": "alto"}
    if p >= prev:
        return {"nivel": "Medio", "clase": "medio"}
    return {"nivel": "Bajo", "clase": "bajo"}


def plan_seguimiento(clase: str) -> str:
    if clase == "alto":
        return "Llamada a las 48 h y cita a los 7 días"
    if clase == "medio":
        return "Llamada a las 48 h"
    return "Seguimiento habitual"


def derivar(d: Dict[str, Any]) -> Dict[str, Any]:
    """Calcula las variables derivadas a partir de los campos del formulario."""
    meds = [d.get(m, "No") for m, _ in MEDICAMENTOS]
    amb, emer, hosp = (float(d.get(k) or 0) for k in
                       ("number_outpatient", "number_emergency", "number_inpatient"))
    d["visitas_previas_total"] = amb + emer + hosp
    d["hosp_previas_4mas"] = 1.0 if hosp >= 4 else 0.0
    d["n_meds_activos"] = float(sum(m != "No" for m in meds))
    d["n_meds_cambiados"] = float(sum(m in ("Up", "Down") for m in meds))
    d["a1c_medido"] = 0.0 if d.get("A1Cresult") == "No medido" else 1.0
    d["glu_medido"] = 0.0 if d.get("max_glu_serum") == "No medido" else 1.0
    d["diag_principal_diabetes"] = 1.0 if d.get("diag_1_grupo") == "diabetes" else 0.0
    return d


def senales(v: Dict[str, Any]) -> List[str]:
    """Rasgos descriptivos del caso (no son explicaciones del modelo)."""
    f = []
    hosp = float(v.get("number_inpatient") or 0)
    emer = float(v.get("number_emergency") or 0)
    if hosp >= 1:
        f.append(f"{int(hosp)} hospitalización(es) en el último año")
    if emer >= 1:
        f.append(f"{int(emer)} visita(s) a emergencia en el último año")
    if float(v.get("time_in_hospital") or 0) >= 7:
        f.append("Estancia prolongada (7 días o más)")
    if float(v.get("number_diagnoses") or 0) >= 9:
        f.append("Muchos diagnósticos registrados (9 o más)")
    if float(v.get("n_meds_cambiados") or 0) >= 1:
        f.append("Cambio de dosis de antidiabéticos durante la estancia")
    if v.get("insulin") in ("Up", "Down"):
        f.append("Ajuste de insulina")
    if v.get("A1Cresult") in (">7", ">8"):
        f.append("Hemoglobina glicosilada (HbA1c) elevada")
    if v.get("A1Cresult") == "No medido":
        f.append("Hemoglobina glicosilada (HbA1c) no medida en la estancia")
    return f


# ------------------------------------------------------------------ vista única
@app.context_processor
def globales():
    return {"API_URL": API_URL}


@app.route("/", methods=["GET", "POST"])
def inicio():
    valores: Dict[str, Any] = {}
    error_api: Optional[str] = None
    try:
        ej = llamar("GET", "/ejemplo")
        valores = dict(zip(ej["columns"], ej["data"][0]))
    except ErrorAPI as e:
        error_api = str(e)

    resultado = None
    if request.method == "POST":
        valores = dict(request.form.items())
        for c, *_ in NUMERICOS:
            valores[c] = valores.get(c) or 0
        valores = derivar(valores)
        try:
            columnas = [c["nombre"] for c in llamar("GET", "/esquema")["columnas"]]
            pid = valores.get("id_paciente") or "paciente"
            datos = {"columns": columnas, "data": [[valores.get(c) for c in columnas]], "ids": [pid]}
            res = llamar("POST", "/predecir", json=datos)
            p = res["resultados"][0]["probabilidad_reingreso_30d"]
            prev = prevalencia()
            nv = nivel_riesgo(p, prev)
            resultado = {
                "id": pid, "p": p, "prev": prev,
                "veces": p / prev if prev else None,
                "plan": plan_seguimiento(nv["clase"]),
                "senales": senales(valores),
                **nv,
            }
            error_api = None
        except ErrorAPI as e:
            flash(str(e), "error")

    return render_template(
        "paciente.html", v=valores, resultado=resultado, error_api=error_api,
        numericos=NUMERICOS, medicamentos=MEDICAMENTOS, opc_med=OPC_MED, opciones=OPCIONES,
    )


@app.template_filter("pct")
def pct(x, dec=1):
    try:
        return f"{float(x) * 100:.{dec}f} %".replace(".", ",")
    except (TypeError, ValueError):
        return "—"


@app.template_filter("num")
def num(x, dec=1):
    try:
        return f"{float(x):.{dec}f}".replace(".", ",")
    except (TypeError, ValueError):
        return "—"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("FLASK_PORT", 5000)), debug=True)
