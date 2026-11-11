"""
Utilidades y transformadores del proyecto de tasación de vehículos usados.
Grupo 5K09-03 · UTN FRM · Ciencia de Datos
"""

import re
import json
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.base import BaseEstimator, TransformerMixin

# Cotización de referencia utilizada a lo largo del proyecto (Entrega 3)
TIPO_CAMBIO_REF = 1565.0

# Reglas de normalización textual de variables categóricas (Entrega 3)
REGLAS_CAJA = [
    (r"manual", "manual"),
    (r"auto|cvt", "automatico")
]

REGLAS_TRAC = [
    (r"delan", "delantera"),
    (r"tras", "trasera"),
    (r"4x4|awd|integral", "4x4")
]

REGLAS_COMB = [
    (r"nafta", "nafta"),
    (r"diesel", "diesel"),
    (r"ctric", "electrico"),
    (r"brid|hev", "hibrido"),
    (r"gas", "gnc")
]


def norm_text(v):
    """Normaliza texto a minúsculas, sin acentos ni espacios redundantes."""
    if pd.isna(v) or v is None:
        return ""
    v = unicodedata.normalize("NFKD", str(v)).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", v.lower()).strip()


def normalizar_categorica(v, reglas):
    """Aplica regex secuencial para normalizar categorías con mojibake o variantes."""
    v_norm = norm_text(v)
    for patron, etiqueta in reglas:
        if re.search(patron, v_norm):
            return etiqueta
    return np.nan


class IngenieriaAutos(BaseEstimator, TransformerMixin):
    """
    Transformador de ingeniería de variables vehicular idéntico al de Entrega 3.
    Normaliza transmision, tracción y combustible, y genera marca_n y modelo_n
    para target encoding, eliminando identificadores y columnas redundantes.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_new = X.copy()
        if "transmision" in X_new.columns:
            X_new["transmision"] = X_new["transmision"].apply(
                lambda v: normalizar_categorica(v, REGLAS_CAJA)
            )
        if "traccion" in X_new.columns:
            X_new["traccion"] = X_new["traccion"].apply(
                lambda v: normalizar_categorica(v, REGLAS_TRAC)
            )
        if "combustible" in X_new.columns:
            X_new["combustible"] = X_new["combustible"].apply(
                lambda v: normalizar_categorica(v, REGLAS_COMB)
            )

        # Preparación para Target Encoding
        if "marca" in X_new.columns and "modelo" in X_new.columns:
            X_new["marca_n"] = X_new["marca"].apply(norm_text)
            X_new["modelo_n"] = X_new["marca_n"] + " " + X_new["modelo"].apply(norm_text)

        cols_a_descartar = [c for c in ["marca", "modelo", "moneda", "clave"] if c in X_new.columns]
        if cols_a_descartar:
            X_new = X_new.drop(columns=cols_a_descartar)

        return X_new


@st.cache_data
def load_data(data_path: str = "data/autos_app.parquet") -> pd.DataFrame:
    """Carga y cachea el dataset vehicular optimizado para la aplicación."""
    path = Path(data_path)
    if not path.exists():
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            return pd.read_csv(csv_path)
        raise FileNotFoundError(f"No se encontró el dataset en {data_path} ni {csv_path}")
    return pd.read_parquet(path)


@st.cache_data
def load_metrics(metrics_path: str = "data/metricas_app.json") -> dict:
    """Carga y cachea los resultados empíricos precalculados de las entregas."""
    path = Path(metrics_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_model(model_path: str = "models/modelo_tasador.joblib"):
    """Carga y cachea el pipeline entrenado del modelo final con semilla fija."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}. Ejecute build_app.py primero.")
    return joblib.load(path)


def format_usd(valor: float) -> str:
    """Formatea monto en dólares estadounidenses al estilo argentino (USD 10.735)."""
    return f"USD {valor:,.0f}".replace(",", ".")


def format_ars(valor_usd: float, tipo_cambio: float = TIPO_CAMBIO_REF) -> str:
    """Formatea monto equivalente en pesos argentinos según tipo de cambio."""
    valor_ars = valor_usd * tipo_cambio
    return f"$ {valor_ars:,.0f}".replace(",", ".")


def percentile_of(price: float, precios_serie: pd.Series) -> float:
    """Calcula el percentil de mercado (0.0 a 1.0) para un precio dado."""
    if len(precios_serie) == 0:
        return 0.5
    return float((precios_serie <= price).mean())


def warnings_for(marca: str, modelo: str, anio: int, km: int, motor: float, hp: float, consumo: float,
                 marcas_validas: set, modelos_validos: set) -> list[str]:
    """Genera advertencias técnicas idénticas a las del reporte web original."""
    out = []
    checks = [
        ("año", anio, 1937, 2026),
        ("kilometraje", km, 0, 8100000),
        ("motor", motor, 0.0, 6.4),
        ("potencia", hp, 0.0, 575.0),
        ("consumo", consumo, 1.4, 541.0),
    ]
    for name, v, lo, hi in checks:
        if v is None or np.isnan(v) or v < lo or v > hi:
            lo_str = f"{lo:,.1f}" if isinstance(lo, float) else f"{lo:,}"
            hi_str = f"{hi:,.1f}" if isinstance(hi, float) else f"{hi:,}"
            lo_str = lo_str.replace(",", ".")
            hi_str = hi_str.replace(",", ".")
            out.append(f"El {name} está fuera del rango visto en entrenamiento ({lo_str}–{hi_str}).")

    if norm_text(marca) not in marcas_validas:
        out.append("La marca no aparecía en el entrenamiento; el modelo la trata como una marca desconocida.")
    
    mod_key = f"{norm_text(marca)} {norm_text(modelo)}"
    if mod_key not in modelos_validos:
        out.append("Este modelo no aparecía en el entrenamiento; se usa el promedio del mercado para ese componente.")

    return out


def comparables_for(marca: str, modelo: str, anio: int, km: int, df: pd.DataFrame, n: int = 4) -> list[dict]:
    """
    Encuentra los vehículos más cercanos en el dataset limpio utilizando la métrica de distancia exacta del original:
    d = |año - anio| * 8 + |km - kilometraje| / 10000 + (0 si mismo_modelo sino 4)
    """
    m_marca = df["marca"].astype(str).apply(norm_text) == norm_text(marca)
    m_modelo = df["modelo"].astype(str).apply(norm_text) == norm_text(modelo)
    
    same_model = df[m_marca & m_modelo]
    pool = same_model if len(same_model) >= 3 else df[m_marca]
    if len(pool) == 0:
        pool = df

    # Vectorizar distancia
    diff_anio = (pool["fabricado_en"] - anio).abs() * 8.0
    diff_km = (pool["kilometraje"] - km).abs() / 10000.0
    diff_mod = np.where(pool["modelo"].astype(str).apply(norm_text) == norm_text(modelo), 0.0, 4.0)
    scores = diff_anio + diff_km + diff_mod

    top_indices = scores.sort_values().head(n).index
    comparables = []
    for idx in top_indices:
        r = pool.loc[idx]
        comparables.append({
            "marca": r["marca"],
            "modelo": r["modelo"],
            "anio": int(r["fabricado_en"]),
            "km": int(r["kilometraje"]),
            "precio_real": float(r["precio_usd"]),
            "precio_pred": float(r.get("precio_pred", r["precio_usd"]))
        })
    return comparables
