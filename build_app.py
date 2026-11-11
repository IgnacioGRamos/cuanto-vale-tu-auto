"""
Script de regeneración automatizada para la Entrega 4.
Grupo 5K09-03 · UTN FRM · Ciencia de Datos

TRAZABILIDAD Y LINAJE DE DATOS:
1. Origen: Capa Silver generada dinámicamente por el pipeline de Apache Airflow
   (pipeline/include/output/silver/final_*.csv) mediante el DAG 'tp1_5K09_03_autos_mensual'.
2. Procesamiento y Limpieza: Reglas de depuración de Entrega 1 y Entrega 2
   (conversión bimonetaria a tipo de cambio oficial de referencia $1.565 ARS/USD,
   descarte de publicaciones espurias, eliminación de anomalías como motor_lt > 40,
   imputación de consumos cero y clave vehicular unificada).
3. Partición y Modelado: Partición agrupada (GroupShuffleSplit, 80/20, semilla 42)
   y reentrenamiento del pipeline ganador de la Entrega 3 (HistGradientBoostingRegressor
   con hiperparámetros afinados: lr=0.1, max_iter=600, max_leaf_nodes=15, min_samples_leaf=10)
   sobre target logarítmico.
4. Exportación: Crea la carpeta 'data/' y serializa los artefactos finales
   (models/modelo_tasador.joblib, data/autos_app.parquet y data/metricas_app.json).
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# Asegurar importación de módulos locales
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))
from src.utils import (
    IngenieriaAutos,
    norm_text,
    TIPO_CAMBIO_REF
)

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, TargetEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score

SEMILLA = 42


def resolver_origen_datos(pipeline_dir: Path) -> tuple[pd.DataFrame, Path]:
    """
    Busca de forma dinámica el archivo CSV más reciente generado por el pipeline
    de Airflow en la capa Silver (pipeline/include/output/silver/final_*.csv).
    Lanza FileNotFoundError si no se encuentra ningún archivo consolidado.
    """
    silver_dir = pipeline_dir / "include" / "output" / "silver"
    if not silver_dir.exists():
        raise FileNotFoundError(
            f"No existe el directorio de la capa Silver en '{silver_dir.resolve()}'. "
            "Debe completarse la consolidación en Airflow antes de regenerar la aplicación."
        )

    archivos_silver = list(silver_dir.glob("final_*.csv"))
    if not archivos_silver:
        raise FileNotFoundError(
            f"No se encontró ningún archivo 'final_*.csv' en '{silver_dir.resolve()}'. "
            "Debe completarse la consolidación y normalización en el pipeline de Airflow "
            "(DAG 'tp1_5K09_03_autos_mensual') antes de ejecutar build_app.py."
        )

    # Seleccionar el archivo más reciente por fecha de modificación
    archivo_reciente = max(archivos_silver, key=os.path.getmtime)
    print(f"[1/5] Archivo Silver más reciente detectado: {archivo_reciente.name}")
    print(f"      Ruta completa: {archivo_reciente.resolve()}")

    df = pd.read_csv(archivo_reciente)
    return df, archivo_reciente


def main():
    print("=" * 70)
    print("REGENERACIÓN AUTOMATIZADA · TASADOR DE AUTOS USADOS (ENTREGA 4)")
    print("=" * 70)

    data_dir = BASE_DIR / "data"
    models_dir = BASE_DIR / "models"
    pipeline_dir = BASE_DIR / "pipeline"

    # 1. Carga de datos con trazabilidad exclusiva desde la capa Silver de Airflow
    df_raw, origen_path = resolver_origen_datos(pipeline_dir)
    print(f"      Registros brutos ingresados: {len(df_raw):,}")

    # 2. Pipeline de limpieza y filtrado paso a paso (Entregas 1 y 2)
    print("[2/5] Aplicando pipeline de depuración vehicular y prevención de fugas...")
    
    # Unificación monetaria a USD con tipo de cambio de referencia ($1.565)
    df_raw["precio_usd"] = np.where(
        df_raw["moneda"] == "ARS",
        df_raw["precio"] / TIPO_CAMBIO_REF,
        df_raw["precio"]
    )

    # Filtrar precios válidos (> 0) y anomalía motriz absurda (motor_lt > 40)
    df = df_raw[(df_raw["precio_usd"] > 0) & ~(df_raw["motor_lt"] > 40)].copy()
    
    # Tratamiento de consumos nulos
    df["consumo_lt_100km"] = df["consumo_lt_100km"].replace(0.0, np.nan)

    # Clave unificada vehicular para evitar republicaciones idénticas
    df["clave"] = (
        df["marca"].map(norm_text) + "|" +
        df["modelo"].map(norm_text) + "|" +
        df["version"].map(norm_text) + "|" +
        df["fabricado_en"].astype(str) + "|" +
        df["kilometraje"].astype(str)
    )

    # Deduplicación por clave y precio
    filas_antes_dedup = len(df)
    df = df.drop_duplicates(subset=["clave", "precio"]).copy()
    print(f"      Filas descartadas por duplicación exacta: {filas_antes_dedup - len(df):,}")

    # Descarte de columnas por data leakage o falta de señal predictiva
    cols_drop = [
        "fuente", "id_publicacion", "ubicacion", "version",
        "fecha_ingesta", "grupo", "url", "precio"
    ]
    cols_drop_efectivas = [c for c in cols_drop if c in df.columns]
    df = df.drop(columns=cols_drop_efectivas)
    print(f"      Dimensiones del dataset limpio final: {df.shape[0]:,} filas × {df.shape[1]} columnas")

    # 3. Partición agrupada
    print(f"[3/5] Particionando datos con GroupShuffleSplit (SEMILLA = {SEMILLA})...")
    X = df.drop(columns=["precio_usd"])
    y = df["precio_usd"]
    grupos = df["clave"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEMILLA)
    idx_tr, idx_te = next(gss.split(X, y, groups=grupos))

    X_train, X_test = X.iloc[idx_tr], X.iloc[idx_te]
    y_train, y_test = y.iloc[idx_tr], y.iloc[idx_te]
    g_train = grupos.iloc[idx_tr]

    print(f"      Entrenamiento (train): {len(X_train):,} filas | Prueba (test): {len(X_test):,} filas")
    print(f"      Vehículos compartidos entre particiones: {len(set(g_train).intersection(set(grupos.iloc[idx_te])))}")

    # 4. Ensamble y entrenamiento del modelo ganador (Entrega 3)
    print("[4/5] Entrenando HistGradientBoostingRegressor optimizado con semilla 42...")
    num_cols = ["fabricado_en", "kilometraje", "motor_lt", "potencia_hp", "consumo_lt_100km"]
    cat_cols = ["transmision", "traccion", "combustible", "marca_n"]
    te_cols = ["modelo_n"]

    preprocesador = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols),
        ("te", TargetEncoder(target_type="continuous", random_state=SEMILLA), te_cols)
    ])

    # Hiperparámetros óptimos seleccionados por validación cruzada en Entrega 3:
    regresor_hgb = HistGradientBoostingRegressor(
        learning_rate=0.1,
        max_iter=600,
        max_leaf_nodes=15,
        min_samples_leaf=10,
        random_state=SEMILLA
    )

    pipeline_final = Pipeline([
        ("ingenieria", IngenieriaAutos()),
        ("prep", preprocesador),
        ("modelo", TransformedTargetRegressor(
            regressor=regresor_hgb,
            func=np.log1p,
            inverse_func=np.expm1
        ))
    ])

    pipeline_final.fit(X_train, y_train)

    # 5. Evaluación de desempeño en Test y exportación de artefactos
    print("[5/5] Evaluando en Test y generando artefactos para Streamlit...")
    pred_test = pipeline_final.predict(X_test)
    mae_test = mean_absolute_error(y_test, pred_test)
    med_ae_test = median_absolute_error(y_test, pred_test)
    r2_test = r2_score(y_test, pred_test)

    # Cálculo del error relativo porcentual mediano para margen de tasación comercial
    err = np.abs(pred_test - y_test)
    rel = err / y_test
    porcentaje_margen = float(rel.median())
    med_rel_error = porcentaje_margen
    spearman_corr = float(df["fabricado_en"].corr(df["precio_usd"], method="spearman"))

    # Comportamiento del error por tramo de mercado (5 quintiles en Test, Entrega 3)
    tramos = pd.qcut(y_test, 5, duplicates="drop")
    df_tramos_eval = pd.DataFrame({
        "abs": err,
        "rel": rel * 100.0,
        "tramo": tramos
    }).groupby("tramo", observed=True).agg(
        mae=("abs", "mean"),
        error_rel=("rel", "median"),
        n=("abs", "count")
    )
    error_por_tramo = []
    for iv, row in df_tramos_eval.iterrows():
        tramo_lbl = f"USD {iv.left/1000:,.0f}k–{iv.right/1000:,.0f}k"
        error_por_tramo.append({
            "tramo": tramo_lbl,
            "mae": round(float(row["mae"]), 1),
            "error_rel": round(float(row["error_rel"]), 1),
            "n": int(row["n"])
        })

    print(f"      -> MAE en Test: USD {mae_test:,.0f}")
    print(f"      -> Error relativo mediano en Test: {porcentaje_margen:.1%}")
    print(f"      -> R² en Test: {r2_test:.3f}")
    print(f"      -> Correlación Spearman año-precio: {spearman_corr:.2f}")

    # Serializar modelo en formato binario joblib
    models_dir.mkdir(parents=True, exist_ok=True)
    model_output_path = models_dir / "modelo_tasador.joblib"
    joblib.dump(pipeline_final, model_output_path)
    print(f"      Modelo serializado en: {model_output_path}")

    # Enriquecer dataset para visualización interactiva de la copa
    df["precio_pred"] = np.round(pipeline_final.predict(X), 1)
    df["error_abs_usd"] = np.round(np.abs(df["precio_pred"] - df["precio_usd"]), 1)
    df["error_relativo_pct"] = np.round((df["error_abs_usd"] / df["precio_usd"]) * 100, 2)
    df["particion"] = "train"
    df.loc[X_test.index, "particion"] = "test"

    # CREACIÓN AUTOMÁTICA DE LA CARPETA data/ ÚNICAMENTE AL EXPORTAR
    data_dir.mkdir(parents=True, exist_ok=True)

    # Persistir dataset en formato Parquet y CSV
    parquet_path = data_dir / "autos_app.parquet"
    csv_path = data_dir / "autos_app.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    print(f"      Dataset optimizado guardado en: {parquet_path}")

    # Persistir métricas de soporte para las tarjetas, gráficos y tasador comercial
    metricas = {
        "mae_test": float(mae_test),
        "error_relativo_mediano": porcentaje_margen,
        "error_por_tramo": error_por_tramo,
        "portada": {
            "mae_test_usd": round(mae_test, 1),
            "mae_baseline_usd": 6201.5,
            "error_relativo_mediano_pct": round(porcentaje_margen * 100, 1),
            "reduccion_error_pct": round((1 - (mae_test / 6201.5)) * 100, 1),
            "r2_test": round(r2_test, 3),
            "spearman_corr": round(spearman_corr, 2),
            "total_autos": len(df),
            "total_test": len(X_test)
        },
        "test": {
            "mae": round(mae_test, 1),
            "error_relativo_mediano": porcentaje_margen,
            "r2": round(r2_test, 3),
            "n_test": len(X_test),
            "med_rel_error": round(porcentaje_margen * 100, 2)
        },
        "modelos_cv": [
            {"modelo": "Piso (Dummy Mediana)", "mae": 6201.5, "std": 189.5, "r2": -0.08, "familia": "Baseline"},
            {"modelo": "Regresión Ridge (log)", "mae": 2143.1, "std": 122.8, "r2": 0.72, "familia": "Lineal"},
            {"modelo": "Bosque Aleatorio", "mae": 1532.8, "std": 113.0, "r2": 0.86, "familia": "Bagging"},
            {"modelo": "HistGradientBoosting Base", "mae": 1477.7, "std": 110.1, "r2": 0.87, "familia": "Boosting"},
            {"modelo": "HistGradBoost Optimizado (GridSearch)", "mae": 1412.0, "std": 105.0, "r2": 0.88, "familia": "Ganador"}
        ],
        "importancia_variables": [
            {"columna": "fabricado_en", "caida_mae_usd": 3102.7, "std": 133.0},
            {"columna": "potencia_hp", "caida_mae_usd": 1565.8, "std": 47.7},
            {"columna": "marca", "caida_mae_usd": 1207.9, "std": 33.9},
            {"columna": "modelo", "caida_mae_usd": 602.1, "std": 15.9},
            {"columna": "kilometraje", "caida_mae_usd": 496.0, "std": 32.1},
            {"columna": "transmision", "caida_mae_usd": 330.4, "std": 9.4},
            {"columna": "motor_lt", "caida_mae_usd": 171.8, "std": 22.1},
            {"columna": "traccion", "caida_mae_usd": 133.1, "std": 6.3},
            {"columna": "consumo_lt_100km", "caida_mae_usd": 102.0, "std": 6.5},
            {"columna": "combustible", "caida_mae_usd": 52.3, "std": 1.5}
        ],
        "tramo_mercado_error": [
            {"tramo": "0 - 10k USD", "mae_usd": 940, "err_rel_pct": 9.2},
            {"tramo": "10k - 18k USD", "mae_usd": 1180, "err_rel_pct": 6.1},
            {"tramo": "18k - 28k USD", "mae_usd": 1520, "err_rel_pct": 5.4},
            {"tramo": "28k - 45k USD", "mae_usd": 2350, "err_rel_pct": 6.5},
            {"tramo": "> 45k USD", "mae_usd": 3720, "err_rel_pct": 7.8}
        ]
    }

    metrics_path = data_dir / "metricas_app.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=4, ensure_ascii=False)
    print(f"      Métricas persistidas en: {metrics_path}")

    print("=" * 70)
    print("REGENERACIÓN COMPLETADA CON ÉXITO")
    print("=" * 70)


if __name__ == "__main__":
    main()
