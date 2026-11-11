"""
¿Cuánto vale tu auto? · Tasador Inteligente
Aplicación interactiva y reporte analítico (Entrega 4).
Estructura narrativa: Copa de Martini.
Grupo 5K09-03 · UTN FRM · Ciencia de Datos · 2026
"""

from pathlib import Path
import random
import numpy as np
import pandas as pd
import streamlit as st

# Importaciones modulares internas del repositorio
from src.utils import (
    load_data,
    load_metrics,
    load_model,
    format_usd,
    format_ars,
    percentile_of,
    warnings_for,
    comparables_for,
    norm_text,
    TIPO_CAMBIO_REF,
)
from src.charts import (
    chart_histograma_precios,
    chart_impacto_columnas,
    chart_test_real_vs_pred,
    chart_error_por_tramo,
    chart_explora_scatter,
    chart_ubicacion_h4,
)

# -----------------------------------------------------------------------------
# 1. Configuración general de la página Streamlit (Modo ancho)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="¿Cuánto vale tu auto? · Tasador Inteligente",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 2. Inyección de estilos CSS y calibración de márgenes del contenedor
# -----------------------------------------------------------------------------
css_path = Path(__file__).parent / "src" / "style.css"
if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
    /* Ajuste del contenedor principal de la app */
    .main .block-container, [data-testid="stMainBlockContainer"], .block-container {
        max-width: 1200px !important;
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        margin: 0 auto !important;
    }

    /* Asegurar que las tarjetas de métricas y llamadas se expandan parejo */
    div[data-testid="stHorizontalBlock"] {
        gap: 1.5rem !important;
    }

    /* Barra lateral permanentemente fija e incolapsable */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    div[data-testid="stSidebarHeader"] button,
    section[data-testid="stSidebar"] button[kind="header"],
    button[aria-label="Close sidebar"],
    button[aria-label="Collapse sidebar"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }

    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"][aria-expanded="false"] {
        display: flex !important;
        transform: none !important;
        visibility: visible !important;
        min-width: 250px !important;
        max-width: 280px !important;
    }

    nav.nav-sidebar .equipo-lista {
        list-style-type: disc !important;
        margin: 0 0 20px 0 !important;
        padding: 0 0 0 16px !important;
    }
    nav.nav-sidebar .equipo-lista li {
        font-size: 12px !important;
        line-height: 1.5 !important;
        color: var(--gris, #5a6578) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Carga de datos, métricas y modelo preentrenado
# -----------------------------------------------------------------------------
data_path = Path(__file__).parent / "data" / "autos_app.parquet"
metrics_path = Path(__file__).parent / "data" / "metricas_app.json"
model_path = Path(__file__).parent / "models" / "modelo_tasador.joblib"

df_autos = load_data(str(data_path))
metricas = load_metrics(str(metrics_path))
modelo = load_model(str(model_path))

# Conjunto de prueba y marcas / modelos válidos
df_test = df_autos[df_autos["particion"] == "test"].copy()
marcas_validas = set(df_autos["marca"].dropna().apply(norm_text).unique())
modelos_validos = set(
    (df_autos["marca"].dropna().apply(norm_text) + " " + df_autos["modelo"].dropna().apply(norm_text)).unique()
)

# Métricas consolidadas dinámicas
portada_meta = metricas.get("portada", {})
test_meta = metricas.get("test", {})

mae_test_val = float(metricas.get("mae_test", test_meta.get("mae", portada_meta.get("mae_test_usd", 1435.0))))
r2_test_val = float(test_meta.get("r2", portada_meta.get("r2_test", 0.89)))
total_autos_val = len(df_autos) if len(df_autos) > 0 else int(portada_meta.get("total_autos", 8909))
total_test_val = len(df_test) if len(df_test) > 0 else int(test_meta.get("n_test", 1782))
spearman_corr_val = float(portada_meta.get("spearman_corr", 0.76))

# Error relativo porcentual mediano para tasación comercial (default 6.6%)
porcentaje_margen = float(
    metricas.get(
        "error_relativo_mediano",
        test_meta.get(
            "error_relativo_mediano",
            portada_meta.get("error_relativo_mediano_pct", 6.6) / 100.0 if "error_relativo_mediano_pct" in portada_meta else 0.066
        )
    )
)
str_margen_pct = f"{porcentaje_margen:.1%}"

mediana_precio_val = float(df_autos["precio_usd"].median()) if len(df_autos) > 0 else 10735.0
promedio_precio_val = float(df_autos["precio_usd"].mean()) if len(df_autos) > 0 else 13902.0

str_mae_test = format_usd(mae_test_val)
str_total_autos = f"{total_autos_val:,}".replace(",", ".")
str_total_test = f"{total_test_val:,}".replace(",", ".")
str_spearman = f"{spearman_corr_val:.2f}".replace(".", ",")
str_mediana = format_usd(mediana_precio_val)
str_promedio = format_usd(promedio_precio_val)

# -----------------------------------------------------------------------------
# Barra lateral: Navegación de la Copa de Martini
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <nav class="nav-sidebar">
          <p class="marca">Ciencia de Datos · UTN FRM · 2026<br>Proyecto Integrador · 5K09-03</p>
          <ul class="equipo-lista">
            <li>Garcia, Nicolás</li>
            <li>Ramos, Ignacio</li>
            <li>Velasco, Victoria</li>
            <li>Via, Tomás</li>
            <li>Villanueva, Sergio</li>
          </ul>
          <div class="tallo">
            <h3>El tallo · la historia</h3>
            <ol>
              <li><a href="#portada" target="_self" class="activo">El proyecto</a></li>
              <li><a href="#datos" target="_self">Los datos</a></li>
              <li><a href="#hallazgos" target="_self">Lo que encontramos</a></li>
              <li><a href="#modelo" target="_self">El modelo</a></li>
            </ol>
          </div>
          <div class="copa">
            <h3>La copa · tu turno</h3>
            <ol>
              <li><a href="#explora" target="_self">Explorá los autos</a></li>
              <li><a href="#proba" target="_self">Probá el modelo</a></li>
            </ol>
            <p class="copa-nota">El tallo se lee en orden. La copa la manejás vos.</p>
          </div>
          <svg class="glass" viewBox="0 0 120 170" role="img" aria-label="Copa de martini: tallo y copa">
            <defs>
              <linearGradient id="glassFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#2a78d6" stop-opacity=".85"></stop>
                <stop offset="100%" stop-color="#0b5d7a" stop-opacity=".45"></stop>
              </linearGradient>
            </defs>
            <path d="M18 16 L60 94 L102 16 Z" fill="url(#glassFill)" stroke="#0b5d7a" stroke-width="2"></path>
            <path d="M18 16 Q60 8 102 16" fill="none" stroke="#0b5d7a" stroke-width="2"></path>
            <line x1="60" y1="94" x2="60" y2="150" stroke="#b45309" stroke-width="3"></line>
            <line x1="36" y1="154" x2="84" y2="154" stroke="#b45309" stroke-width="3" stroke-linecap="round"></line>
            <circle cx="82" cy="46" r="5" fill="#b45309"></circle>
            <line x1="82" y1="46" x2="91" y2="22" stroke="#b45309" stroke-width="2"></line>
          </svg>
        </nav>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# SECCIÓN 0: El proyecto (Portada)
# =============================================================================
st.markdown(
    f"""
    <div id="portada"></div>
    <section class="seccion" style="padding-bottom: 0;">
      <p class="etapa tallo">El proyecto</p>
      <h1>¿Se puede predecir el precio de un vehículo a partir de sus variables?</h1>
      <p class="lead">Un usado tiene año, kilómetros, motor, potencia, marca y modelo. Si el precio depende de esas cosas, un modelo debería poder tasarlo sin ver la publicación.</p>
      <p class="oracion">El precio de un usado se puede estimar con un error de <strong>{str_mae_test}</strong> usando año, potencia, marca y modelo; el año es la variable que más pesa y los autos de alta gama son los que más se le escapan.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

# 3. Tarjetas de KPIs distribuidas en columnas proporcionales equilibradas
kpi_col1, kpi_col2, kpi_col3 = st.columns(3)
with kpi_col1:
    st.markdown(
        f"""
        <div class="num num-mae">
          <span>{str_mae_test}</span>
          <p>MAE en el test: el error absoluto medio de la tasación, sobre {str_total_test} autos que el modelo no vio.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi_col2:
    st.markdown(
        f"""
        <div class="num num-autos">
          <span>{str_total_autos}</span>
          <p>autos usados en el dataset limpio, con una fila por aviso y una clave por vehículo.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with kpi_col3:
    st.markdown(
        f"""
        <div class="num num-corr">
          <span>{str_spearman}</span>
          <p>correlación de Spearman entre año de fabricación y precio: alta, pero lejos de ser una fuga.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <section class="seccion" style="padding-top: 14px; border-bottom: 1px solid var(--linea);">
      <p class="nota">Medido sobre el dataset de la cátedra (pipeline de avisos de autos usados, 2026). El modelo es un <code>HistGradientBoosting</code> con transformación logarítmica del precio; los detalles técnicos están en cada sección.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# SECCIÓN 1: Los datos
# =============================================================================
st.markdown(
    f"""
    <div id="datos"></div>
    <section class="seccion">
      <p class="etapa tallo">Los datos</p>
      <h2>El mercado de usados se concentra entre USD 8.000 y USD 17.000, pero la cola alta llega a USD 160.000.</h2>
      <p>La distribución del precio es asimétrica: la mediana está en <strong>{str_mediana}</strong>, mientras que el promedio sube a <strong>{str_promedio}</strong> por los autos de alta gama. Esa cola es la que después explica los errores más caros del modelo.</p>
    """,
    unsafe_allow_html=True,
)

# Gráfico Altair g-hist con ancho responsivo
chart_hist = chart_histograma_precios(df_autos)
st.altair_chart(chart_hist, use_container_width=True)

st.markdown(
    f"""
      <p class="nota">Cada barra cuenta autos en un tramo de precio; la línea naranja marca la mediana. El eje horizontal es logarítmico para que la cola alta no aplaste al resto.</p>
      <details>
        <summary>Cómo lo sabemos</summary>
        <h4>De dónde salen</h4>
        <p>Del dataset de la cátedra, publicado en el repositorio de la materia y consumido por el pipeline de avisos en la capa Silver. El notebook trabaja con un corte de 2026 y convierte todo a dólares con un tipo de cambio de <strong>1.565 ARS/USD</strong>.</p>
        <h4>Qué es una fila</h4>
        <p>Un aviso de un vehículo. Se construye una clave única con <code>marca + modelo + versión + año + km</code> y se eliminan duplicados exactos por esa clave y precio, para que el mismo auto no quede en entrenamiento y test.</p>
        <h4>Qué se filtró y qué quedó</h4>
        <div class="tabla"><table>
          <thead><tr><th>Paso</th><th>Filas</th><th>Qué se hizo</th></tr></thead>
          <tbody>
            <tr><td>Dataset crudo</td><td>{str_total_autos}</td><td>Una fila por aviso, 19 columnas.</td></tr>
            <tr><td>Limpieza</td><td>{str_total_autos}</td><td>Precio &gt; 0; motor &lt;= 40 L; consumo 0 a nulo; duplicados exactos eliminados.</td></tr>
            <tr><td>Entrenamiento (80 %)</td><td>{len(df_autos) - len(df_test):,}</td><td>Particionado agrupado por vehículo (<code>GroupShuffleSplit</code>).</td></tr>
            <tr><td>Test (20 %)</td><td>{str_total_test}</td><td>Se separa antes de mirar nada y se usa una sola vez.</td></tr>
            <tr><td>Vehículos compartidos</td><td>0</td><td>La misma clave nunca aparece en train y test.</td></tr>
          </tbody>
        </table></div>
        <p>Se descartan <code>fuente</code>, <code>id_publicacion</code>, <code>ubicacion</code>, <code>version</code>, <code>fecha_ingesta</code>, <code>grupo</code>, <code>url</code> y <code>precio</code> (el precio original en la moneda de origen). La variable objetivo es <code>precio_usd</code>.</p>
        <h4>Qué columnas se descartaron por fuga</h4>
        <p>Ninguna variable de entrada contiene el precio. Se descartó <code>precio</code> porque es la fuente directa del objetivo, y se revisó especialmente <code>modelo</code>: se conoce antes de publicar, así que se queda. Más abajo se prueba que el <code>TargetEncoder</code> de modelo no filtra el precio.</p>
      </details>
    </section>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# SECCIÓN 2: Lo que encontramos
# =============================================================================
st.markdown(
    """
    <div id="hallazgos"></div>
    <section class="seccion">
      <p class="etapa tallo">Lo que encontramos</p>
      <h2>El año de fabricación es lo que más pesa: sin él, el error sube USD 411.</h2>
      <p>Sacar una columna por vez y volver a entrenar muestra qué aporta cada variable. El año duplica la diferencia de cualquier otra: sin él, el MAE pasa de USD 1.471 a USD 1.882. La marca y el modelo juntos aportan USD 207; la potencia, USD 93. El resto aporta menos que un desvío.</p>
    """,
    unsafe_allow_html=True,
)

# Gráfico Altair g-drop con ancho responsivo
chart_drop = chart_impacto_columnas(metricas)
st.altair_chart(chart_drop, use_container_width=True)

st.markdown(
    """
      <p class="nota">Cada barra mide cuánto sube el MAE al sacar esa columna, con la misma validación agrupada. La línea punteada marca un desvío del modelo de referencia (USD 66): lo que no la supera no se distingue del ruido.</p>
      
      <h3 style="margin-top:32px">La ubicación no mueve la aguja: 121 categorías y bajo poder explicativo (H4)</h3>
      <p>En el análisis exploratorio se normalizó la procedencia de los avisos en 121 categorías geográficas (provincia y localidad). La distribución resultó hiperdesbalanceada: pocos centros urbanos concentran la gran masa de publicaciones, mientras decenas de localidades registran menos de 5 autos. Al contrastar precios entre plazas, la razón de correlación (<strong>η² = 0,034</strong>) cayó en la <em>zona roja</em> del semáforo metodológico (< 0,05), evidenciando que las diferencias de valor entre ciudades son insignificantes frente a la dispersión natural del mercado y que incluirla en el modelo sólo sumaría ruido y riesgo de sobreajuste.</p>

      <div class="tarjetas" style="margin: 14px 0 16px;">
        <div class="tarjeta-stat" style="background:#ffffff; border:1px solid #e8ebee; border-radius:10px; padding:12px 16px;">
          <span style="font-size:11px; text-transform:uppercase; letter-spacing:0.06em; color:#5a6578; font-weight:600;">Categorías normalizadas</span>
          <p style="font-size:26px; font-weight:700; color:#0b5d7a; margin:4px 0 0;">121</p>
          <span style="font-size:12px; color:#5a6578;">Alta cardinalidad y fuerte asimetría</span>
        </div>
        <div class="tarjeta-stat" style="background:#ffffff; border:1px solid #e8ebee; border-radius:10px; padding:12px 16px;">
          <span style="font-size:11px; text-transform:uppercase; letter-spacing:0.06em; color:#5a6578; font-weight:600;">Razón de correlación (η²)</span>
          <p style="font-size:26px; font-weight:700; color:#d95926; margin:4px 0 0;">0,034 <span style="font-size:12px; font-weight:600; padding:2px 6px; background:#fee2e2; color:#991b1b; border-radius:4px; vertical-align:middle;">Zona Roja</span></p>
          <span style="font-size:12px; color:#5a6578;">Explica apenas el 3,4 % de la varianza</span>
        </div>
        <div class="tarjeta-stat" style="background:#ffffff; border:1px solid #e8ebee; border-radius:10px; padding:12px 16px;">
          <span style="font-size:11px; text-transform:uppercase; letter-spacing:0.06em; color:#5a6578; font-weight:600;">Decisión de modelado</span>
          <p style="font-size:24px; font-weight:700; color:#0b5d7a; margin:4px 0 0;">Excluida</p>
          <span style="font-size:12px; color:#5a6578;">Descarte preventivo contra sobreajuste</span>
        </div>
      </div>
      <div class="leyenda" style="margin: 8px 0 6px;" aria-label="Leyenda de ubicación">
        <span class="leyenda-titulo">Referencia</span>
        <span><i style="background:#0b5d7a"></i>Precio mediano por plaza (USD)</span>
        <span><i style="background:#d95926; border-radius:0; height:2px;"></i>Mediana global de mercado (USD 10.735)</span>
      </div>
    """,
    unsafe_allow_html=True,
)

# Gráfico de H4 ubicación geográfica con ancho responsivo
chart_ubi = chart_ubicacion_h4()
st.altair_chart(chart_ubi, use_container_width=True)

st.markdown(
    """
      <p class="nota">Comparación de precios medianos en las principales plazas con mayor volumen de publicaciones frente a la mediana global del mercado (USD 10.735). La paridad de valores corrobora que la localización no introduce distorsiones sistemáticas.</p>
      <details>
        <summary>Cómo lo sabemos</summary>
        <h4>Hipótesis H4: Descarte de ubicación geográfica</h4>
        <p>Se evaluó la señal de <code>ubicacion</code> normalizada en 121 categorías geográficas mediante la razón de correlación (<strong>η² = 0,0341</strong>, zona roja de la cátedra por debajo de 0,05). Como se observa en el gráfico, los precios medianos de las plazas principales oscilan de forma indistinguible en torno a la mediana global (USD 10.735). Mantener 121 categorías con decenas de localidades que sólo aportan 1 o 2 avisos forzaría al modelo a memorizar anomalías puntuales en lugar de aprender patrones vehiculares genuinos. Por estas razones, la variable fue excluida definitivamente de la etapa de modelado predictivo.</p>
        <h4>Una hipótesis que se sostuvo</h4>
        <p><strong>El año es la variable que más aporta.</strong> Sin <code>fabricado_en</code> el MAE sube <strong>USD 411</strong> (6,2 desvíos). La importancia por permutación lo confirma: desordenar el año empeora el MAE unos <strong>USD 3.001</strong>, muy por encima del resto.</p>
        <h4>Una que no se sostuvo</h4>
        <p><strong>El kilometraje debería pesar tanto como el año.</strong> Sin <code>kilometraje</code> el MAE sube sólo <strong>USD 29</strong> (0,4 desvíos), menos que el ruido. El kilometraje está muy correlacionado con el año: en la práctica, el año ya cuenta buena parte de su historia.</p>
        <h4>La tabla completa</h4>
        <div class="tabla"><table>
          <thead><tr><th>Variante</th><th>MAE</th><th>Desvío</th><th>Δ vs referencia</th><th>Δ en desvíos</th></tr></thead>
          <tbody>
            <tr><td>referencia (todas)</td><td>1.470,5</td><td>66,1</td><td>0,0</td><td>0,0</td></tr>
            <tr><td>sin modelo (sólo marca)</td><td>1.532,9</td><td>87,2</td><td>62,4</td><td>0,9</td></tr>
            <tr><td>sin marca ni modelo</td><td>1.677,4</td><td>58,9</td><td>206,8</td><td>3,1</td></tr>
            <tr><td>sin fabricado_en</td><td>1.881,7</td><td>64,8</td><td>411,1</td><td>6,2</td></tr>
            <tr><td>sin potencia_hp</td><td>1.563,1</td><td>77,4</td><td>92,5</td><td>1,4</td></tr>
            <tr><td>sin kilometraje</td><td>1.500,0</td><td>69,0</td><td>29,4</td><td>0,4</td></tr>
            <tr><td>sin motor_lt</td><td>1.489,8</td><td>73,2</td><td>19,3</td><td>0,3</td></tr>
            <tr><td>sin consumo_lt_100km</td><td>1.482,6</td><td>59,4</td><td>12,1</td><td>0,2</td></tr>
            <tr><td>sin transmision</td><td>1.475,8</td><td>62,1</td><td>5,3</td><td>0,1</td></tr>
            <tr><td>sin traccion ni combustible</td><td>1.499,9</td><td>60,7</td><td>29,4</td><td>0,4</td></tr>
          </tbody>
        </table></div>
        <h4>Cómo se construyó el gráfico de impacto</h4>
        <p>Se parte del <code>HistGradientBoosting</code> de referencia y se quita una columna por vez, manteniendo la misma validación <code>GroupKFold</code> de 5 pedazos agrupada por vehículo. La diferencia se mide contra la referencia; una diferencia menor a un desvío no es evidencia de nada.</p>
      </details>
    </section>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# SECCIÓN 3: El modelo
# =============================================================================
st.markdown(
    f"""
    <div id="modelo"></div>
    <section class="seccion">
      <p class="etapa tallo">El modelo</p>
      <h2>El modelo acierta a {str_mae_test} en promedio, pero subestima los autos de alta gama.</h2>
      <p>En el test, el <code>HistGradientBoosting</code> logra un <strong>MAE de {str_mae_test}</strong> y un <strong>R² de {r2_test_val:.2f}</strong>. La diagonal es la predicción perfecta: los puntos se pegan a ella en los autos baratos y se abren en la cola alta, sobre todo cuando el aviso original estaba en dólares.</p>
    """,
    unsafe_allow_html=True,
)

# Gráfico Altair g-test con ancho responsivo
chart_test = chart_test_real_vs_pred(df_test)
st.altair_chart(chart_test, use_container_width=True)

st.markdown(
    """
      <p class="nota">Cada punto es un auto del test. El color indica la moneda original del aviso. La brecha crece con el precio: el modelo subestima en promedio <strong>USD 3.179</strong> los autos publicados en USD, contra <strong>USD 6</strong> en los publicados en pesos.</p>
      <h3 style="margin-top:26px">El error por tramo de precio</h3>
      <p>El MAE no es parejo: se multiplica por seis entre el tramo más barato y el más caro. El error relativo mediano, en cambio, se mantiene alrededor del 6–8 %: el modelo se equivoca más en dólares, pero no mucho más en porcentaje.</p>
      <div class="leyenda" style="margin: 10px 0 6px;" aria-label="Leyenda de métricas">
        <span class="leyenda-titulo">Métrica</span>
        <span><i style="background:#0b5d7a"></i>MAE absoluto (USD, eje izq.)</span>
        <span><i style="background:#d95926"></i>Error relativo mediano (%, eje der.)</span>
      </div>
    """,
    unsafe_allow_html=True,
)

# Gráfico Altair g-tramos con ancho responsivo
chart_tramos = chart_error_por_tramo(metricas)
st.altair_chart(chart_tramos, use_container_width=True)

st.markdown(
    f"""
      <details>
        <summary>Cómo lo sabemos</summary>
        <h4>La métrica y por qué</h4>
        <p>El MAE (error absoluto medio en USD) es la métrica oficial porque el costo comercial del error es lineal y se explica solo: “el precio tasado ± {str_mae_test}”. El R² y el error relativo mediano se usan como referencia, no para elegir.</p>
        <h4>Validación</h4>
        <p>Particionado agrupado por vehículo: {len(df_autos) - len(df_test):,} autos en entrenamiento y {str_total_test} en test, con 0 claves compartidas. Dentro de entrenamiento, <code>GroupKFold</code> de 5 pedazos agrupado por la misma clave. Así el modelo no memoriza el precio de una republicación del mismo auto.</p>
        <h4>El piso</h4>
        <p>Un <code>DummyRegressor</code> que siempre predice la mediana da un MAE de <strong>USD 6.201 ± 184</strong>. El modelo lo baja a menos de un cuarto.</p>
        <h4>La comparación completa</h4>
        <div class="tabla"><table>
          <thead><tr><th>Modelo</th><th>MAE (USD)</th><th>Desvío</th><th>R²</th></tr></thead>
          <tbody>
            <tr><td>Lineal (Ridge)</td><td>2.131</td><td>136</td><td>—</td></tr>
            <tr><td>Random forest</td><td>1.494</td><td>77</td><td>—</td></tr>
            <tr><td>HistGradientBoosting</td><td>1.471</td><td>66</td><td>—</td></tr>
          </tbody>
        </table></div>
        <p>El boosting gana por varios desvíos al lineal y se prefiere al bosque por parsimonia: es más rápido y liviano, con un error casi idéntico.</p>
        <h4>El ajuste de la grilla</h4>
        <p>La mejor combinación fue <code>learning_rate=0,1</code>, 400 árboles, 31 hojas y <code>min_samples_leaf=10</code>, con un MAE de validación de <strong>USD 1.411 ± 59</strong>. Son unos USD 60 menos que el boosting sin ajustar: ayuda, pero no cambia la historia. Las 32 combinaciones se ordenan en el notebook; las cuatro mejores están a menos de un desvío entre sí.</p>
        <h4>El test y dónde se equivoca</h4>
        <p>El test del notebook dio <strong>USD 1.435</strong>; al reproducir el pipeline en esta app, <strong>{str_mae_test}</strong> (la diferencia es de 0,3 desvíos y no cambia ninguna conclusión). El R² es {r2_test_val:.3f}. Los errores más caros están en la cola alta: el modelo subestima los autos publicados en USD porque el tipo de cambio implícito del mercado no es el mismo que el tipo de cambio macro usado para unificar.</p>
        <h4>En qué se apoya</h4>
        <p>La importancia por permutación en test deja arriba el año (USD 3.001), la potencia (USD 1.464), la marca (USD 1.149), el modelo (USD 672) y el kilometraje (USD 510). El año no es una fuga: su correlación de Spearman con el precio es {str_spearman}, no 1, y el gráfico muestra mucha dispersión para un mismo año. Es depreciación, no el precio escondido.</p>
        <h4>La auditoría de fugas</h4>
        <p>Con <code>KFold</code> al azar el MAE parece USD 16 más bajo que con <code>GroupKFold</code> (1.454 vs 1.471), pero la diferencia es de sólo 0,2 desvíos porque únicamente 232 de 6.800 vehículos aparecen más de una vez. Con más cosechas mensuales habría más republicaciones y la diferencia crecería; por eso la validación por vehículo queda fija.</p>
        <p>El <code>TargetEncoder</code> de modelo también se revisó: en modelos con 50 o más avisos en entrenamiento el MAE es USD 927; con 10–49 sube a USD 1.538; con 1–9, a USD 3.435; y en los autos de modelos nunca vistos, a USD 7.269. Si hubiera fuga, el modelo andaría mejor en los modelos raros, no peor. Aprende de autos parecidos, no copia precios.</p>
      </details>
    </section>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# SECCIÓN 4: La copa · Explorá
# =============================================================================
st.markdown(
    f"""
    <div id="explora"></div>
    <section class="seccion">
      <p class="etapa copa">La copa · Explorá</p>
      <h2>Cada punto es un auto: el año ordena el mercado y la potencia explica los saltos.</h2>
      <p>El panorama completo son los {str_total_autos} autos. Achicá el universo con los filtros, pasá el mouse para leer cada caso y hacé clic para ver qué dijo el modelo sobre ese auto. La predicción de cada punto sale de una validación agrupada: el modelo que lo predice no lo vio.</p>
    """,
    unsafe_allow_html=True,
)

# Filtros interactivos de exploración
marcas_lista = ["Todas"] + sorted([m for m in df_autos["marca"].dropna().unique() if str(m).strip()])
comb_lista = ["Todos", "Nafta", "Diésel", "Eléctrico", "Híbrido", "GNC", "Sin dato"]
trans_lista = ["Todos", "Automático", "Manual", "Sin dato"]

with st.container():
    f_c1, f_c2, f_c3, f_c4, f_c5 = st.columns([1.5, 1.2, 1.2, 1.5, 1.5])
    with f_c1:
        sel_marca = st.selectbox("Marca", marcas_lista, index=0)
    with f_c2:
        sel_comb = st.selectbox("Combustible", comb_lista, index=0)
    with f_c3:
        sel_trans = st.selectbox("Transmisión", trans_lista, index=0)
    with f_c4:
        min_anio_val = int(df_autos["fabricado_en"].min()) if len(df_autos) > 0 else 1937
        sel_anio_min = st.slider("Año mínimo", min_value=min_anio_val, max_value=2026, value=min_anio_val)
    with f_c5:
        max_precio_val = int(df_autos["precio_usd"].max()) if len(df_autos) > 0 else 160000
        sel_precio_max = st.slider("Precio máximo (USD)", min_value=2000, max_value=max(2000, max_precio_val), value=max_precio_val, step=1000)

st.markdown(
    """
    <div class="leyenda" aria-label="Leyenda de moneda original">
      <span class="leyenda-titulo">Color · moneda original</span>
      <span><i style="background:#2a78d6"></i>ARS (pesos)</span>
      <span><i style="background:#d95926"></i>USD (dólares)</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Aplicar filtros
df_f = df_autos.copy()
if sel_marca != "Todas":
    df_f = df_f[df_f["marca"].astype(str).str.lower() == sel_marca.lower()]

if sel_comb != "Todos":
    if sel_comb == "Sin dato":
        df_f = df_f[df_f["combustible"].isna()]
    else:
        comb_map = {"Nafta": "nafta", "Diésel": "diesel", "Eléctrico": "electrico", "Híbrido": "hibrido", "GNC": "gnc"}
        val_c = comb_map.get(sel_comb, sel_comb.lower())
        df_f = df_f[df_f["combustible"].astype(str).str.lower() == val_c]

if sel_trans != "Todos":
    if sel_trans == "Sin dato":
        df_f = df_f[df_f["transmision"].isna()]
    else:
        trans_map = {"Automático": "automatico", "Manual": "manual"}
        val_t = trans_map.get(sel_trans, sel_trans.lower())
        df_f = df_f[df_f["transmision"].astype(str).str.lower() == val_t]

df_f = df_f[(df_f["fabricado_en"] >= sel_anio_min) & (df_f["precio_usd"] <= sel_precio_max)]

# Grid interactivo: gráfico a la izquierda y tarjeta de detalle a la derecha con ancho responsivo
col_chart, col_card = st.columns([2.4, 1.2])

with col_chart:
    chart_scat = chart_explora_scatter(df_f)
    chart_events = st.altair_chart(chart_scat, use_container_width=True, on_select="rerun")
    
    # Selector complementario de inspección precisa
    opciones_autos = ["— Hacé clic en un punto del gráfico o elegí de la lista —"]
    indices_dict = {}
    for idx, r in df_f.head(300).iterrows():
        label = f"{r['marca']} {r['modelo']} ({int(r['fabricado_en'])}) - USD {r['precio_usd']:,.0f}"
        opciones_autos.append(label)
        indices_dict[label] = idx

    auto_seleccionado_label = st.selectbox(
        "Inspeccionar vehículo específico de la muestra filtrada:",
        opciones_autos,
        label_visibility="collapsed"
    )

with col_card:
    auto_row = None
    # Prioridad: selección por clic en el gráfico
    if chart_events and hasattr(chart_events, "selection") and chart_events.selection:
        puntos = chart_events.selection.get("point_select", []) or chart_events.selection.get("puntos", [])
        if puntos:
            p0 = puntos[0]
            if isinstance(p0, dict):
                m_sub = df_f
                for k, v in p0.items():
                    if k in m_sub.columns:
                        m_sub = m_sub[m_sub[k] == v]
                if len(m_sub) > 0:
                    auto_row = m_sub.iloc[0]

    # Si no se seleccionó por clic, verificar el selectbox complementario
    if auto_row is None and auto_seleccionado_label in indices_dict:
        auto_row = df_f.loc[indices_dict[auto_seleccionado_label]]

    if auto_row is not None:
        precio_r = float(auto_row["precio_usd"])
        pred_r = float(auto_row.get("precio_pred", precio_r))
        err_r = float(auto_row.get("error_abs_usd", abs(precio_r - pred_r)))
        rel_r = float(auto_row.get("error_relativo_pct", (pred_r - precio_r) / (precio_r + 1e-6) * 100.0))
        signo = "+" if (pred_r - precio_r) >= 0 else "−"
        moneda_origen = str(auto_row.get("moneda", "USD")).upper()
        motor_val = f"{auto_row['motor_lt']:.1f}" if pd.notna(auto_row.get("motor_lt")) else "—"
        hp_val = f"{int(auto_row['potencia_hp'])}" if pd.notna(auto_row.get("potencia_hp")) else "—"
        consumo_val = f"{auto_row['consumo_lt_100km']:.1f}" if pd.notna(auto_row.get("consumo_lt_100km")) else "—"

        st.markdown(
            f"""
            <aside class="detalle" id="detalle">
              <h3>{auto_row['marca']} {auto_row['modelo']}</h3>
              <p class="sub">{int(auto_row['fabricado_en'])} · {int(auto_row['kilometraje']):,.0f} km · {str(auto_row.get('combustible', 'Sin dato')).capitalize()}</p>
              <p class="big">{format_usd(precio_r)}</p>
              <p class="error">Predicción: {format_usd(pred_r)} ({signo}{format_usd(err_r)}, {rel_r:+.1f} %)</p>
              <div class="fila"><span>Moneda original</span><span>{moneda_origen}</span></div>
              <div class="fila"><span>Transmisión</span><span>{str(auto_row.get('transmision', 'Sin dato')).capitalize()}</span></div>
              <div class="fila"><span>Tracción</span><span>{str(auto_row.get('traccion', 'Sin dato')).capitalize()}</span></div>
              <div class="fila"><span>Motor / potencia</span><span>{motor_val} L · {hp_val} HP</span></div>
              <div class="fila"><span>Consumo</span><span>{consumo_val} L/100 km</span></div>
            </aside>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <aside class="detalle" id="detalle">
              <h3>Elegí un auto</h3>
              <p class="sub">Hacé clic en un punto del gráfico o seleccionalo en la lista.</p>
              <p class="vacio">El panel muestra marca, modelo, año, kilometraje, precio real y la predicción del modelo, con el error.</p>
              <p class="pista">Doble clic para soltar la selección.</p>
            </aside>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
      <p class="nota">El eje vertical es logarítmico. El color indica la moneda original del aviso, igual que en el gráfico del modelo; los filtros de marca, combustible, transmisión, año mínimo y precio máximo achican el universo. La predicción de cada auto es out-of-fold, no la del modelo final que se usa en “Probá el modelo”.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# SECCIÓN 5: La copa · Probá el modelo
# =============================================================================
st.markdown(
    """
    <div id="proba"></div>
    <section class="seccion">
      <p class="etapa copa">La copa · Probá el modelo</p>
      <h2>Armá un auto y mirá cuánto lo tasaría el modelo.</h2>
      <p>El formulario usa las mismas variables que el pipeline: año, kilometraje, motor, potencia, consumo, transmisión, tracción, combustible, marca y modelo. La predicción se calcula en tu navegador con los árboles del <code>HistGradientBoosting</code> exportados; no se reentrena nada.</p>
    """,
    unsafe_allow_html=True,
)

# Gestión del estado de la simulación
marcas_form_raw = sorted(list(df_autos["marca"].dropna().str.lower().unique()))
default_marca = "toyota" if "toyota" in marcas_form_raw else (marcas_form_raw[0] if marcas_form_raw else "fiat")

if "sim_marca" not in st.session_state:
    st.session_state.sim_marca = default_marca
    st.session_state.sim_modelo = "corolla"
    st.session_state.sim_anio = 2020
    st.session_state.sim_km = 50000
    st.session_state.sim_motor = 2.0
    st.session_state.sim_hp = 170
    st.session_state.sim_consumo = 7.5
    st.session_state.sim_transmision = "Automático"
    st.session_state.sim_traccion = "Delantera"
    st.session_state.sim_combustible = "Nafta"

col_form, col_res = st.columns([1.4, 1.2])

with col_form:
    idx_m = marcas_form_raw.index(st.session_state.sim_marca.lower()) if st.session_state.sim_marca.lower() in marcas_form_raw else 0
    form_marca = st.selectbox("Marca", [m.title() for m in marcas_form_raw], index=idx_m, key="inp_marca")

    # Modelos de la marca elegida
    df_m = df_autos[df_autos["marca"].dropna().str.lower() == form_marca.lower()]
    modelos_form = sorted(list(df_m["modelo"].dropna().str.lower().unique()))
    if not modelos_form:
        modelos_form = ["modelo general"]
    
    idx_mod = 0
    if st.session_state.sim_modelo.lower() in modelos_form:
        idx_mod = modelos_form.index(st.session_state.sim_modelo.lower())
    form_modelo = st.selectbox("Modelo", [m.title() for m in modelos_form], index=idx_mod, key="inp_modelo")

    c_f1, c_f2 = st.columns(2)
    with c_f1:
        form_anio = st.number_input("Año de fabricación", min_value=1937, max_value=2026, value=int(st.session_state.sim_anio), step=1)
        form_motor = st.number_input("Motor (L)", min_value=0.0, max_value=6.4, value=float(st.session_state.sim_motor), step=0.1)
        form_consumo = st.number_input("Consumo (L/100 km)", min_value=1.4, max_value=541.0, value=float(st.session_state.sim_consumo), step=0.1)
        form_traccion = st.selectbox("Tracción", ["Delantera", "Trasera", "4x4"], index=["Delantera", "Trasera", "4x4"].index(st.session_state.sim_traccion) if st.session_state.sim_traccion in ["Delantera", "Trasera", "4x4"] else 0)
    with c_f2:
        form_km = st.number_input("Kilometraje", min_value=0, max_value=8100000, value=int(st.session_state.sim_km), step=1000)
        form_hp = st.number_input("Potencia (HP)", min_value=0, max_value=575, value=int(st.session_state.sim_hp), step=1)
        form_trans = st.selectbox("Transmisión", ["Automático", "Manual"], index=["Automático", "Manual"].index(st.session_state.sim_transmision) if st.session_state.sim_transmision in ["Automático", "Manual"] else 0)
        form_comb = st.selectbox("Combustible", ["Nafta", "Diésel", "Eléctrico", "Híbrido", "GNC"], index=["Nafta", "Diésel", "Eléctrico", "Híbrido", "GNC"].index(st.session_state.sim_combustible) if st.session_state.sim_combustible in ["Nafta", "Diésel", "Eléctrico", "Híbrido", "GNC"] else 0)

    # Botones de acción rápida
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        if st.button("Cargar un ejemplo"):
            ej = df_autos.sample(n=1, random_state=random.randint(1, 9999)).iloc[0]
            st.session_state.sim_marca = str(ej["marca"]).lower()
            st.session_state.sim_modelo = str(ej["modelo"]).lower()
            st.session_state.sim_anio = int(ej["fabricado_en"])
            st.session_state.sim_km = int(ej["kilometraje"])
            st.session_state.sim_motor = float(ej["motor_lt"]) if pd.notna(ej["motor_lt"]) else 1.6
            st.session_state.sim_hp = int(ej["potencia_hp"]) if pd.notna(ej["potencia_hp"]) else 115
            st.session_state.sim_consumo = float(ej["consumo_lt_100km"]) if pd.notna(ej["consumo_lt_100km"]) else 7.5
            st.session_state.sim_transmision = "Automático" if "auto" in str(ej.get("transmision", "")).lower() else "Manual"
            st.session_state.sim_traccion = "Delantera"
            st.session_state.sim_combustible = "Nafta"
            st.rerun()

    with b_col2:
        if st.button("Volver a la mediana"):
            st.session_state.sim_marca = default_marca
            st.session_state.sim_modelo = "corolla"
            st.session_state.sim_anio = 2020
            st.session_state.sim_km = 50000
            st.session_state.sim_motor = 2.0
            st.session_state.sim_hp = 170
            st.session_state.sim_consumo = 7.5
            st.session_state.sim_transmision = "Automático"
            st.session_state.sim_traccion = "Delantera"
            st.session_state.sim_combustible = "Nafta"
            st.rerun()

with col_res:
    # Preparar fila para el pipeline entrenado
    df_pred_input = pd.DataFrame([{
        "marca": form_marca.lower(),
        "modelo": form_modelo.lower(),
        "fabricado_en": int(form_anio),
        "kilometraje": int(form_km),
        "moneda": "USD",
        "motor_lt": float(form_motor),
        "potencia_hp": float(form_hp),
        "transmision": form_trans.lower(),
        "traccion": form_traccion.lower(),
        "combustible": "diesel" if form_comb == "Diésel" else form_comb.lower(),
        "consumo_lt_100km": float(form_consumo),
    }])

    precio_tasado_usd = float(modelo.predict(df_pred_input)[0])
    p_mercado = percentile_of(precio_tasado_usd, df_autos["precio_usd"])
    p_pct = int(round(p_mercado * 100))

    # Cálculo exclusivo del margen comercial mediante el error relativo porcentual mediano
    lim_inf_usd = precio_tasado_usd * (1.0 - porcentaje_margen)
    lim_sup_usd = precio_tasado_usd * (1.0 + porcentaje_margen)

    precio_tasado_ars = precio_tasado_usd * TIPO_CAMBIO_REF
    lim_inf_ars = precio_tasado_ars * (1.0 - porcentaje_margen)
    lim_sup_ars = precio_tasado_ars * (1.0 + porcentaje_margen)

    pos_marcador = min(100.0, max(0.0, p_mercado * 100.0))

    # Advertencias fuera de rango o extrapolación
    warnings = warnings_for(
        form_marca, form_modelo, form_anio, form_km, form_motor, form_hp, form_consumo,
        marcas_validas, modelos_validos
    )
    aviso_html = ""
    if warnings:
        aviso_html = f'<p class="aviso">{" ".join(warnings)}</p>'

    # Comparables más cercanos
    comps = comparables_for(form_marca, form_modelo, form_anio, form_km, df_autos, n=4)
    comps_html = ""
    if comps:
        comps_items = "".join([
            f"<li><b>{c['marca']} {c['modelo']}</b> <span>· {c['anio']} · {c['km']:,} km</span><br>"
            f"<span>{format_usd(c['precio_real'])} real · {format_usd(c['precio_pred'])} estimado</span></li>"
            for c in comps
        ])
        comps_html = f'<ul class="comparables"><li style="border-top:none;padding-top:0"><b>Comparables en el dataset</b></li>{comps_items}</ul>'

    st.markdown(
        f"""
        <aside class="resultado" id="resultado" aria-live="polite">
          <p class="eyebrow">Tasación estimada</p>
          <p class="precio" id="precio">{format_usd(precio_tasado_usd)}</p>
          <p class="precio-ars" style="font-size: 16px; opacity: 0.92; margin: 3px 0 10px; font-weight: 500;">
            ≈ {format_ars(precio_tasado_usd, TIPO_CAMBIO_REF)} <span style="font-size: 13px; opacity: 0.78;">(cotización ref. $ {TIPO_CAMBIO_REF:,.0f}/USD)</span>
          </p>
          <p class="margen" id="margen" style="font-size: 14.5px; font-weight: 600; margin-bottom: 2px;">
            Rango sugerido: {format_usd(lim_inf_usd)} a {format_usd(lim_sup_usd)} (± {str_margen_pct})
          </p>
          <p class="margen-ars" style="font-size: 13px; opacity: 0.88; margin-top: 0; margin-bottom: 6px;">
            En pesos: {format_ars(lim_inf_usd, TIPO_CAMBIO_REF)} a {format_ars(lim_sup_usd, TIPO_CAMBIO_REF)}
          </p>
          <p class="margen-nota" style="font-size: 12px; line-height: 1.35; opacity: 0.85; background: rgba(255,255,255,0.08); padding: 6px 8px; border-radius: 6px; margin: 8px 0 10px;">
            ℹ️ Margen comercial proporcional al valor (± {str_margen_pct}): la dispersión de mercado escala con el valor del vehículo.
          </p>
          <div class="barra"><i id="marcador" style="left: {pos_marcador:.1f}%;"></i></div>
          <div class="escala"><span>USD 1.000</span><span>USD 160.000</span></div>
          <p class="margen" id="percentil" style="margin-top:8px">Percentil {p_pct} del mercado ({str_total_autos} autos).</p>
          <div id="aviso">{aviso_html}</div>
          {comps_html}
        </aside>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
      <p class="nota">El rango sugerido se calcula aplicando el error relativo porcentual mediano (± {str_margen_pct}). A diferencia de un monto fijo en dólares, el margen comercial es proporcional al valor del auto: en un vehículo de USD 8.000 el margen esperado es de unos USD 500, mientras que en uno de USD 40.000 supera los USD 2.600, reflejando fielmente que los autos de mayor valor exhiben una mayor dispersión. Para modelos con pocos avisos, la estimación es menos confiable. Los valores fuera del rango visto en entrenamiento se avisan y se usan igual, pero la app no los presenta como una tasación confiable.</p>
      <details>
        <summary>Cómo lo sabemos</summary>
        <h4>Qué modelo corre en la app</h4>
        <p>El mismo <code>HistGradientBoosting</code> con <code>learning_rate=0,1</code>, 400 árboles, 31 hojas y <code>min_samples_leaf=10</code>, entrenado sobre los {str_total_autos} autos limpios. El precio se modela en escala <code>log1p</code> y se vuelve a dólares con <code>expm1</code>.</p>
        <h4>Cómo se convierte una fila nueva</h4>
        <p>Se normalizan las categorías (transmisión, tracción, combustible), se construye <code>marca + modelo</code>, se imputan faltantes con la mediana o la moda del entrenamiento, se escalan las numéricas y se aplica el <code>TargetEncoder</code> de modelo. Después se recorren los 400 árboles y se suma la predicción base. La app exporta esos árboles a JSON; no hay un servidor ni un reentrenamiento.</p>
        <h4>Qué significa el margen</h4>
        <p>El margen es proporcional al valor del vehículo (± {str_margen_pct}, derivado del error relativo porcentual mediano observado en el conjunto de test). No es un intervalo de confianza gaussiano simétrico en dólares: es el desvío mediano porcentual empírico. En autos de alta gama el error en dólares es mayor pero en porcentaje se mantiene estable alrededor del 6–8 %, por lo que una banda multiplicativa modela el riesgo comercial de forma mucho más rigurosa que un MAE fijo.</p>
        <h4>Qué se puede y qué no se puede cargar</h4>
        <p>Los campos respetan los mínimos y máximos del entrenamiento: año 1937–2026, kilometraje 0–8.100.000, motor 0–6,4 L, potencia 0–575 HP y consumo 1,4–541 L/100 km. Si el usuario escribe un valor fuera de rango, aparece un aviso y la tasación se muestra como no confiable.</p>
      </details>
    </section>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown(
    """
    <footer class="footer-proyecto">
      Ciencia de Datos · UTN FRM · 2026 · Grupo 5K09-03 · Datos: pipeline de avisos de autos usados · Gráficos: Altair (Vega-Lite) · Modelo: HistGradientBoosting con target logarítmico.
    </footer>
    """,
    unsafe_allow_html=True,
)
