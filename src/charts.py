"""
Visualizaciones Altair que replican exactamente los gráficos del HTML de referencia:
1. g-hist: Histograma de precios en escala logarítmica con línea de mediana en USD 10.735.
2. g-drop: Impacto por columna (cuánto sube el MAE al retirarla) con línea de 1 desvío en USD 66.
3. g-test: Real vs. predicho en el Test set (1.782 autos) por moneda original con diagonal de ajuste.
4. g-tramos: Error MAE por tramos de precio de mercado.
5. g-explora: Dispersión año vs. precio (escala log) coloreado por moneda original.

Grupo 5K09-03 · UTN FRM · Ciencia de Datos
"""

import altair as alt
import pandas as pd
import numpy as np

# Deshabilitar límite de filas para visualización fluida
alt.data_transformers.disable_max_rows()

# Paleta exacta del HTML de referencia
COLOR_AZUL = "#2a78d6"       # Publicaciones en ARS y barras principales
COLOR_NARANJA = "#d95926"    # Publicaciones en USD y líneas de referencia
COLOR_NEUTRO = "#7a7974"     # Línea diagonal y neutros
COLOR_ACENTO = "#0b5d7a"
COLOR_COPA = "#b45309"


def chart_histograma_precios(df: pd.DataFrame) -> alt.Chart:
    """
    g-hist (Los datos):
    Histograma de precios en tramos logarítmicos con línea de mediana en USD 10.735.
    """
    val_min = 900.0
    val_max = 160000.0
    # 36 tramos logarítmicos idénticos a la referencia
    log_bins = np.linspace(np.log10(val_min), np.log10(val_max), 37)
    bins = 10 ** log_bins

    precios = df["precio_usd"].dropna()
    counts, _ = np.histogram(precios, bins=bins)

    hist_data = []
    for i in range(len(counts)):
        b_start = bins[i]
        b_end = bins[i + 1]
        l_start = log_bins[i]
        l_end = log_bins[i + 1]
        label = f"USD {b_start/1000:,.0f}k – {b_end/1000:,.0f}k"
        hist_data.append({
            "bin_start": b_start,
            "bin_end": b_end,
            "log_start": l_start,
            "log_end": l_end,
            "log_mid": (l_start + l_end) / 2,
            "count": int(counts[i]),
            "label": label,
            "cero": 0
        })
    df_hist = pd.DataFrame(hist_data)

    barras = (
        alt.Chart(df_hist)
        .mark_bar(
            color=COLOR_AZUL,
            cornerRadiusTopLeft=2,
            cornerRadiusTopRight=2,
            opacity=0.88
        )
        .encode(
            x=alt.X(
                "log_mid:Q",
                title="Precio (USD, escala log)",
                scale=alt.Scale(domain=[2.95, 5.25], nice=False),
                axis=alt.Axis(
                    values=[3.0, 3.30103, 3.69897, 4.0, 4.30103, 4.69897, 5.0],
                    labelExpr="format(pow(10, datum.value), ',.0f')",
                    grid=True,
                    gridColor="#ececec",
                    domainColor="#bbbbbb"
                )
            ),
            y=alt.Y(
                "count:Q",
                title="Autos",
                axis=alt.Axis(grid=True, gridColor="#ececec", domainColor="#bbbbbb")
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Tramo"),
                alt.Tooltip("count:Q", title="Autos", format=",")
            ]
        )
    )

    # Mediana en USD 10.735
    mediana_val = 10735.0
    df_med = pd.DataFrame([{
        "x": np.log10(mediana_val),
        "y": 0,
        "y2": int(df_hist["count"].max() * 1.05),
        "t": "Mediana USD 10.735"
    }])

    regla = (
        alt.Chart(df_med)
        .mark_rule(color=COLOR_NARANJA, strokeWidth=2)
        .encode(x="x:Q", y="y:Q", y2="y2:Q")
    )

    texto = (
        alt.Chart(df_med)
        .mark_text(
            align="left",
            color=COLOR_NARANJA,
            dx=6,
            dy=-4,
            fontSize=12,
            fontWeight=600
        )
        .encode(x="x:Q", y="y2:Q", text="t:N")
    )

    return (
        (barras + regla + texto)
        .properties(height=300, width="container")
        .configure_view(stroke=None)
    )


def chart_impacto_columnas(metricas: dict = None) -> alt.Chart:
    """
    g-drop (Lo que encontramos):
    Cuánto sube el MAE al sacar cada columna (validación GroupKFold).
    """
    drop_data = pd.DataFrame([
        {"variante": "sin transmision", "label": "Transmisión", "mae": 1475.8, "sd": 62.1, "delta": 5.3, "aporta": False},
        {"variante": "sin consumo_lt_100km", "label": "Consumo", "mae": 1482.6, "sd": 59.4, "delta": 12.1, "aporta": False},
        {"variante": "sin motor_lt", "label": "Motor", "mae": 1489.8, "sd": 73.2, "delta": 19.3, "aporta": False},
        {"variante": "sin kilometraje", "label": "Kilometraje", "mae": 1500.0, "sd": 69.0, "delta": 29.4, "aporta": False},
        {"variante": "sin traccion ni combustible", "label": "Tracción + combustible", "mae": 1499.9, "sd": 60.7, "delta": 29.4, "aporta": False},
        {"variante": "sin modelo (sólo marca)", "label": "Modelo (sólo marca)", "mae": 1532.9, "sd": 87.2, "delta": 62.4, "aporta": False},
        {"variante": "sin potencia_hp", "label": "Potencia", "mae": 1563.1, "sd": 77.4, "delta": 92.5, "aporta": True},
        {"variante": "sin marca ni modelo", "label": "Marca + modelo", "mae": 1677.4, "sd": 58.9, "delta": 206.8, "aporta": True},
        {"variante": "sin fabricado_en", "label": "Año de fabricación", "mae": 1881.7, "sd": 64.8, "delta": 411.1, "aporta": True}
    ])

    barras = (
        alt.Chart(drop_data)
        .mark_bar(cornerRadiusEnd=3, height=20, color=COLOR_AZUL)
        .encode(
            y=alt.Y(
                "label:N",
                sort=alt.EncodingSortField(field="delta", order="ascending"),
                title=None,
                axis=alt.Axis(labelFontSize=12, domainColor="#bbbbbb")
            ),
            x=alt.X(
                "delta:Q",
                title="Cuánto sube el MAE (USD)",
                axis=alt.Axis(grid=True, gridColor="#ececec", domainColor="#bbbbbb", format="$,.0f")
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Columna"),
                alt.Tooltip("delta:Q", title="Δ MAE (USD)", format="$,.1f"),
                alt.Tooltip("mae:Q", title="MAE resultante (USD)", format="$,.1f")
            ]
        )
    )

    texto = (
        alt.Chart(drop_data)
        .mark_text(align="left", color="#1a1a1a", dx=6, fontSize=11, fontWeight=600)
        .encode(
            y=alt.Y("label:N", sort=alt.EncodingSortField(field="delta", order="ascending")),
            x="delta:Q",
            text=alt.Text("delta:Q", format="+$,.0f")
        )
    )

    df_linea = pd.DataFrame([{"x": 66.1, "y": 8.8, "t": "1 desvío = USD 66"}])
    regla = (
        alt.Chart(df_linea)
        .mark_rule(color=COLOR_NARANJA, strokeDash=[5, 4], strokeWidth=1.5)
        .encode(x="x:Q")
    )

    texto_regla = (
        alt.Chart(df_linea)
        .mark_text(align="left", color=COLOR_NARANJA, dx=4, dy=-8, fontSize=11)
        .encode(x="x:Q", y=alt.value(20), text="t:N")
    )

    return (
        (barras + texto + regla + texto_regla)
        .properties(height=320, width="container")
        .configure_view(stroke=None)
    )


def chart_test_real_vs_pred(df_test: pd.DataFrame) -> alt.Chart:
    """
    g-test (El modelo):
    Real vs predicho en el Test (1.782 autos) en escala log-log, coloreado por moneda original.
    """
    cols_needed = [
        "precio_usd", "precio_pred", "moneda", "marca", "modelo",
        "fabricado_en", "error_abs_usd", "error_relativo_pct"
    ]
    df_plot = df_test[[c for c in cols_needed if c in df_test.columns]].copy()

    puntos = (
        alt.Chart(df_plot)
        .mark_circle(opacity=0.55, size=20)
        .encode(
            x=alt.X(
                "precio_usd:Q",
                scale=alt.Scale(type="log", domain=[900, 160000], nice=False),
                title="Precio real (USD, escala log)",
                axis=alt.Axis(
                    values=[1000, 2000, 5000, 10000, 20000, 50000, 100000],
                    format="$,.0f",
                    grid=True,
                    gridColor="#ececec",
                    domainColor="#bbbbbb"
                )
            ),
            y=alt.Y(
                "precio_pred:Q",
                scale=alt.Scale(type="log", domain=[900, 160000], nice=False),
                title="Precio predicho (USD, escala log)",
                axis=alt.Axis(
                    values=[1000, 2000, 5000, 10000, 20000, 50000, 100000],
                    format="$,.0f",
                    grid=True,
                    gridColor="#ececec",
                    domainColor="#bbbbbb"
                )
            ),
            color=alt.Color(
                "moneda:N",
                scale=alt.Scale(domain=["ARS", "USD"], range=[COLOR_AZUL, COLOR_NARANJA]),
                title="Moneda original"
            ),
            tooltip=[
                alt.Tooltip("marca:N", title="Marca"),
                alt.Tooltip("modelo:N", title="Modelo"),
                alt.Tooltip("fabricado_en:Q", title="Año", format="d"),
                alt.Tooltip("precio_usd:Q", title="Precio real", format="$,.0f"),
                alt.Tooltip("precio_pred:Q", title="Predicción", format="$,.0f"),
                alt.Tooltip("error_relativo_pct:Q", title="Error relativo (%)", format=".1f")
            ]
        )
    )

    df_diag = pd.DataFrame([{"x": 900, "y": 900}, {"x": 160000, "y": 160000}])
    diagonal = (
        alt.Chart(df_diag)
        .mark_line(color=COLOR_NEUTRO, strokeDash=[6, 5], strokeWidth=1.5)
        .encode(x="x:Q", y="y:Q")
    )

    return (
        (puntos + diagonal)
        .properties(height=360, width="container")
        .configure_view(stroke=None)
    )


def chart_error_por_tramo(metricas: dict = None) -> alt.Chart:
    """
    g-tramos (El modelo):
    Barras de MAE por tramo de precio del mercado automotor.
    """
    df_tramos = pd.DataFrame([
        {"tramo": "USD 1k–8k", "mae": 633, "rel": 0.076, "n": 368, "texto": "USD 633\n(7.6%)"},
        {"tramo": "USD 8k–10k", "mae": 710, "rel": 0.065, "n": 348, "texto": "USD 710\n(6.5%)"},
        {"tramo": "USD 10k–13k", "mae": 879, "rel": 0.061, "n": 353, "texto": "USD 879\n(6.1%)"},
        {"tramo": "USD 13k–19k", "mae": 1199, "rel": 0.060, "n": 356, "texto": "USD 1,199\n(6.0%)"},
        {"tramo": "USD 19k–105k", "mae": 3856, "rel": 0.080, "n": 357, "texto": "USD 3,856\n(8.0%)"}
    ])

    barras = (
        alt.Chart(df_tramos)
        .mark_bar(color=COLOR_AZUL, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X(
                "tramo:N",
                sort=["USD 1k–8k", "USD 8k–10k", "USD 10k–13k", "USD 13k–19k", "USD 19k–105k"],
                title="Tramo de precio (USD)",
                axis=alt.Axis(labelAngle=0, domainColor="#bbbbbb", gridColor="#ececec")
            ),
            y=alt.Y(
                "mae:Q",
                title="MAE (USD)",
                axis=alt.Axis(grid=True, gridColor="#ececec", domainColor="#bbbbbb", format="$,.0f")
            ),
            tooltip=[
                alt.Tooltip("tramo:N", title="Tramo"),
                alt.Tooltip("mae:Q", title="MAE (USD)", format="$,.0f"),
                alt.Tooltip("rel:Q", title="Error relativo mediano", format=".1%"),
                alt.Tooltip("n:Q", title="Autos", format=",")
            ]
        )
    )

    texto = (
        alt.Chart(df_tramos)
        .mark_text(color="#1a1a1a", dy=-8, fontSize=10, fontWeight=600)
        .encode(
            x=alt.X("tramo:N", sort=["USD 1k–8k", "USD 8k–10k", "USD 10k–13k", "USD 13k–19k", "USD 19k–105k"]),
            y="mae:Q",
            text=alt.Text("mae:Q", format="$,.0f")
        )
    )

    return (
        (barras + texto)
        .properties(height=300, width="container")
        .configure_view(stroke=None)
    )


def chart_explora_scatter(df_filtrado: pd.DataFrame) -> alt.Chart:
    """
    g-explora (La copa · Explorá):
    Dispersión de vehículos: X = Año de fabricación, Y = Precio (USD, log), Color = Moneda original.
    """
    cols_needed = [
        "fabricado_en", "precio_usd", "moneda", "marca", "modelo",
        "kilometraje", "precio_pred", "error_relativo_pct"
    ]
    df_plot = df_filtrado[[c for c in cols_needed if c in df_filtrado.columns]].copy()

    chart = (
        alt.Chart(df_plot)
        .mark_circle(opacity=0.45, size=22)
        .encode(
            x=alt.X(
                "fabricado_en:Q",
                title="Año de fabricación",
                axis=alt.Axis(
                    format="d",
                    labelAngle=0,
                    values=[1940, 1960, 1980, 2000, 2020],
                    grid=True,
                    gridColor="#ececec",
                    domainColor="#bbbbbb"
                ),
                scale=alt.Scale(domain=[1937, 2030], nice=False)
            ),
            y=alt.Y(
                "precio_usd:Q",
                title="Precio (USD, escala log)",
                scale=alt.Scale(type="log", domain=[900, 160000], nice=False),
                axis=alt.Axis(
                    values=[1000, 2000, 5000, 10000, 20000, 50000, 100000],
                    format="$,.0f",
                    grid=True,
                    gridColor="#ececec",
                    domainColor="#bbbbbb"
                )
            ),
            color=alt.Color(
                "moneda:N",
                scale=alt.Scale(domain=["ARS", "USD"], range=[COLOR_AZUL, COLOR_NARANJA]),
                legend=None
            ),
            tooltip=[
                alt.Tooltip("marca:N", title="Marca"),
                alt.Tooltip("modelo:N", title="Modelo"),
                alt.Tooltip("fabricado_en:Q", title="Año", format="d"),
                alt.Tooltip("kilometraje:Q", title="Kilometraje", format=","),
                alt.Tooltip("moneda:N", title="Moneda original"),
                alt.Tooltip("precio_usd:Q", title="Precio real", format="$,.0f"),
                alt.Tooltip("precio_pred:Q", title="Predicción del modelo", format="$,.0f"),
                alt.Tooltip("error_relativo_pct:Q", title="Error relativo (%)", format=".1f")
            ]
        )
        .properties(height=430, width="container")
        .interactive()
        .configure_view(stroke=None)
    )
    return chart
