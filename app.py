"""
================================================================================
  MÓDULO 3: DASHBOARD STREAMLIT — Sistema de Predicción Turística
  Ejecutar: streamlit run app.py
================================================================================
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Tourism Intelligence Hub",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta de colores coherente
COLORES = {
    "primary": "#1A6B8A",
    "secondary": "#E8A838",
    "accent": "#C94040",
    "success": "#2E8B57",
    "neutral": "#5A6472",
    "bg_dark": "#0F1923",
    "bg_card": "#1C2B3A",
}

PALETTE = [
    "#1A6B8A", "#E8A838", "#C94040", "#2E8B57",
    "#7B5EA7", "#E07B54", "#3D9BD4", "#88C057",
]

st.markdown("""
<style>
    /* Fondo principal oscuro */
    .stApp { background-color: #0F1923; color: #E8EDF2; }
    
    /* Cards de métricas */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1C2B3A 0%, #162330 100%);
        border: 1px solid rgba(26,107,138,0.27);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(26,107,138,0.15);
    }
    div[data-testid="metric-container"] label { color: #8AACBF !important; font-size: 0.8rem; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #E8A838 !important; font-size: 1.8rem; font-weight: 700;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #0D1720; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] { color: #8AACBF; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #E8A838 !important; border-bottom-color: #E8A838 !important; }
    
    /* Headers */
    h1, h2, h3 { color: #E8EDF2 !important; }
    
    /* Botones */
    .stButton > button {
        background: linear-gradient(135deg, #1A6B8A, #0D4F6A);
        color: white; border: none; border-radius: 8px;
        padding: 8px 24px; font-weight: 600;
    }
    
    /* Dataframes */
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    
    /* Separador secciones */
    .section-header {
        background: linear-gradient(90deg, rgba(26,107,138,0.13), transparent);
        border-left: 3px solid #E8A838;
        padding: 8px 16px; margin: 16px 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS UI
# ─────────────────────────────────────────────────────────────────────────────

def plotly_theme(fig: go.Figure, height: int = 400) -> go.Figure:
    """Aplica el tema oscuro corporativo a cualquier figura Plotly."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(28,43,58,0.6)",
        font=dict(family="Inter, sans-serif", color="#E8EDF2", size=12),
        title_font=dict(size=15, color="#E8A838", family="Inter, sans-serif"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8AACBF")),
        xaxis=dict(gridcolor="rgba(26,107,138,0.13)", showgrid=True, linecolor="rgba(26,107,138,0.27)"),
        yaxis=dict(gridcolor="rgba(26,107,138,0.13)", showgrid=True, linecolor="rgba(26,107,138,0.27)"),
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def card_metric(titulo, valor, delta=None, sufijo=""):
    st.metric(label=titulo, value=f"{valor:,}{sufijo}", delta=delta)


def seccion(titulo: str):
    st.markdown(f'<div class="section-header"><b>{titulo}</b></div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CARGA Y CACHÉ DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="⚙️ Procesando datos... (puede tardar 1-2 min la primera vez)")
def cargar_y_procesar(archivo_bytes: bytes, nombre_archivo: str) -> pd.DataFrame:
    """Parsea el Excel y ejecuta el pipeline completo (cacheado)."""
    import tempfile, os
    from data_processor import procesar_excel

    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp.write(archivo_bytes)
        tmp_path = tmp.name

    try:
        df = procesar_excel(tmp_path)
        df["Fecha"] = pd.to_datetime(df["Fecha"])
        return df
    finally:
        try: os.unlink(tmp_path)
        except: pass


@st.cache_data(show_spinner="🔮 Calculando predicciones...")
def obtener_forecast(_df: pd.DataFrame):
    from ml_models import forecast_prophet
    return forecast_prophet(_df, periodos=26, freq="W")

@st.cache_data(show_spinner="🛒 Analizando canastas...")
def obtener_reglas(_df: pd.DataFrame):
    from ml_models import calcular_reglas_asociacion
    return calcular_reglas_asociacion(_df)

@st.cache_data(show_spinner="👥 Segmentando clientes...")
def obtener_rfm(_df: pd.DataFrame):
    from ml_models import calcular_rfm, segmentar_clientes_rfm
    rfm = calcular_rfm(_df)
    return segmentar_clientes_rfm(rfm)

@st.cache_data(show_spinner="📅 Calculando estacionalidad...")
def obtener_estacionalidad(_df: pd.DataFrame):
    from ml_models import analisis_estacionalidad
    return analisis_estacionalidad(_df)

@st.cache_data(show_spinner="🏨 Ranking de hoteles...")
def obtener_ranking_hoteles(_df: pd.DataFrame):
    from ml_models import ranking_alojamientos
    return ranking_alojamientos(_df)

@st.cache_data(show_spinner="📊 Analizando operadores...")
def obtener_rendimiento_operadores(_df: pd.DataFrame):
    from ml_models import rendimiento_operadores
    return rendimiento_operadores(_df)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0;'>
        <h1 style='color:#E8A838; font-size:1.8rem; margin:0;'>🏔️</h1>
        <h2 style='color:#E8EDF2; font-size:1.1rem; margin:4px 0;'>Tourism Intelligence</h2>
        <p style='color:#8AACBF; font-size:0.75rem; margin:0;'>Powered by Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    archivo_subido = st.file_uploader(
        "📂 Cargar Excel Operativo",
        type=["xlsx", "xls"],
        help="El mismo Excel con el formato de operaciones diarias (fecha → filas de datos)"
    )

    st.divider()
    st.markdown("### 🎛️ Filtros Globales")

    # Los filtros se activan después de cargar datos
    filtros = {
        "anios": None,
        "categorias": None,
        "operadores": None,
    }

# ─────────────────────────────────────────────────────────────────────────────
# PANTALLA DE BIENVENIDA (sin datos)
# ─────────────────────────────────────────────────────────────────────────────

if archivo_subido is None:
    st.markdown("""
    <div style='text-align:center; padding: 80px 20px;'>
        <h1 style='color:#E8A838; font-size:3rem; margin-bottom:8px;'>🏔️ Tourism Intelligence Hub</h1>
        <p style='color:#8AACBF; font-size:1.2rem; max-width:600px; margin:0 auto 40px;'>
            Transformá 10 años de operaciones en decisiones estratégicas.<br>
            Cargá tu Excel para comenzar.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    modulos = [
        ("📅", "Forecasting", "Predicción de demanda\npara los próximos 6 meses"),
        ("🛒", "Market Basket", "Qué actividades se\ncompran juntas"),
        ("👥", "Segmentación RFM", "Champions, Leales,\nEn Riesgo y más"),
        ("🏨", "B2B Ranking", "Los alojamientos que\nmás clientes derivan"),
    ]
    for col, (ico, tit, desc) in zip([col1, col2, col3, col4], modulos):
        with col:
            st.markdown(f"""
            <div style='background:#1C2B3A; border:1px solid rgba(26,107,138,0.27); border-radius:12px;
                        padding:24px; text-align:center; height:160px;'>
                <div style='font-size:2rem;'>{ico}</div>
                <b style='color:#E8A838;'>{tit}</b>
                <p style='color:#8AACBF; font-size:0.8rem; margin-top:8px;'>{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

df_raw = cargar_y_procesar(archivo_subido.read(), archivo_subido.name)

# Filtros dinámicos en sidebar
with st.sidebar:
    anios_disponibles = sorted(df_raw["Anio"].unique())
    anios_sel = st.multiselect(
        "📆 Años", anios_disponibles, default=anios_disponibles,
        help="Filtrá por año para comparar períodos"
    )

    cats_disponibles = sorted(df_raw["Categoria_Actividad"].unique())
    cats_sel = st.multiselect(
        "🎯 Categorías de Actividad", cats_disponibles, default=cats_disponibles
    )

    ops_disponibles = sorted(df_raw["OPERADOR"].unique()) if "OPERADOR" in df_raw.columns else []
    ops_sel = st.multiselect(
        "👤 Operadores", ops_disponibles, default=ops_disponibles[:20]
    )

    st.divider()
    st.caption(f"📊 Dataset: {len(df_raw):,} registros")
    st.caption(f"📅 {df_raw['Fecha'].min().date()} → {df_raw['Fecha'].max().date()}")

# Aplicar filtros
df = df_raw[df_raw["Anio"].isin(anios_sel)].copy()
if cats_sel:
    df = df[df["Categoria_Actividad"].isin(cats_sel)]


# ─────────────────────────────────────────────────────────────────────────────
# TABS PRINCIPALES
# ─────────────────────────────────────────────────────────────────────────────

tab_home, tab_forecast, tab_estacionalidad, tab_mba, tab_rfm, tab_b2b, tab_datos = st.tabs([
    "🏠 Resumen",
    "🔮 Forecasting",
    "📅 Estacionalidad",
    "🛒 Canastas",
    "👥 Clientes RFM",
    "🤝 B2B & Operadores",
    "📋 Datos Crudos",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: RESUMEN EJECUTIVO
# ═══════════════════════════════════════════════════════════════════════════════

with tab_home:
    st.markdown("## 🏠 Resumen Ejecutivo")

    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    total_pax = int(df["Q"].sum())
    total_reservas = len(df)
    clientes_unicos = df["ID_Cliente"].nunique()
    pax_promedio = round(df["Q"].mean(), 1)
    tasa_recurrencia = round(df[df["Es_Recurrente"] == 1]["ID_Cliente"].nunique() / max(clientes_unicos, 1) * 100, 1)

    with col1: card_metric("🧳 Total Pasajeros", total_pax)
    with col2: card_metric("📋 Reservas", total_reservas)
    with col3: card_metric("👥 Clientes Únicos", clientes_unicos)
    with col4: card_metric("📊 Pax Promedio", pax_promedio)
    with col5: card_metric("🔄 Tasa Recurrencia", tasa_recurrencia, sufijo="%")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        seccion("📈 Evolución Anual de Pasajeros")
        evol = df.groupby("Anio")["Q"].sum().reset_index()
        fig = px.bar(
            evol, x="Anio", y="Q",
            text_auto=True,
            color="Q",
            color_continuous_scale=["#1A6B8A", "#E8A838"],
            labels={"Q": "Pasajeros", "Anio": "Año"},
        )
        fig.update_traces(textfont_size=11, textposition="outside")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(plotly_theme(fig, 350), use_container_width=True)

    with col_b:
        seccion("🏷️ Distribución por Categoría de Actividad")
        cats = df.groupby("Categoria_Actividad")["Q"].sum().reset_index().sort_values("Q", ascending=False)
        fig = px.pie(
            cats, values="Q", names="Categoria_Actividad",
            hole=0.55, color_discrete_sequence=PALETTE,
        )
        fig.update_traces(textinfo="percent+label", textfont_size=11)
        st.plotly_chart(plotly_theme(fig, 350), use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        seccion("📅 Heatmap: Pasajeros por Mes y Año")
        pivot = df.pivot_table(values="Q", index="Mes", columns="Anio", aggfunc="sum", fill_value=0)
        meses = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[meses[m-1] for m in pivot.index],
            colorscale=[[0, "#0F1923"], [0.5, "#1A6B8A"], [1, "#E8A838"]],
            text=pivot.values, texttemplate="%{text}",
            hovertemplate="Año: %{x}<br>Mes: %{y}<br>Pax: %{z}<extra></extra>",
        ))
        st.plotly_chart(plotly_theme(fig, 380), use_container_width=True)

    with col_d:
        seccion("📊 Top 10 Actividades Más Vendidas")
        top_act = df.groupby("ACTIVIDAD")["Q"].sum().reset_index().sort_values("Q", ascending=False).head(10)
        fig = px.bar(
            top_act, x="Q", y="ACTIVIDAD", orientation="h",
            color="Q", color_continuous_scale=["#1A6B8A", "#E8A838"],
            labels={"Q": "Pasajeros", "ACTIVIDAD": ""},
        )
        fig.update_coloraxes(showscale=False)
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(plotly_theme(fig, 380), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════

with tab_forecast:
    st.markdown("## 🔮 Forecasting de Demanda — Próximos 6 Meses")
    st.info("📌 El modelo Prophet detecta automáticamente estacionalidad anual, semanal y tendencias de largo plazo.")

    col_left, col_right = st.columns([3, 1])
    with col_right:
        periodos = st.slider("Semanas a predecir", 4, 52, 26, step=4)
        freq = st.radio("Frecuencia", ["Semanal (W)", "Mensual (M)"], index=0)
        freq_code = "W" if "Semanal" in freq else "M"

    with st.spinner("Entrenando modelo..."):
        try:
            modelo, forecast, serie = obtener_forecast(df)

            # Gráfico principal
            seccion("📈 Serie Histórica + Predicción")
            fig = go.Figure()

            # Histórico
            fig.add_trace(go.Scatter(
                x=serie["ds"], y=serie["y"],
                name="Histórico",
                line=dict(color="#1A6B8A", width=2),
                mode="lines",
            ))

            # Predicción
            forecast_futuro = forecast[forecast["ds"] > serie["ds"].max()]
            fig.add_trace(go.Scatter(
                x=forecast_futuro["ds"], y=forecast_futuro["yhat"],
                name="Predicción",
                line=dict(color="#E8A838", width=2.5, dash="dash"),
                mode="lines",
            ))

            # Banda de confianza
            fig.add_trace(go.Scatter(
                x=pd.concat([forecast_futuro["ds"], forecast_futuro["ds"][::-1]]),
                y=pd.concat([forecast_futuro["yhat_upper"], forecast_futuro["yhat_lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(232,168,56,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="IC 90%",
            ))

            fig.add_vline(
                x=serie["ds"].max().timestamp() * 1000,
                line_width=1, line_dash="dot", line_color="#8AACBF",
                annotation_text="Hoy", annotation_position="top right",
            )

            st.plotly_chart(plotly_theme(fig, 450), use_container_width=True)

            # Tabla de predicciones
            seccion("📋 Próximas 12 Semanas en Detalle")
            tabla_pred = forecast_futuro[["ds", "yhat", "yhat_lower", "yhat_upper"]].head(12).copy()
            tabla_pred.columns = ["Semana", "Pax Estimados", "Mínimo", "Máximo"]
            tabla_pred["Semana"] = tabla_pred["Semana"].dt.strftime("%d %b %Y")
            tabla_pred[["Pax Estimados", "Mínimo", "Máximo"]] = tabla_pred[
                ["Pax Estimados", "Mínimo", "Máximo"]
            ].round(0).astype(int)
            st.dataframe(tabla_pred, use_container_width=True, hide_index=True)

            # Alerta de picos
            pico = tabla_pred.loc[tabla_pred["Pax Estimados"].idxmax()]
            st.success(f"🚀 **Pico proyectado:** Semana del {pico['Semana']} → {pico['Pax Estimados']:,} pasajeros estimados. ¡Coordiná recursos con anticipación!")

        except Exception as e:
            st.error(f"Error en el modelo: {e}")
            st.code("pip install prophet", language="bash")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: ESTACIONALIDAD
# ═══════════════════════════════════════════════════════════════════════════════

with tab_estacionalidad:
    st.markdown("## 📅 Análisis de Estacionalidad y Concentración de Demanda")

    datos_est = obtener_estacionalidad(df)

    col1, col2 = st.columns(2)

    with col1:
        seccion("📊 Demanda Promedio por Mes (histórico)")
        por_mes = df.groupby("Mes")["Q"].mean().reset_index()
        meses_labels = ["Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic"]
        por_mes["Mes_Nombre"] = por_mes["Mes"].apply(lambda m: meses_labels[m-1])
        fig = px.bar(
            por_mes, x="Mes_Nombre", y="Q",
            color="Q", color_continuous_scale=["#1A6B8A", "#E8A838"],
            text_auto=".0f",
            labels={"Q": "Pax Promedio Diario", "Mes_Nombre": ""},
        )
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(plotly_theme(fig, 350), use_container_width=True)

    with col2:
        seccion("📅 Demanda por Día de la Semana")
        por_dia = datos_est["por_dia"]
        dias_labels = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
        por_dia["Dia_Label"] = por_dia["DiaSemana"].apply(lambda d: dias_labels[d])
        fig = px.bar(
            por_dia, x="Dia_Label", y="Total_Pax",
            color="Total_Pax",
            color_continuous_scale=["#1A6B8A", "#C94040"],
            text_auto=".0f",
            labels={"Total_Pax": "Total Pasajeros", "Dia_Label": ""},
        )
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(plotly_theme(fig, 350), use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        seccion("☀️ Pasajeros por Temporada y Año")
        temp = datos_est["por_temporada"]
        fig = px.line(
            temp, x="Anio", y="Q", color="Temporada",
            markers=True, color_discrete_sequence=PALETTE,
            labels={"Q": "Pasajeros", "Anio": "Año"},
        )
        st.plotly_chart(plotly_theme(fig, 350), use_container_width=True)

    with col4:
        seccion("🏖️ Fin de Semana vs Días Hábiles")
        finde = datos_est["finde_vs_semana"].copy()
        finde["Tipo"] = finde["EsDiaEspecial"].map({0: "Días Hábiles", 1: "Finde / Feriado"})
        fig = px.bar(
            finde, x="Tipo", y="mean",
            color="Tipo", color_discrete_map={
                "Días Hábiles": "#1A6B8A",
                "Finde / Feriado": "#E8A838",
            },
            text_auto=".1f",
            labels={"mean": "Pax Promedio por Día", "Tipo": ""},
        )
        st.plotly_chart(plotly_theme(fig, 350), use_container_width=True)

    seccion("📆 Evolución Mensual Multi-Año (Comparativa)")
    pivot_line = df.pivot_table(values="Q", index="Mes", columns="Anio", aggfunc="sum")
    fig = go.Figure()
    for anio in pivot_line.columns:
        fig.add_trace(go.Scatter(
            x=[meses_labels[m-1] for m in pivot_line.index],
            y=pivot_line[anio].fillna(0),
            name=str(anio), mode="lines+markers",
            line=dict(width=1.5),
        ))
    st.plotly_chart(plotly_theme(fig, 420), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: MARKET BASKET ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_mba:
    st.markdown("## 🛒 Market Basket Analysis — Qué se vende junto")
    st.info("Basado en co-ocurrencia de actividades en la misma reserva/voucher. Ayuda a crear paquetes y up-sells estratégicos.")

    col_param1, col_param2, col_param3 = st.columns(3)
    with col_param1:
        min_sup = st.slider("Soporte mínimo %", 1, 20, 3) / 100
    with col_param2:
        min_conf = st.slider("Confianza mínima %", 20, 90, 40) / 100
    with col_param3:
        min_lift = st.slider("Lift mínimo", 1.0, 5.0, 1.2, step=0.1)

    with st.spinner("Calculando reglas..."):
        try:
            from ml_models import calcular_reglas_asociacion
            reglas = calcular_reglas_asociacion(df, min_sup, min_conf, min_lift)

            if reglas.empty:
                st.warning("⚠️ No se encontraron reglas con estos parámetros. Bajá el soporte mínimo.")
            else:
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1: st.metric("📏 Reglas Encontradas", len(reglas))
                with col_m2: st.metric("🎯 Confianza Máx.", f"{reglas['confianza_%'].max():.1f}%")
                with col_m3: st.metric("⚡ Lift Máximo", f"{reglas['lift'].max():.2f}x")

                seccion("🔝 Top Reglas por Lift (ordenadas)")
                st.dataframe(
                    reglas.head(20).style.background_gradient(
                        subset=["lift"], cmap="YlOrRd"
                    ),
                    use_container_width=True, hide_index=True,
                )

                seccion("📊 Visualización: Lift vs Confianza")
                fig = px.scatter(
                    reglas, x="confianza_%", y="lift",
                    size="soporte_%", color="lift",
                    hover_data=["antecedente", "consecuente"],
                    color_continuous_scale=["#1A6B8A", "#E8A838", "#C94040"],
                    labels={"confianza_%": "Confianza (%)", "lift": "Lift"},
                    text="consecuente",
                )
                fig.update_traces(textposition="top center", textfont_size=9)
                st.plotly_chart(plotly_theme(fig, 450), use_container_width=True)

                # Recomendaciones de paquetes
                seccion("💡 Paquetes Sugeridos Automáticamente")
                for _, row in reglas.head(5).iterrows():
                    st.markdown(f"""
                    <div style='background:#1C2B3A; border-left:3px solid #E8A838;
                                padding:12px 16px; margin:8px 0; border-radius:0 8px 8px 0;'>
                        <b style='color:#E8A838;'>Si compran:</b>
                        <span style='color:#E8EDF2;'> {row['antecedente']}</span>
                        &nbsp;&nbsp;→&nbsp;&nbsp;
                        <b style='color:#2E8B57;'>Ofrecer:</b>
                        <span style='color:#E8EDF2;'> {row['consecuente']}</span>
                        <span style='color:#8AACBF; float:right; font-size:0.8rem;'>
                            Confianza: {row['confianza_%']}% | Lift: {row['lift']}x
                        </span>
                    </div>
                    """, unsafe_allow_html=True)

        except ImportError:
            st.error("Instalá mlxtend: `pip install mlxtend`")
        except Exception as e:
            st.error(f"Error MBA: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: SEGMENTACIÓN RFM
# ═══════════════════════════════════════════════════════════════════════════════

with tab_rfm:
    st.markdown("## 👥 Segmentación de Clientes RFM")
    st.info("R=Recencia (días desde última visita) | F=Frecuencia (reservas) | M=Valor (pax acumulados)")

    with st.spinner("Segmentando..."):
        rfm = obtener_rfm(df)
        from ml_models import estrategia_por_segmento

        col1, col2 = st.columns(2)

        with col1:
            seccion("🥧 Distribución de Segmentos")
            dist = rfm["Segmento"].value_counts().reset_index()
            dist.columns = ["Segmento", "Clientes"]
            fig = px.pie(
                dist, values="Clientes", names="Segmento",
                hole=0.5, color_discrete_sequence=PALETTE,
            )
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(plotly_theme(fig, 380), use_container_width=True)

        with col2:
            seccion("📊 RFM Scatter: Frecuencia vs Valor")
            fig = px.scatter(
                rfm, x="Frecuencia", y="Valor",
                color="Segmento", size="Valor",
                hover_data=["NOMBRE", "Recencia", "Alojamiento_Frecuente"],
                color_discrete_sequence=PALETTE,
                labels={"Frecuencia": "Frecuencia (reservas)", "Valor": "Valor (pax total)"},
                log_x=True, log_y=True,
            )
            st.plotly_chart(plotly_theme(fig, 380), use_container_width=True)

        seccion("📈 Métricas Promedio por Segmento")
        resumen_rfm = (
            rfm.groupby("Segmento")
            .agg(
                Clientes=("ID_Cliente", "count"),
                Recencia_Media=("Recencia", "mean"),
                Frecuencia_Media=("Frecuencia", "mean"),
                Valor_Total=("Valor", "sum"),
                Valor_Medio=("Valor", "mean"),
            )
            .round(1)
            .reset_index()
            .sort_values("Valor_Total", ascending=False)
        )
        st.dataframe(resumen_rfm, use_container_width=True, hide_index=True)

        seccion("🎯 Estrategias de Marketing por Segmento")
        for segmento in rfm["Segmento"].unique():
            estrategia = estrategia_por_segmento(segmento)
            n_clientes = int((rfm["Segmento"] == segmento).sum())
            with st.expander(f"{segmento} — {n_clientes} clientes"):
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.markdown(f"**🎬 Acción**\n\n{estrategia.get('accion', '-')}")
                with col_b:
                    st.markdown(f"**📱 Canal**\n\n{estrategia.get('canal', '-')}")
                with col_c:
                    st.markdown(f"**💰 Oferta**\n\n{estrategia.get('oferta', '-')}")
                with col_d:
                    st.markdown(f"**📆 Frecuencia**\n\n{estrategia.get('frecuencia', '-')}")

        seccion("🔍 Explorador de Clientes por Segmento")
        seg_sel = st.selectbox("Filtrar por segmento:", rfm["Segmento"].unique())
        cols_show = ["NOMBRE", "Recencia", "Frecuencia", "Valor", "Alojamiento_Frecuente", "Actividades"]
        st.dataframe(
            rfm[rfm["Segmento"] == seg_sel][cols_show].sort_values("Valor", ascending=False),
            use_container_width=True, hide_index=True
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: B2B & OPERADORES
# ═══════════════════════════════════════════════════════════════════════════════

with tab_b2b:
    st.markdown("## 🤝 Análisis B2B — Alojamientos y Operadores")

    col1, col2 = st.columns(2)

    with col1:
        seccion("🏨 Ranking de Alojamientos (por pasajeros derivados)")
        hoteles = obtener_ranking_hoteles(df)
        if not hoteles.empty:
            fig = px.bar(
                hoteles.head(15), x="Total_Pax", y="ALOJAMIENTO",
                orientation="h",
                color="Clientes_Unicos",
                color_continuous_scale=["#1A6B8A", "#E8A838"],
                text="Total_Pax",
                labels={"Total_Pax": "Total Pasajeros", "ALOJAMIENTO": ""},
                hover_data=["Clientes_Unicos", "Actividades_Frecuentes"],
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(plotly_theme(fig, 450), use_container_width=True)
            st.dataframe(hoteles, use_container_width=True, hide_index=True)
        else:
            st.warning("Sin datos de alojamiento disponibles.")

    with col2:
        seccion("📊 Rendimiento por Operador / Canal de Venta")
        ops = obtener_rendimiento_operadores(df)
        if not ops.empty:
            fig = px.scatter(
                ops, x="Ticket_Promedio", y="Total_Pax",
                size="Clientes_Unicos", color="Ticket_Promedio",
                text="OPERADOR",
                color_continuous_scale=["#1A6B8A", "#E8A838"],
                labels={"Ticket_Promedio": "Ticket Promedio (Pax)", "Total_Pax": "Volumen Total"},
                hover_data=["Total_Reservas"],
            )
            fig.update_traces(textposition="top center", textfont_size=9)
            fig.update_coloraxes(showscale=False)
            st.plotly_chart(plotly_theme(fig, 380), use_container_width=True)
            st.dataframe(ops, use_container_width=True, hide_index=True)
        else:
            st.warning("Sin datos de operador disponibles.")

    seccion("📈 Tendencia de Alojamientos Top 5 en el Tiempo")
    top5_hoteles = hoteles.head(5)["ALOJAMIENTO"].tolist() if not hoteles.empty else []
    if top5_hoteles:
        df_hotel_trend = (
            df[df["ALOJAMIENTO"].isin(top5_hoteles)]
            .groupby(["Anio", "ALOJAMIENTO"])["Q"]
            .sum().reset_index()
        )
        fig = px.line(
            df_hotel_trend, x="Anio", y="Q", color="ALOJAMIENTO",
            markers=True, color_discrete_sequence=PALETTE,
            labels={"Q": "Pasajeros", "Anio": "Año"},
        )
        st.plotly_chart(plotly_theme(fig, 350), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7: DATOS CRUDOS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_datos:
    st.markdown("## 📋 Explorador de Datos Procesados")

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        fecha_min = st.date_input("Desde", value=df["Fecha"].min().date())
    with col_f2:
        fecha_max = st.date_input("Hasta", value=df["Fecha"].max().date())
    with col_f3:
        buscar = st.text_input("🔍 Buscar (nombre, actividad...)")

    df_vis = df[
        (df["Fecha"].dt.date >= fecha_min) &
        (df["Fecha"].dt.date <= fecha_max)
    ]
    if buscar:
        mask = df_vis.astype(str).apply(
            lambda col: col.str.contains(buscar, case=False, na=False)
        ).any(axis=1)
        df_vis = df_vis[mask]

    st.caption(f"Mostrando {len(df_vis):,} registros")

    cols_vis = ["Fecha", "NOMBRE", "Q", "ACTIVIDAD", "Categoria_Actividad",
                "ALOJAMIENTO", "OPERADOR", "VOUCHER", "Es_Recurrente",
                "Segmento" if "Segmento" not in df.columns else "Temporada"]
    cols_vis = [c for c in cols_vis if c in df_vis.columns]

    st.dataframe(df_vis[cols_vis].head(500), use_container_width=True, hide_index=True)

    csv = df_vis.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        "⬇️ Descargar CSV filtrado",
        data=csv, file_name="datos_filtrados.csv", mime="text/csv"
    )
