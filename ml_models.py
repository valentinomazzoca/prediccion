"""
================================================================================
  MÓDULO 2: MODELOS DE MACHINE LEARNING
  ├── Forecasting de Demanda (Prophet + ARIMA fallback)
  ├── Market Basket Analysis (Apriori / MLxtend)
  └── Segmentación RFM (K-Means con análisis de silhouette)
================================================================================
"""

import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# MODELO 1: FORECASTING DE DEMANDA
# ─────────────────────────────────────────────────────────────────────────────

def preparar_serie_temporal(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """
    Agrega el DataFrame por semana (o mes) sumando pasajeros (Q).
    Retorna df con columnas 'ds' y 'y' (formato Prophet).
    freq: 'W' = semanal, 'M' = mensual
    """
    serie = (
        df.groupby(pd.Grouper(key="Fecha", freq=freq))["Q"]
        .sum()
        .reset_index()
        .rename(columns={"Fecha": "ds", "Q": "y"})
    )
    serie = serie[serie["y"] > 0]  # eliminar períodos sin actividad
    return serie


def forecast_prophet(df: pd.DataFrame, periodos: int = 26, freq: str = "W") -> tuple:
    """
    Forecasting con Prophet (26 semanas ≈ 6 meses por defecto).
    Retorna: (modelo, df_forecast, df_componentes)
    """
    try:
        from prophet import Prophet
    except ImportError:
        print("⚠️  Prophet no instalado. Usando ARIMA como fallback.")
        return forecast_arima(df, periodos, freq)

    serie = preparar_serie_temporal(df, freq)

    modelo = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=(freq == "W"),
        daily_seasonality=False,
        seasonality_mode="multiplicative",  # turismo tiene picos multiplicativos
        changepoint_prior_scale=0.15,        # flexible ante cambios de tendencia
        seasonality_prior_scale=10.0,
        interval_width=0.90,
    )

    # Agregar estacionalidad del hemisferio sur (diciembre = verano = pico)
    modelo.add_seasonality(
        name="temporada_sur",
        period=365.25 / 2,
        fourier_order=5,
    )

    modelo.fit(serie)

    futuro = modelo.make_future_dataframe(periods=periodos, freq=freq)
    forecast = modelo.predict(futuro)

    # Asegurarse de que no haya predicciones negativas
    forecast["yhat"] = forecast["yhat"].clip(lower=0)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)

    return modelo, forecast, serie


def forecast_arima(df: pd.DataFrame, periodos: int = 6, freq: str = "M") -> tuple:
    """
    Fallback con ARIMA/SARIMA si Prophet no está disponible.
    Usa pmdarima (auto_arima) para selección automática de parámetros.
    """
    try:
        import pmdarima as pm
    except ImportError:
        raise ImportError("Instalá prophet o pmdarima: pip install prophet pmdarima")

    serie = preparar_serie_temporal(df, freq="M")
    serie = serie.set_index("ds")["y"]

    modelo = pm.auto_arima(
        serie,
        seasonal=True, m=12,
        stepwise=True, information_criterion="aic",
        suppress_warnings=True,
    )

    forecast_vals, conf_int = modelo.predict(n_periods=periodos, return_conf_int=True)
    fechas_futuras = pd.date_range(
        start=serie.index[-1] + pd.DateOffset(months=1),
        periods=periodos, freq="MS"
    )

    forecast_df = pd.DataFrame({
        "ds": fechas_futuras,
        "yhat": forecast_vals.clip(0),
        "yhat_lower": conf_int[:, 0].clip(0),
        "yhat_upper": conf_int[:, 1].clip(0),
    })

    historico = serie.reset_index().rename(columns={"ds": "ds", "y": "yhat"})
    historico["yhat_lower"] = historico["yhat"]
    historico["yhat_upper"] = historico["yhat"]

    return modelo, pd.concat([historico, forecast_df]), serie.reset_index().rename(
        columns={"ds": "ds", "y": "y"}
    )


# ─────────────────────────────────────────────────────────────────────────────
# MODELO 2: MARKET BASKET ANALYSIS (Apriori)
# ─────────────────────────────────────────────────────────────────────────────

def construir_matriz_canastas(df: pd.DataFrame, nivel: str = "VOUCHER") -> pd.DataFrame:
    """
    Construye la matriz binaria de co-ocurrencia de actividades.
    Agrupa por Fecha+Alojamiento (proxy de mismo grupo/transacción).
    Usa Categoria_Actividad para mejor soporte estadístico.
    """
    df = df.copy()

    # Filtrar actividades vacías
    df_mba = df[df["ACTIVIDAD"].str.strip().str.len() >= 3].copy()

    # VOUCHER tiene muchos vacíos → usar Fecha+Alojamiento como transacción
    voucher_ok = (
        nivel in df_mba.columns
        and df_mba[nivel].str.strip().replace("", np.nan).notna().sum() > len(df_mba) * 0.3
    )
    if voucher_ok:
        df_mba["_txn"] = df_mba[nivel].str.strip()
    else:
        aloj = df_mba["ALOJAMIENTO"].str.strip().replace("", "SIN_ALOJ")
        df_mba["_txn"] = df_mba["Fecha"].dt.date.astype(str) + "__" + aloj

    # Usar categorías (no nombres exactos) → mejora soporte
    df_mba["_item"] = df_mba["Categoria_Actividad"]

    # Pivot y convertir a bool explícito para mlxtend
    canastas = (
        df_mba.groupby(["_txn", "_item"])
        .size()
        .unstack(fill_value=0)
        .gt(0)
    )
    # mlxtend exige exactamente dtype bool, no object
    canastas = canastas.astype(bool)

    return canastas


def calcular_reglas_asociacion(
    df: pd.DataFrame,
    min_support: float = 0.02,
    min_confidence: float = 0.40,
    min_lift: float = 1.2,
) -> pd.DataFrame:
    """
    Aplica Apriori y genera reglas de asociación.
    Retorna DataFrame con: antecedente → consecuente, support, confidence, lift.
    """
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
    except ImportError:
        raise ImportError("Instalá mlxtend: pip install mlxtend")

    canastas = construir_matriz_canastas(df)

    if len(canastas) < 10:
        print("⚠️  Pocas transacciones para MBA. Se necesitan al menos 10.")
        return pd.DataFrame()

    itemsets = apriori(canastas, min_support=min_support, use_colnames=True)

    if itemsets.empty:
        # Relajar el soporte si no hay itemsets
        itemsets = apriori(canastas, min_support=min_support / 2, use_colnames=True)

    if itemsets.empty:
        return pd.DataFrame()

    reglas = association_rules(
        itemsets, metric="lift", min_threshold=min_lift
    )
    reglas = reglas[reglas["confidence"] >= min_confidence]
    reglas = reglas.sort_values("lift", ascending=False)

    # Formatear para legibilidad
    reglas["antecedente"] = reglas["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    reglas["consecuente"] = reglas["consequents"].apply(lambda x: ", ".join(sorted(x)))
    reglas["soporte_%"] = (reglas["support"] * 100).round(1)
    reglas["confianza_%"] = (reglas["confidence"] * 100).round(1)
    reglas["lift"] = reglas["lift"].round(2)

    return reglas[["antecedente", "consecuente", "soporte_%", "confianza_%", "lift"]]


# ─────────────────────────────────────────────────────────────────────────────
# MODELO 3: SEGMENTACIÓN RFM
# ─────────────────────────────────────────────────────────────────────────────

def calcular_rfm(df: pd.DataFrame, fecha_referencia: datetime = None) -> pd.DataFrame:
    """
    Calcula las métricas RFM por cliente:
    - Recencia   (R): días desde su última visita
    - Frecuencia (F): número de reservas
    - Valor      (M): total de pasajeros (proxy monetario)
    """
    if fecha_referencia is None:
        fecha_referencia = df["Fecha"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("ID_Cliente")
        .agg(
            Ultima_Visita=("Fecha", "max"),
            Frecuencia=("Fecha", "count"),
            Valor=("Q", "sum"),
            NOMBRE=("NOMBRE", "first"),
            Actividades=("ACTIVIDAD", lambda x: ", ".join(x.dropna().unique()[:3])),
            Alojamiento_Frecuente=("ALOJAMIENTO", lambda x: x.mode()[0] if len(x) > 0 else ""),
        )
        .reset_index()
    )

    rfm["Recencia"] = (fecha_referencia - rfm["Ultima_Visita"]).dt.days

    return rfm


def segmentar_clientes_rfm(
    rfm: pd.DataFrame, n_clusters: int = None
) -> pd.DataFrame:
    """
    K-Means sobre las métricas RFM normalizadas.
    Determina automáticamente el número óptimo de clusters (3-6)
    usando el coeficiente de silhouette.
    """
    features = rfm[["Recencia", "Frecuencia", "Valor"]].copy()

    # Transformación logarítmica para reducir skewness
    features["Recencia"] = np.log1p(features["Recencia"])
    features["Frecuencia"] = np.log1p(features["Frecuencia"])
    features["Valor"] = np.log1p(features["Valor"])

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    if n_clusters is None:
        # Búsqueda automática del K óptimo
        scores = {}
        rango = range(3, min(7, len(rfm) // 5 + 1))
        for k in rango:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            if len(set(labels)) > 1:
                scores[k] = silhouette_score(X, labels)
        n_clusters = max(scores, key=scores.get) if scores else 4
        print(f"   → K óptimo: {n_clusters} clusters (silhouette scores: {scores})")

    km_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["Cluster"] = km_final.fit_predict(X)

    # Etiquetas interpretables: ordenar clusters por valor/frecuencia
    resumen = rfm.groupby("Cluster").agg(
        R_media=("Recencia", "mean"),
        F_media=("Frecuencia", "mean"),
        M_media=("Valor", "mean"),
    )
    # Score compuesto: bajo R + alto F + alto M = mejor cliente
    resumen["Score"] = (
        -resumen["R_media"] / resumen["R_media"].max()
        + resumen["F_media"] / resumen["F_media"].max()
        + resumen["M_media"] / resumen["M_media"].max()
    )
    orden = resumen["Score"].rank(ascending=False).astype(int)

    etiquetas = {
        1: "🌟 Champions",
        2: "💎 Leales",
        3: "🔄 En Riesgo",
        4: "💤 Hibernando",
        5: "🆕 Nuevos",
        6: "❓ Sin Clasificar",
    }

    rfm["Segmento"] = rfm["Cluster"].map(
        lambda c: etiquetas.get(orden.get(c, 6), "❓ Sin Clasificar")
    )

    print(f"\n📊 Distribución de segmentos RFM:")
    print(rfm["Segmento"].value_counts().to_string())

    return rfm


def estrategia_por_segmento(segmento: str) -> dict:
    """Retorna la estrategia de marketing recomendada por segmento."""
    estrategias = {
        "🌟 Champions": {
            "accion": "Programa VIP / Embajadores",
            "canal": "WhatsApp personalizado / Email exclusivo",
            "oferta": "Early access a nuevas excursiones, descuentos > 15%",
            "frecuencia": "Contacto mensual",
        },
        "💎 Leales": {
            "accion": "Programa de fidelización",
            "canal": "Email + Redes sociales",
            "oferta": "Puntos acumulables, paquetes combinados",
            "frecuencia": "Contacto cada 45 días",
        },
        "🔄 En Riesgo": {
            "accion": "Campaña de reactivación urgente",
            "canal": "WhatsApp + Email con asunto llamativo",
            "oferta": "Descuento 10-15% por tiempo limitado, 'Te extrañamos'",
            "frecuencia": "Contacto inmediato + 1 seguimiento",
        },
        "💤 Hibernando": {
            "accion": "Campaña de win-back estacional",
            "canal": "Email masivo + Retargeting digital",
            "oferta": "Novedad de temporada + beneficio especial",
            "frecuencia": "1 vez por temporada",
        },
        "🆕 Nuevos": {
            "accion": "Onboarding y educación de producto",
            "canal": "Email secuencia bienvenida + WhatsApp",
            "oferta": "Segunda excursión con descuento, bundle starter",
            "frecuencia": "3 contactos en los primeros 30 días",
        },
    }
    return estrategias.get(segmento, {"accion": "Análisis adicional requerido"})


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS DE COMPORTAMIENTO (Marketing & Ventas)
# ─────────────────────────────────────────────────────────────────────────────

def analisis_estacionalidad(df: pd.DataFrame) -> dict:
    """
    Devuelve tablas de concentración de demanda por:
    - Mes, Semana del año, Día de la semana, Temporada
    """
    return {
        "por_mes": (
            df.groupby(["Anio", "Mes", "Mes_Nombre"])["Q"]
            .sum().reset_index()
            .sort_values(["Anio", "Mes"])
        ),
        "por_semana": (
            df.groupby("Semana")["Q"]
            .agg(["sum", "mean", "count"])
            .rename(columns={"sum": "Total_Pax", "mean": "Prom_Pax", "count": "Reservas"})
            .reset_index()
        ),
        "por_dia": (
            df.groupby(["DiaSemana", "DiaSemana_Nombre"])["Q"]
            .agg(["sum", "mean"])
            .rename(columns={"sum": "Total_Pax", "mean": "Prom_Pax"})
            .reset_index()
            .sort_values("DiaSemana")
        ),
        "por_temporada": (
            df.groupby(["Temporada", "Anio"])["Q"]
            .sum().reset_index()
        ),
        "finde_vs_semana": (
            df.groupby("EsDiaEspecial")["Q"]
            .agg(["sum", "mean", "count"])
            .reset_index()
        ),
    }


def ranking_alojamientos(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Ranking de alojamientos que más clientes derivan."""
    df_hotel = df[df["ALOJAMIENTO"].str.strip() != ""].copy()
    ranking = (
        df_hotel.groupby("ALOJAMIENTO")
        .agg(
            Total_Pax=("Q", "sum"),
            Total_Reservas=("Q", "count"),
            Clientes_Unicos=("ID_Cliente", "nunique"),
            Pax_Promedio=("Q", "mean"),
            Actividades_Frecuentes=("ACTIVIDAD", lambda x: x.mode()[0] if len(x) > 0 else ""),
        )
        .reset_index()
        .sort_values("Total_Pax", ascending=False)
        .head(top_n)
    )
    ranking["Pax_Promedio"] = ranking["Pax_Promedio"].round(2)
    ranking.insert(0, "Ranking", range(1, len(ranking) + 1))
    return ranking


def rendimiento_operadores(df: pd.DataFrame) -> pd.DataFrame:
    """Análisis de rendimiento por operador/canal de venta."""
    col = "OPERADOR" if "OPERADOR" in df.columns else None
    if col is None:
        return pd.DataFrame()

    return (
        df[df[col].str.strip() != ""]
        .groupby(col)
        .agg(
            Total_Pax=("Q", "sum"),
            Total_Reservas=("Q", "count"),
            Ticket_Promedio=("Q", "mean"),
            Clientes_Unicos=("ID_Cliente", "nunique"),
            Actividad_Top=("ACTIVIDAD", lambda x: x.mode()[0] if len(x) > 0 else ""),
        )
        .reset_index()
        .sort_values("Total_Pax", ascending=False)
        .assign(Ticket_Promedio=lambda x: x["Ticket_Promedio"].round(2))
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    parquet = sys.argv[1] if len(sys.argv) > 1 else "output/datos_procesados.parquet"
    df = pd.read_parquet(parquet)
    df["Fecha"] = pd.to_datetime(df["Fecha"])

    print("\n[1] FORECASTING")
    _, forecast, serie = forecast_prophet(df, periodos=26)
    print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10))

    print("\n[2] MARKET BASKET ANALYSIS")
    reglas = calcular_reglas_asociacion(df)
    if not reglas.empty:
        print(reglas.head(10).to_string(index=False))

    print("\n[3] SEGMENTACIÓN RFM")
    rfm = calcular_rfm(df)
    rfm = segmentar_clientes_rfm(rfm)
    print(rfm[["NOMBRE", "Recencia", "Frecuencia", "Valor", "Segmento"]].head(20))
