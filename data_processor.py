"""
================================================================================
  MÓDULO 1: INGENIERÍA DE DATOS — Parseo, Limpieza y Enriquecimiento
  Basado en la estructura REAL del archivo RESERVAS_RAM.xlsx:

  - 130+ hojas (una por mes), desde ENERO 2016 hasta 2026
  - Cada hoja: fecha (datetime en col 0) → header → filas de datos → repite
  - Layout de columnas varía entre años (con/sin TEL, con/sin TERNSF)
================================================================================
"""

import re
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HOJAS_SKIP = {
    "Hoja2","Hoja27","Hoja5","Hoja1","Hoja24","Hoja28","Hoja25","Hoja23",
    "Hoja3","Hoja26","Hoja 28","OMV ACTIVIDADES AVENTURA 2019-2",
    "SETIEMBRE 19","OCTUBRE 19","NOVIEMBRE 19",
    "RESERVAS BKP 10SEP AL 22 OCT","HYT PLANILLA 2019-2023",
    "PLANILLA DE TRANSFER",
}

FERIADOS_FIJOS = {
    (1,1),(3,24),(4,2),(5,1),(5,25),(6,20),(7,9),
    (8,17),(10,12),(11,20),(12,8),(12,25),
}

CATEGORIAS = {
    "Aventura": ["raft","rafting","rapel","rappel","trek","trekking","canopy",
                 "tirolesa","kayak","escalada","bici","cabalgata","via ferrata",
                 "arborismo","cueva","aventura","jeep","4x4","quad","enduro",
                 "afo","aforadora"],
    "Nieve & Ski": ["ski","snowboard","nieve","telesilla","pista","penitentes",
                    "puquios","vallecitos","dia de nieve"],
    "Gastronomía & Bodegas": ["cena","almuerzo","degustacion","bodega","vino",
                               "wine","cata","restaurante","picada","maridaje","gastro"],
    "Relax & Wellness": ["spa","termas","masaje","yoga","meditacion","wellness","relax"],
    "Cultural & City Tour": ["city","tour","museo","ciudad","historico","visita",
                              "plaza","catedral","parque","centro","cultural","luna llena"],
    "Transfer & Logística": ["transfer","traslado","aeropuerto","terminal","remis",
                              "micro","bus","pick up"],
}

HEADER_WORDS = {"TERNSF","NOMBRE","ACTIVIDAD","Q","OBSERV","ALOJAMIENTO",
                "VOUCHER","OPERADOR","HAB","OBS","TEL","NOMBRE Y APELLIDO"}

COL_ALIAS = {
    "TERNSF":"TERNSF","Q":"Q","NOMBRE":"NOMBRE","NOMBRE Y APELLIDO":"NOMBRE",
    "ACTIVIDAD":"ACTIVIDAD","OBSERV":"OBSERV","OBS":"OBS",
    "ALOJAMIENTO":"ALOJAMIENTO","TEL":"TEL","TELEFONO":"TEL","CONTACTO":"TEL",
    "HAB":"HAB","OPERADOR":"OPERADOR","VOUCHER":"VOUCHER",
}


def normalizar(txt) -> str:
    if not isinstance(txt, str) or not txt.strip():
        return ""
    txt = unicodedata.normalize("NFD", txt.lower().strip())
    return re.sub(r"\s+"," ","".join(c for c in txt if unicodedata.category(c)!="Mn")).strip()

def limpiar_nombre(val) -> str:
    if pd.isna(val) or str(val).strip() in ("","nan","None"): return ""
    return re.sub(r"\s+"," ", str(val).upper().strip())

def limpiar_cantidad(val) -> int:
    """Q: solo acepta 1-200. Filtra precios, fechas y basura."""
    if pd.isna(val) or str(val).strip() in ("","nan","None"): return 1
    raw = str(val).strip()
    # Si contiene letras no es una cantidad
    if re.search(r"[a-zA-Z$]", raw): return 1
    num_str = re.sub(r"[^\d.]","", raw.replace(",","."))
    if not num_str: return 1
    try:
        n = int(float(num_str))
        # Fechas disfrazadas (ej: 20220208000000) o precios altos
        if n > 500: return 1
        return max(1, n)
    except: return 1

def limpiar_telefono(val) -> str:
    if pd.isna(val) or not str(val).strip(): return ""
    return re.sub(r"[^\d]","",str(val))[-12:]

def limpiar_texto(val) -> str:
    if pd.isna(val) or str(val).strip() in ("","nan","None"): return ""
    return str(val).strip()

def categorizar(actividad: str) -> str:
    norm = normalizar(actividad)
    if not norm: return "Sin Categoría"
    for cat, kws in CATEGORIAS.items():
        if any(kw in norm for kw in kws): return cat
    return "Otros"

def crear_id_cliente(nombre: str, tel: str) -> str:
    n = normalizar(nombre)[:25].replace(" ","_")
    t = tel[-8:] if len(tel) >= 8 else tel
    return f"{n}__{t}" if t else n

def es_feriado(fecha) -> bool:
    try: return (fecha.month, fecha.day) in FERIADOS_FIJOS
    except: return False

def calcular_temporada(mes: int) -> str:
    if mes in (12,1,2): return "Alta Verano"
    if mes in (3,4):    return "Hombro Otoño"
    if mes in (5,6):    return "Baja"
    if mes in (7,8):    return "Alta Invierno"
    if mes in (9,10):   return "Primavera"
    if mes == 11:       return "Pre-Temporada"
    return "Sin Temporada"


def es_fila_header(fila) -> bool:
    vals = {str(v).strip().upper() for v in fila if pd.notna(v) and str(v).strip() not in ("","nan")}
    return len(vals & HEADER_WORDS) >= 3

def es_fila_fecha(fila) -> object:
    """Retorna Timestamp si la primera celda no nula es una fecha, sino None."""
    for v in fila:
        if pd.isna(v) or str(v).strip() in ("","nan"):
            continue
        # pandas con dtype=str lee fechas excel como "2016-01-01 00:00:00"
        ts = pd.to_datetime(v, errors="coerce")
        if pd.notna(ts) and 2010 <= ts.year <= 2030:
            return ts
        # Si el primer valor no es fecha, no es fila de fecha
        return None
    return None

def construir_col_map(fila) -> dict:
    col_map = {}
    for idx, val in enumerate(fila):
        key = str(val).strip().upper()
        interno = COL_ALIAS.get(key)
        if interno and interno not in col_map:
            col_map[interno] = idx
    return col_map

def parsear_hoja(df_raw: pd.DataFrame, nombre_hoja: str) -> list:
    registros = []
    fecha_actual = None
    col_map = {}

    for _, fila in df_raw.iterrows():
        fila_list = fila.tolist()

        ts = es_fila_fecha(fila_list)
        if ts is not None:
            fecha_actual = ts
            continue

        if es_fila_header(fila_list):
            col_map = construir_col_map(fila_list)
            continue

        if not [v for v in fila_list if pd.notna(v) and str(v).strip() not in ("","nan")]:
            continue
        if fecha_actual is None or not col_map:
            continue

        def get(col):
            idx = col_map.get(col)
            if idx is None: return np.nan
            try:
                v = fila_list[idx]
                return v if pd.notna(v) and str(v).strip() not in ("","nan") else np.nan
            except: return np.nan

        nombre = limpiar_nombre(get("NOMBRE"))
        actividad = limpiar_texto(get("ACTIVIDAD")).upper()
        if not nombre and not actividad: continue

        registros.append({
            "Fecha": fecha_actual,
            "Hoja": nombre_hoja,
            "TERNSF": limpiar_texto(get("TERNSF")).upper(),
            "Q": limpiar_cantidad(get("Q")),
            "NOMBRE": nombre,
            "ACTIVIDAD": actividad,
            "OBSERV": limpiar_texto(get("OBSERV")),
            "ALOJAMIENTO": limpiar_texto(get("ALOJAMIENTO")).upper(),
            "TEL": limpiar_telefono(get("TEL")),
            "HAB": limpiar_texto(get("HAB")),
            "OPERADOR": limpiar_texto(get("OPERADOR")).upper(),
            "VOUCHER": limpiar_texto(get("VOUCHER")),
            "OBS": limpiar_texto(get("OBS")).upper(),
        })
    return registros


def procesar_excel(ruta_excel: str) -> pd.DataFrame:
    print("\n" + "="*60)
    print("  TOURISM INTELLIGENCE — PIPELINE DE PROCESAMIENTO")
    print("="*60)
    print(f"📂 {ruta_excel}")

    xls = pd.ExcelFile(ruta_excel, engine="openpyxl")
    hojas_validas = [h for h in xls.sheet_names if h not in HOJAS_SKIP]
    print(f"📋 Hojas a procesar: {len(hojas_validas)} / {len(xls.sheet_names)}")

    todos_registros = []
    for i, hoja in enumerate(hojas_validas):
        try:
            df_raw = pd.read_excel(ruta_excel, sheet_name=hoja,
                                   header=None, dtype=str, engine="openpyxl")
            registros = parsear_hoja(df_raw, hoja)
            todos_registros.extend(registros)
            if (i+1) % 25 == 0 or i == len(hojas_validas)-1:
                print(f"   ✅ {i+1}/{len(hojas_validas)} hojas | {len(todos_registros):,} registros")
        except Exception as e:
            print(f"   ⚠️  '{hoja}': {e}")

    if not todos_registros:
        raise ValueError("❌ No se extrajeron registros.")

    df = pd.DataFrame(todos_registros)

    # Temporal
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    df = df.dropna(subset=["Fecha"])
    df = df[df["Fecha"].dt.year >= 2015]
    df["Anio"]            = df["Fecha"].dt.year
    df["Mes"]             = df["Fecha"].dt.month
    df["Mes_Nombre"]      = df["Fecha"].dt.strftime("%B")
    df["Semana"]          = df["Fecha"].dt.isocalendar().week.astype(int)
    df["DiaSemana"]       = df["Fecha"].dt.dayofweek
    df["DiaSemana_Nombre"]= df["Fecha"].dt.strftime("%A")
    df["EsFinDeSemana"]   = df["DiaSemana"].isin([5,6]).astype(int)
    df["EsFeriado"]       = df["Fecha"].apply(es_feriado).astype(int)
    df["EsDiaEspecial"]   = ((df["EsFinDeSemana"]==1)|(df["EsFeriado"]==1)).astype(int)
    df["Trimestre"]       = df["Fecha"].dt.quarter
    df["Temporada"]       = df["Mes"].apply(calcular_temporada)

    # ID cliente y recurrencia
    df["ID_Cliente"] = df.apply(lambda r: crear_id_cliente(r["NOMBRE"],r["TEL"]), axis=1)
    visitas = df[df["ID_Cliente"].str.len()>2].groupby("ID_Cliente")["Fecha"].agg(
        Primera_Visita="min", Ultima_Visita="max", Total_Visitas="count"
    )
    visitas["Es_Recurrente"] = (visitas["Total_Visitas"]>1).astype(int)
    df = df.merge(visitas[["Es_Recurrente","Total_Visitas"]], on="ID_Cliente", how="left")
    df["Es_Recurrente"] = df["Es_Recurrente"].fillna(0).astype(int)
    df["Total_Visitas"]  = df["Total_Visitas"].fillna(1).astype(int)

    # Categorías
    df["Categoria_Actividad"] = df["ACTIVIDAD"].apply(categorizar)

    print(f"\n📊 RESUMEN:")
    print(f"   Registros  : {len(df):,}")
    print(f"   Rango      : {df['Fecha'].min().date()} → {df['Fecha'].max().date()}")
    print(f"   Clientes   : {df['ID_Cliente'].nunique():,}")
    print(f"   Pasajeros  : {df['Q'].sum():,}")
    for cat, n in df["Categoria_Actividad"].value_counts().items():
        print(f"   {cat}: {n:,}")
    return df


def guardar_datos_procesados(df, directorio="."):
    Path(directorio).mkdir(parents=True, exist_ok=True)
    c = Path(directorio)/"datos_procesados.csv"
    df.to_csv(c, index=False, encoding="utf-8-sig")
    print(f"💾 {c}")
    return {"csv":str(c)}


if __name__ == "__main__":
    import sys
    ruta = sys.argv[1] if len(sys.argv)>1 else "RESERVAS_RAM.xlsx"
    df = procesar_excel(ruta)
    guardar_datos_procesados(df, "output")
