"""
DIAGNÓSTICO — Corré esto primero para ver cómo lee Python tu Excel.
Uso: python diagnostico.py tu_archivo.xlsx
"""
import sys
import pandas as pd
import re

ruta = sys.argv[1] if len(sys.argv) > 1 else "datos_turismo.xlsx"

print("=" * 70)
print("DIAGNÓSTICO DE ESTRUCTURA DEL EXCEL")
print("=" * 70)

# Leer todas las hojas disponibles
xl = pd.ExcelFile(ruta)
print(f"\n📋 Hojas encontradas: {xl.sheet_names}")
print(f"   Analizando hoja: '{xl.sheet_names[0]}'")

df = pd.read_excel(ruta, sheet_name=0, header=None, dtype=str)
print(f"\n📐 Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")

print("\n" + "─" * 70)
print("PRIMERAS 60 FILAS BRUTAS (para ver el patrón):")
print("─" * 70)
for i in range(min(60, len(df))):
    fila = df.iloc[i].tolist()
    no_nulos = [str(v) for v in fila if pd.notna(v) and str(v).strip() not in ("", "nan", "None")]
    if no_nulos:
        print(f"Fila {i:03d}: {no_nulos}")

print("\n" + "─" * 70)
print("BUSCANDO TODAS LAS FILAS QUE PARECEN FECHAS:")
print("─" * 70)

MESES = ["enero","febrero","marzo","abril","mayo","junio",
         "julio","agosto","septiembre","setiembre","octubre","noviembre","diciembre"]

fechas_encontradas = []
for i in range(len(df)):
    fila = df.iloc[i].tolist()
    primera = next((str(v) for v in fila if pd.notna(v) and str(v).strip() not in ("","nan")), "")
    primera_low = primera.lower()
    tiene_mes = any(m in primera_low for m in MESES)
    tiene_anio = bool(re.search(r"\b20[012]\d\b|\b201[0-9]\b", primera_low))
    tiene_num = bool(re.search(r"\b\d{1,2}\b", primera_low))
    if tiene_mes or (tiene_anio and tiene_num):
        fechas_encontradas.append((i, primera[:80]))
        if len(fechas_encontradas) <= 30:
            print(f"  Fila {i:04d}: '{primera[:80]}'")

print(f"\n  → Total filas de fecha detectadas: {len(fechas_encontradas)}")
if fechas_encontradas:
    print(f"  → Primera fecha: fila {fechas_encontradas[0][0]}  → '{fechas_encontradas[0][1]}'")
    print(f"  → Última fecha:  fila {fechas_encontradas[-1][0]} → '{fechas_encontradas[-1][1]}'")

print("\n" + "─" * 70)
print("VALORES ÚNICOS EN CADA COLUMNA (primeras 8):")
print("─" * 70)
for col_idx in range(min(8, df.shape[1])):
    unicos = df.iloc[:, col_idx].dropna().unique()[:5]
    print(f"  Col {col_idx}: {list(unicos)}")

print("\n" + "─" * 70)
print("FILAS QUE PARECEN ENCABEZADOS (contienen 'NOMBRE' o 'ACTIVIDAD'):")
print("─" * 70)
headers_encontrados = []
for i in range(len(df)):
    fila_str = " ".join(str(v) for v in df.iloc[i].tolist() if pd.notna(v)).upper()
    if "NOMBRE" in fila_str or "ACTIVIDAD" in fila_str:
        headers_encontrados.append(i)
        if len(headers_encontrados) <= 5:
            print(f"  Fila {i:04d}: {df.iloc[i].dropna().tolist()[:6]}")

print(f"\n  → Total filas de header encontradas: {len(headers_encontrados)}")
print(f"  → Frecuencia aproximada: cada {round(len(df)/max(len(headers_encontrados),1))} filas")

print("\n✅ Diagnóstico completo. Compartí este output para ajustar el parser.")
