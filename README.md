# 🏔️ Tourism Intelligence Hub
### Sistema de Predicción de Comportamiento y Segmentación de Clientes

---

## 📁 Estructura del Proyecto

```
turismo_ml/
├── data_processor.py    # Módulo 1: Ingeniería de Datos
├── ml_models.py         # Módulo 2: Modelos de ML
├── app.py               # Módulo 3: Dashboard Streamlit
├── requirements.txt     # Dependencias
└── README.md
```

---

## ⚡ Instalación y Ejecución (3 pasos)

```bash
# 1. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Lanzar el dashboard
streamlit run app.py
```

Luego abrí tu navegador en: **http://localhost:8501**

---

## 📊 Formato del Excel Soportado

El sistema reconoce automáticamente el formato:

```
viernes, 1 de enero de 2016          ← Fila de fecha (puede ser celda fusionada)
TERNSF | Q | NOMBRE | ACTIVIDAD | OBSERV | ALOJAMIENTO | HAB | OPERADOR | VOUCHER | OBS
  ...  | 2 | García Juan | Trekking | ... | Hotel Andino | 102 | AGENCIA X | V001 | ...
  ...  | 4 | López María | City Tour | ... | Apart Sol | 205 | DIRECTO | V002 | ...

sábado, 2 de enero de 2016           ← Siguiente día
TERNSF | Q | NOMBRE | ...
```

**Variantes de fecha soportadas:**
- `lunes, 1 de enero de 2016`
- `lunes, enero 1, 2016`
- `01/01/2016`

---

## 🧠 Qué hace cada módulo

### Módulo 1: `data_processor.py`
- ✅ Parsea el formato especial de fecha-como-fila
- ✅ Normalización temporal (día, mes, año, temporada, feriados ARG)
- ✅ ID único de cliente (nombre + teléfono normalizado)
- ✅ Detección de clientes recurrentes
- ✅ Categorización de actividades por palabras clave
- ✅ Limpieza de Q (cantidad), nombres, teléfonos

### Módulo 2: `ml_models.py`
- ✅ **Forecasting** con Prophet (estacionalidad multiplicativa hemisferio sur)
- ✅ **Market Basket Analysis** con Apriori (mlxtend)
- ✅ **Segmentación RFM** con K-Means (K óptimo automático)
- ✅ Análisis de estacionalidad
- ✅ Ranking de alojamientos y operadores

### Módulo 3: `app.py`
- ✅ Dashboard con 7 tabs completos
- ✅ Filtros dinámicos por año, categoría, operador
- ✅ Tema oscuro corporativo
- ✅ Exportación CSV
- ✅ Todo calculado sobre el Excel real cargado

---

## 🎛️ Tabs del Dashboard

| Tab | Contenido |
|-----|-----------|
| 🏠 Resumen | KPIs, evolución anual, heatmap, top actividades |
| 🔮 Forecasting | Predicción 6 meses con bandas de confianza |
| 📅 Estacionalidad | Por mes, día, temporada, finde vs semana |
| 🛒 Canastas | Reglas de asociación y paquetes sugeridos |
| 👥 Clientes RFM | Segmentos con estrategias de marketing |
| 🤝 B2B | Ranking hoteles y análisis por operador |
| 📋 Datos | Explorador filtrable + descarga CSV |

---

## ⚙️ Personalización

### Agregar categorías de actividad
En `data_processor.py`, modificar `CATEGORIAS_ACTIVIDAD`:
```python
CATEGORIAS_ACTIVIDAD = {
    "Tu Categoría": ["keyword1", "keyword2", ...],
    ...
}
```

### Cambiar temporadas
```python
def calcular_temporada(mes: int) -> str:
    # Adaptá según tu región y negocio
```

### Cambiar parámetros del forecasting
En `app.py`, tab Forecasting, el slider de semanas controla el horizonte.
En `ml_models.py`, `changepoint_prior_scale` controla la flexibilidad de la tendencia.

---

## 🐛 Troubleshooting

| Error | Solución |
|-------|----------|
| `No se encontró la fila de encabezados` | Verificá que el Excel tenga Q, NOMBRE, ACTIVIDAD en alguna fila |
| `Prophet no instalado` | `pip install prophet` (requiere pystan) |
| `mlxtend no encontrado` | `pip install mlxtend` |
| Dashboard lento | El caché de Streamlit acelera los recálculos. Primera carga es más lenta. |

---

## 📞 Flujo de Trabajo Recomendado

```
Excel → data_processor.py → datos_procesados.parquet
                                       ↓
                              ml_models.py (entrenamiento)
                                       ↓
                              app.py (dashboard interactivo)
```

Para datasets grandes (>100k filas), corré primero:
```bash
python data_processor.py mi_excel.xlsx
# Guarda output/datos_procesados.parquet
```
Luego el dashboard cargará el parquet directamente (mucho más rápido).
