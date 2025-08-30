import pandas as pd
from pathlib import Path
import os

#  Detectar carpeta base donde está el script
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


CSV_PATH = BASE_DIR / "superstore_clean.csv"
EXCEL_PATH = BASE_DIR / "superstore_clean.practica2_reporte.xlsx"


if not CSV_PATH.exists():
    raise FileNotFoundError(f"❌ No se encontró el archivo CSV en: {CSV_PATH}")

# Leer dataset
df = pd.read_csv(CSV_PATH)


info_dict = {
    "Número de filas": [df.shape[0]],
    "Número de columnas": [df.shape[1]],
    "Columnas": [", ".join(df.columns)],
}

info_df = pd.DataFrame(info_dict)

# Tipos de datos y valores nulos

tipos_df = pd.DataFrame({
    "Columna": df.columns,
    "Tipo de dato": df.dtypes.astype(str),
    "Valores nulos": df.isnull().sum(),
    "Valores únicos": df.nunique()
})


# Estadísticas descriptivas

desc_df = df.describe(include="all").transpose()


#  Conteo por categorías (si hay columnas categóricas)

categorical_counts = {}
for col in df.select_dtypes(include=["object"]).columns:
    categorical_counts[col] = df[col].value_counts()



with pd.ExcelWriter(EXCEL_PATH, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="Datos completos")
    info_df.to_excel(writer, index=False, sheet_name="Info general")
    tipos_df.to_excel(writer, index=False, sheet_name="Tipos de datos")
    desc_df.to_excel(writer, sheet_name="Estadísticas descriptivas")

    # Guardar conteos categóricos en hojas separadas
    for col, series in categorical_counts.items():
        series.to_frame(name="Frecuencia").to_excel(writer, sheet_name=f"Conteo_{col}")

print(f"✅ Reporte generado en: {EXCEL_PATH}")


