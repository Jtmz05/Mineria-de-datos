# Práctica 3 - Data Visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# CARGA DEL ARCHIVO EXCEL
archivo = Path("superstore_clean.practica2_reporte.xlsx")


xls = pd.ExcelFile(archivo)
print("Hojas encontradas:", xls.sheet_names)

df = pd.read_excel(archivo, sheet_name=xls.sheet_names[0])

if "Order Date" in df.columns:
    df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

#  CREACIÓN DE GRÁFICAS 
plt.style.use("ggplot")

# Histograma de ventas
plt.figure(figsize=(8, 5))
sns.histplot(df["Sales"], bins=30, kde=True)
plt.title("Histograma de Ventas")
plt.xlabel("Ventas")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig("grafico_histograma.png")
plt.close()

# Gráfico de pastel de categorías
if "Category" in df.columns:
    plt.figure(figsize=(6, 6))
    df["Category"].value_counts().plot.pie(autopct="%1.1f%%")
    plt.title("Distribución de Categorías")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig("grafico_pie.png")
    plt.close()

# Boxplot de Ventas por Categoría
if "Category" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="Category", y="Sales", data=df)
    plt.title("Boxplot de Ventas por Categoría")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("grafico_boxplot.png")
    plt.close()

# Serie temporal de Ventas por mes
if "Order Date" in df.columns:
    plt.figure(figsize=(10, 5))
    ventas_por_mes = df.groupby(df["Order Date"].dt.to_period("M"))["Sales"].sum()
    ventas_por_mes.plot()
    plt.title("Serie temporal de Ventas")
    plt.xlabel("Fecha")
    plt.ylabel("Ventas Totales")
    plt.tight_layout()
    plt.savefig("grafico_serie.png")
    plt.close()

# Diagrama de dispersión (Sales vs Quantity)
if "Quantity" in df.columns:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="Quantity", y="Sales", data=df, alpha=0.6)
    plt.title("Dispersión: Ventas vs Cantidad")
    plt.xlabel("Cantidad")
    plt.ylabel("Ventas")
    plt.tight_layout()
    plt.savefig("grafico_scatter.png")
    plt.close()

print("✅ Gráficas generadas y guardadas como PNG en tu carpeta de trabajo.")
