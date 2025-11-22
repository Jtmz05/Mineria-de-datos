import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_excel("superstore_clean.practica2_reporte.xlsx")


df["Order Date"] = pd.to_datetime(df["Order Date"])
df["Mes"] = df["Order Date"].dt.to_period("M").astype(str)

# Ventas mensuales por categoría
pivot = df.pivot_table(values="Sales", index="Mes", columns="Category", aggfunc="sum").fillna(0)

# Gráfica de líneas por categoría
pivot.plot(figsize=(12,6))
plt.title("Ventas mensuales por categoría")
plt.xlabel("Mes")
plt.ylabel("Ventas")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("pia_ventas_categorias.png")
plt.show()

# Pronóstico con regresión lineal
x = np.arange(len(pivot)).reshape(-1, 1)
for categoria in pivot.columns:
    y = pivot[categoria].values
    modelo = LinearRegression().fit(x, y)
    pred = modelo.predict(x)
    plt.plot(pred, label=f"Pronóstico {categoria}")

plt.title("Pronóstico de tendencias por categoría")
plt.xlabel("Tiempo (índice mensual)")
plt.ylabel("Ventas estimadas")
plt.legend()
plt.tight_layout()
plt.savefig("pia_pronostico.png")
plt.show()
