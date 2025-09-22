# Práctica 5 - Modelos Lineales y Correlación

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# CARGAR DATASET (primera hoja del Excel) 
df = pd.read_excel("superstore_clean.practica2_reporte.xlsx", sheet_name=0)

# VERIFICAR COLUMNA 'Profit' 
if "Profit" not in df.columns:
    print("⚠️ No existe la columna 'Profit', se crea como Sales * 0.2")
    df["Profit"] = df["Sales"] * 0.2

# VARIABLES 
X = df[["Sales"]]     # variable independiente
y = df["Profit"]      # variable dependiente

#  MODELO LINEAL 
modelo = LinearRegression()
modelo.fit(X, y)

y_pred = modelo.predict(X)
r2 = r2_score(y, y_pred)

print("Coeficiente (pendiente):", modelo.coef_[0])
print("Intercepto:", modelo.intercept_)
print("R²:", r2)

#  GRAFICAR 
plt.scatter(X, y, color="blue", alpha=0.5, label="Datos reales")
plt.plot(X, y_pred, color="red", linewidth=2, label="Regresión lineal")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.title("Regresión Lineal: Sales vs Profit")
plt.legend()
plt.savefig("Practica5_regresion.png")
plt.show()


resultados = pd.DataFrame({
    "Coeficiente": [modelo.coef_[0]],
    "Intercepto": [modelo.intercept_],
    "R2_score": [r2]
})
resultados.to_csv("Practica5_resultados.csv", index=False)

print("✅ Resultados guardados en Practica5_resultados.csv")
print("✅ Gráfico guardado como Practica5_regresion.png")

