# Práctica 8 - Forecasting con Regresión Lineal (Serie de Tiempo)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np


df = pd.read_excel("superstore_clean.practica2_reporte.xlsx", sheet_name=0)


df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df = df.dropna(subset=["Order Date", "Sales"])
df = df.sort_values("Order Date")


df_monthly = df.groupby(pd.Grouper(key="Order Date", freq="M")).sum(numeric_only=True)["Sales"].reset_index()
df_monthly.columns = ["Fecha", "Ventas"]

# CREAR VARIABLES PARA EL MODELO 
# Convertimos la fecha a número
df_monthly["Dias"] = (df_monthly["Fecha"] - df_monthly["Fecha"].min()).dt.days

X = df_monthly[["Dias"]]
y = df_monthly["Ventas"]

#  DIVIDIR ENTRENAMIENTO Y PRUEBA
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = LinearRegression()
model.fit(X_train, y_train)

# PREDICCIONES 
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

 
dias_futuros = np.arange(df_monthly["Dias"].max(), df_monthly["Dias"].max() + 180, 30)  # próximos 6 meses
ventas_futuras = model.predict(dias_futuros.reshape(-1, 1))


fechas_futuras = pd.date_range(df_monthly["Fecha"].max(), periods=len(dias_futuros)+1, freq="M")[1:]
forecast_df = pd.DataFrame({"Fecha": fechas_futuras, "Ventas_Predichas": ventas_futuras})
forecast_df.to_csv("Practica8_forecast.csv", index=False)

print("'Practica8_forecast.csv'")
print(f"R² del modelo: {r2:.3f}")

#  GRAFICAR RESULTADOS 
plt.figure(figsize=(10,6))
plt.plot(df_monthly["Fecha"], df_monthly["Ventas"], label="Ventas Reales", marker="o")
plt.plot(X_test["Dias"].apply(lambda x: df_monthly["Fecha"].min() + pd.Timedelta(days=x)), y_pred, 
         label="Predicción (Test)", color="orange", linestyle="--")
plt.plot(forecast_df["Fecha"], forecast_df["Ventas_Predichas"], 
         label="Predicción Futura", color="red", linestyle=":")
plt.xlabel("Fecha")
plt.ylabel("Ventas")
plt.title("Forecasting de Ventas - Regresión Lineal")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Practica8_forecast_plot.png")
plt.show()
