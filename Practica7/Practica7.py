# Práctica 7 - Data Clustering con K-Means

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns


df = pd.read_excel("superstore_clean.practica2_reporte.xlsx", sheet_name=0)

variables = ["Sales", "Quantity"]
X = df[variables]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MODELO K-MEANS 
k = 3  # número de clusters
kmeans = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

centroids = kmeans.cluster_centers_

# === GUARDAR RESULTADOS ===
df.to_csv("Practica7_resultados_clusters.csv", index=False)
print("Resultados guardados en Practica7_resultados_clusters.csv")


print("\n=== Ejemplo de agrupamiento ===")
print(df[["Sales", "Quantity", "Cluster"]].head(10))

# GRAFICAR CLUSTERS 
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df["Sales"], y=df["Quantity"],
    hue=df["Cluster"], palette="Set2", s=70
)
plt.title("Clustering con K-Means")
plt.xlabel("Ventas (Sales)")
plt.ylabel("Cantidad (Quantity)")
plt.legend(title="Cluster")
plt.savefig("Practica7_clusters.png")
plt.show()

#MÉTODO DEL CODO PARA ELEGIR K 
inertia = []
K_range = range(1, 10)
for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42)
    kmeans_temp.fit(X_scaled)
    inertia.append(kmeans_temp.inertia_)

plt.figure(figsize=(7,5))
plt.plot(K_range, inertia, 'bo-')
plt.xlabel("Número de Clusters (k)")
plt.ylabel("Inercia")
plt.title("Método del Codo - Selección de K")
plt.grid(True)
plt.savefig("Practica7_elbow_method.png")
plt.show()

print("Gráfica de clusters Practica7_clusters.png")
print("Gráfica Practica7_elbow_method.png")
