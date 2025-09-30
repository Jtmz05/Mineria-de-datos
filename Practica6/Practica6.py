# Práctica 6 - Clasificación con KNN

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

df = pd.read_excel("superstore_clean.practica2_reporte.xlsx", sheet_name=0)

if "Category" not in df.columns:
    raise ValueError("⚠️ No existe la columna 'Category' en tu dataset.")

X = df[["Sales", "Quantity"]]   # características numéricas
y = df["Category"]              # variable objetivo (categórica)

# Convertir categorías a números
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Escalar datos para KNN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42
)

#  MODELO KNN
knn = KNeighborsClassifier(n_neighbors=5)  # K=5 vecinos
knn.fit(X_train, y_train)

# PREDICCIONES
y_pred = knn.predict(X_test)


acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(" Accuracy:", acc)
print("\n=== Reporte de Clasificación ===")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# GRAFICAR MATRIZ DE CONFUSIÓN 
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=encoder.classes_,
            yticklabels=encoder.classes_)
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión - KNN")
plt.savefig("Practica6_confusion_matrix.png")
plt.show()


resultados = pd.DataFrame({
    "Accuracy": [acc],
    "Vecinos (K)": [5]
})
resultados.to_csv("Practica6_resultados.csv", index=False)

print(" Resultados guardados en Practica6_resultados.csv")
print("Matriz de confusión guardada como Practica6_confusion_matrix.png")
