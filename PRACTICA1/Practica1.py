import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# 1. Abrir ventana para seleccionar archivo CSV
Tk().withdraw()  # Oculta la ventana principal de Tkinter
file_path = askopenfilename(title="Selecciona el archivo CSV", filetypes=[("CSV files", "*.csv")])

# 2. Cargar dataset
df = pd.read_csv(file_path)

print("Dimensiones iniciales:", df.shape)
print("\nVista previa del dataset:\n", df.head())

# 3. Verificar valores nulos
print("\nValores nulos por columna:\n", df.isnull().sum())

# 4. Eliminar duplicados
df = df.drop_duplicates()

# 5. Convertir columna de fecha
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

print("\nTipos de datos después:\n", df.dtypes)

# 6. Guardar dataset limpio en la misma carpeta donde estaba el archivo original
output_file = file_path.replace(".csv", "_clean.csv")
df.to_csv(output_file, index=False)

print(f"\nArchivo limpio guardado como: {output_file}")