import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt


df = pd.read_excel("superstore_clean.practica2_reporte.xlsx")


columna_texto = "Category"


texto = " ".join(df[columna_texto].astype(str))

# Generar Word Cloud
wordcloud = WordCloud(width=1000, height=500, background_color="white").generate(texto)

# Mostrar Word Cloud
plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


wordcloud.to_file("practica9.png")
print("Word Cloud generado ")
