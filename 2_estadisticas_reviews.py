# Librerías necesarias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Selección del subconjunto de ciudad
subset_name = "/kaggle/input/gijon"  
# También se puede cambiar a otra ciudad
subset_name = "/kaggle/input/newyorkcity"

# Carga de datasets
restau = pd.read_pickle(subset_name + "/restaurants.pkl")
reviews = pd.read_pickle(subset_name + "/reviews.pkl")
users = pd.read_pickle(subset_name + "/users.pkl")

# Visualización de columnas disponibles
print("COLUMNAS RESTAURANTE")
print(restau.columns)
print(restau.head())

print("COLUMNAS REVIEWS")
print(reviews.columns)
print(reviews.head())

print("COLUMNAS USUARIO")
print(users.columns)
print(users.head())

# Estadísticas generales
total_reviews = reviews.shape[0]
total_restaurants = restau.shape[0]
total_users = users.shape[0]

print("Estadísticas generales:")
print(f"Total de reseñas: {total_reviews}")
print(f"Total de restaurantes: {total_restaurants}")
print(f"Total de usuarios: {total_users}")

# Media de reseñas por usuario
reviews_por_usuario = reviews.groupby('userId').size()
print(f"\nNúmero medio de reseñas por usuario: {reviews_por_usuario.mean():.2f}")

# Media de reseñas por restaurante
reviews_por_restaurante = reviews.groupby('restaurantId').size()
print(f"Número medio de reseñas por restaurante: {reviews_por_restaurante.mean():.2f}")

# Estadísticas de imágenes
if 'images' in reviews.columns:
    reviews['num_images'] = reviews['images'].apply(lambda imgs: len(imgs) if isinstance(imgs, list) else 0)
    total_images = reviews['num_images'].sum()
    print(f"\nTotal de imágenes en las reseñas: {total_images}")
    print(f"Número medio de imágenes por usuario: {reviews.groupby('userId')['num_images'].sum().mean():.2f}")
    print(f"Número medio de imágenes por restaurante: {reviews.groupby('restaurantId')['num_images'].sum().mean():.2f}")
else:
    print("\nNo se encontró la columna 'images' en el dataset.")

# Número de reseñas con imágenes
reviews_with_images = reviews[reviews['images'].apply(lambda imgs: isinstance(imgs, list) and len(imgs) > 0)]
print(f"\nTotal de reseñas con imágenes: {reviews_with_images.shape[0]}")

# Carga de subconjunto filtrado para Gijón (opcional)
df_gijon = pd.read_pickle("/kaggle/input/borrarsoloimages/reviews_with_images.pkl")
print("Usuarios con reseñas con imágenes:", df_gijon['userId'].nunique())
print("Restaurantes con reseñas con imágenes:", df_gijon['restaurantId'].nunique())

# Histograma: número de reseñas por usuario (hasta 10)
user_review_counts = reviews['userId'].value_counts()
plt.figure(figsize=(6, 4))
bars = plt.hist(user_review_counts, bins=range(1, 12), color='#4B6C8F', edgecolor='black', rwidth=0.85)
plt.xlabel("Cantidad de reseñas", fontsize=11)
plt.ylabel("Número de usuarios", fontsize=11)
centros = bars[1][:-1] + 0.5
plt.xticks(centros, range(1, 11), fontsize=9)
plt.yticks(fontsize=9)
plt.grid(axis='y', linestyle='--', alpha=0.5)
for i, height in enumerate(bars[0]):
    if height > 0:
        plt.text(bars[1][i] + 0.5, height + max(bars[0]) * 0.02, str(int(height)), ha='center', fontsize=8,
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1))
plt.tight_layout()
plt.savefig("grafica_top10_resenas_por_usuario.pdf")
plt.show()

# Histograma: número de reseñas por restaurante (hasta 190)
restaurant_review_counts = reviews['restaurantId'].value_counts()
filtered_counts = restaurant_review_counts[restaurant_review_counts <= 190]
plt.figure(figsize=(8, 4))
plt.hist(filtered_counts, bins=range(1, 192), color='#4B6C8F', edgecolor='black', rwidth=0.85)
plt.xlabel("Cantidad de reseñas", fontsize=11)
plt.ylabel("Número de restaurantes", fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("grafica_restaurantes_reseñas_150.pdf")
plt.show()

# Boxplot de puntuaciones de los 10 usuarios más activos
top_10_users = user_review_counts.head(10).index
short_ids_10 = {uid: uid[:4] + "..." + uid[-4:] for uid in top_10_users}
filtered = reviews[reviews['userId'].isin(top_10_users)].copy()
filtered['short_userId'] = filtered['userId'].map(short_ids_10)
ordered_short_ids = [short_ids_10[uid] for uid in top_10_users]
filtered['short_userId'] = pd.Categorical(filtered['short_userId'], categories=ordered_short_ids, ordered=True)

plt.figure(figsize=(8, 4))
sns.set_style("white")
sns.boxplot(
    x="short_userId", y="rating", data=filtered, color="#4B6C8F", width=0.4, linewidth=1.2,
    boxprops=dict(edgecolor="black"), whiskerprops=dict(color="black"),
    capprops=dict(color="black"), medianprops=dict(color="black")
)
plt.xlabel("Usuario (ID abreviado)", fontsize=11)
plt.ylabel("Puntuación", fontsize=11)
plt.xticks(fontsize=9, rotation=30)
plt.yticks([10, 20, 30, 40, 50], fontsize=9)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig("grafica_boxplot_puntuaciones_top10_activos.pdf")
plt.show()

print("Top 10 usuarios más activos:")
print(user_review_counts.head(10))

# Gráfico de pastel: % de reseñas con y sin imágenes
has_image = reviews['images'].apply(lambda x: isinstance(x, list) and len(x) > 0)
image_counts = has_image.value_counts()
plt.figure(figsize=(5, 5))
plt.pie(image_counts, labels=["Con imagen", "Sin imagen"], autopct="%1.1f%%", colors=["#4B6C8F", "#CCCCCC"], startangle=90)
plt.tight_layout()
plt.savefig("grafica_porcentaje_imagenes.pdf")
plt.show()

# Histograma: número de imágenes por reseña
num_images = reviews['images'].apply(lambda x: len(x) if isinstance(x, list) else 0)
num_images = num_images.replace([float('inf'), float('-inf')], pd.NA).dropna()
max_val = int(num_images.max())
bins = np.arange(0, max_val + 2)
centros = np.arange(0.5, max_val + 1.5)

plt.figure(figsize=(5, 4))
sns.set_style("white")
bars = plt.hist(num_images, bins=bins, color="#4B6C8F", edgecolor="black", rwidth=0.7)
plt.xlabel("Cantidad de imágenes", fontsize=11)
plt.ylabel("Número de reseñas", fontsize=11)
plt.xticks(ticks=centros, labels=[str(i) for i in range(0, max_val + 1)], fontsize=9)
plt.yticks(fontsize=9)
plt.grid(axis='y', linestyle='--', alpha=0.4)
for i, height in enumerate(bars[0]):
    if height > 0:
        plt.text(centros[i], height + max(bars[0]) * 0.01, str(int(height)), ha='center', fontsize=8)
plt.tight_layout()
plt.savefig("grafica_hist_imagenes_por_resena_ticks_centrados.pdf")
plt.show()
