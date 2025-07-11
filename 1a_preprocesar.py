# Instalación de dependencias necesarias
!pip install tqdm

# Importación de librerías
import pandas as pd
import requests
import ast
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm.notebook import tqdm
from PIL import Image
from io import BytesIO

# Habilita el uso de tqdm en funciones apply de pandas
tqdm.pandas()

# Carga del DataFrame con reseñas (que es el archivo pikle a usar)
reviews = pd.read_pickle("/content/reviews.pkl")

# Filtra solo las reseñas que contienen al menos una imagen
reviews_with_images = reviews[reviews['images'].apply(lambda imgs: isinstance(imgs, list) and len(imgs) > 0)].copy()

# Función que extrae la URL de baja resolución desde distintos formatos de entrada
def extract_url(item):
    """
    Extrae la URL 'image_url_lowres' desde un objeto que puede ser:
    - Un diccionario
    - Una cadena con formato de diccionario o JSON
    """
    if isinstance(item, dict):
        return item.get("image_url_lowres", None)
    elif isinstance(item, str):
        try:
            parsed = ast.literal_eval(item)
            return parsed.get("image_url_lowres", None)
        except:
            try:
                parsed = json.loads(item.replace("'", "\""))
                return parsed.get("image_url_lowres", None)
            except:
                return None
    return None

# Función que comprueba si una URL apunta a una imagen accesible y válida (ya que algunas resultan inválidas en el momento de hacer el TFG)
def is_url_accessible(url):
    """
    Verifica si la URL responde con código 200 y si su contenido puede abrirse como una imagen válida.
    """
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            try:
                Image.open(BytesIO(resp.content)).verify()
                return True
            except:
                return False
        return False
    except:
        return False

# Función que procesa cada lista de imágenes en una reseña, manteniendo solo las que son válidas
def clean_image_list_with_progress(image_items):
    """
    Extrae las URLs de las imágenes de una reseña, verifica su validez y devuelve una lista filtrada.
    """
    urls = [extract_url(item) for item in image_items]
    urls = [url for url in urls if url is not None]

    valid_urls = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for url, is_valid in zip(urls, executor.map(is_url_accessible, urls)):
            if is_valid:
                valid_urls.append({'image_url_lowres': url})
    return valid_urls

# Aplica la limpieza de URLs válidas a cada reseña, mostrando barra de progreso
print("Procesando reseñas con imágenes...")
reviews_with_images['images'] = reviews_with_images['images'].progress_apply(clean_image_list_with_progress)

# Elimina reseñas que no conservan imágenes válidas tras el filtrado
reviews_clean = reviews_with_images[reviews_with_images['images'].apply(lambda imgs: len(imgs) > 0)].copy()

# Selecciona las columnas más relevantes (las que se usarán para hacer la tarea de recomendación)
cols_to_keep = ['images', 'rating', 'restaurantId', 'reviewId', 'text', 'userId']
reviews_filtered = reviews_clean[cols_to_keep].copy()

# Reasigna los identificadores de usuarios y restaurantes a enteros consecutivos (con pd.factorize)
reviews_filtered['user_id_new'] = pd.factorize(reviews_filtered['userId'])[0]
reviews_filtered['restaurant_id_new'] = pd.factorize(reviews_filtered['restaurantId'])[0]

# Guarda el conjunto de datos procesado en formato CSV (para visualización más cómoda si se quiere) y Pickle
reviews_filtered.to_csv("reviews_with_clean_images_strict.csv", index=False)
reviews_filtered.to_pickle("reviews_with_clean_images_strict.pkl")

# Muestra información final para verificación
print("Primeras filas tras limpieza:")
print(reviews_filtered.head())
print("\nCantidad de usuarios:", reviews_filtered['user_id_new'].nunique())
print("Cantidad de restaurantes:", reviews_filtered['restaurant_id_new'].nunique())
