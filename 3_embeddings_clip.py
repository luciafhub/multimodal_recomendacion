# Instalación de la librería CLIP desde GitHub
!pip install git+https://github.com/openai/CLIP.git

# Importación de librerías necesarias
import pandas as pd
import numpy as np
import torch
import clip
from tqdm.auto import tqdm

# Configuración del modelo y dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Tamaño del batch para procesar texto
BATCH_TEXT = 32

# Ruta del archivo de entrada y salida
input_pkl = "/kaggle/input/gijon"
output_pkl = "/kaggle/working/gijon_textemb.pkl"

# Carga del dataset de reseñas con texto
df = pd.read_pickle(input_pkl)

# Procesamiento de textos y generación de embeddings en lotes
texts = df["text"].fillna("").astype(str).tolist()
text_emb_en = []

for i in tqdm(range(0, len(texts), BATCH_TEXT), desc="📝 Embedding de textos"):
    batch = texts[i : i + BATCH_TEXT]
    toks = clip.tokenize(batch, truncate=True).to(device)
    with torch.no_grad():
        embs = model.encode_text(toks).cpu().numpy()
    text_emb_en.append(embs)

# Concatenación de todos los embeddings
text_emb_en = np.vstack(text_emb_en)

# Añadir los embeddings al DataFrame y guardar
df["text_emb"] = list(text_emb_en)
df.to_pickle(output_pkl)
print(f"Texto embebido añadido y guardado en {output_pkl}")

# ----------------------------------------------------
# Segunda parte: obtención de embeddings de imágenes
# ----------------------------------------------------

# Importación de librerías necesarias
import os
import json, ast, requests
from PIL import Image
from io import BytesIO

# Ruta del archivo de entrada (con embeddings de texto) y salida
input_pkl = "/kaggle/input/gijon_textemb.pkl"
output_pkl = "/kaggle/working/gijon_emb.pkl"

# Configuración de modelo y batch de imágenes
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
BATCH_IMG = 16

# Selección de columnas relevantes
cols_to_keep = ['images', 'rating', 'restaurantId', 'reviewId', 'text', 'text_emb', 'userId']

# Carga del dataset y filtrado de columnas
df = pd.read_pickle(input_pkl)
df = df[cols_to_keep]

# Función auxiliar para extraer una URL desde una entrada de imagen
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
            return ast.literal_eval(item).get("image_url_lowres", None)
        except:
            try:
                return json.loads(item.replace("'", '"')).get("image_url_lowres", None)
            except:
                return None
    return None

# Extracción de la primera imagen válida por reseña
urls = []
idxs = []
for idx, imgs in enumerate(df["images"]):
    if isinstance(imgs, list) and len(imgs) > 0:
        url = extract_url(imgs[0])
        if url:
            urls.append(url)
            idxs.append(idx)

# Procesamiento por lotes de imágenes
image_embs = []
sel_urls = []
valid_idxs = []

for i in tqdm(range(0, len(urls), BATCH_IMG), desc="🖼️ Procesando lotes de imágenes"):
    batch_urls = urls[i:i + BATCH_IMG]
    batch_imgs = []
    batch_valid = []

    # Descargar imágenes válidas y convertirlas a tensores
    for j, url in enumerate(batch_urls):
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                img = Image.open(BytesIO(r.content)).convert("RGB")
                img_tensor = preprocess(img)
                batch_imgs.append(img_tensor)
                batch_valid.append(j)
        except:
            continue

    if not batch_imgs:
        continue

    # Codificar imágenes con CLIP
    batch_tensor = torch.stack(batch_imgs).to(device)
    with torch.no_grad():
        batch_embs = model.encode_image(batch_tensor).cpu().numpy()

    # Guardar embeddings y sus índices
    for j in range(len(batch_embs)):
        orig_idx = idxs[i + batch_valid[j]]
        image_embs.append(batch_embs[j])
        sel_urls.append(urls[i + batch_valid[j]])
        valid_idxs.append(orig_idx)

# Crear nuevo DataFrame solo con reseñas con imagen válida
df = df.iloc[valid_idxs].copy()
df["image_emb"] = image_embs
df["sel_image_url"] = sel_urls
df = df[cols_to_keep + ["image_emb", "sel_image_url"]]

# Guardar resultado final
df.to_pickle(output_pkl)
print(f"Embeddings guardados en {output_pkl} ({len(df)} reseñas con imagen válida)")
