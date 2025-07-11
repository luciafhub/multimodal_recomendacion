# Importación de librerías necesarias
import pandas as pd
import numpy as np

# Fijar semilla para reproducibilidad
np.random.seed(10)

# ------------------------------------------
# Carga del dataset con embeddings y limpieza previa
# ------------------------------------------
df = pd.read_pickle("/kaggle/input/gijon_embed.pkl")

# Reasignación de identificadores únicos consecutivos
df['user_id_new'] = pd.factorize(df['userId'])[0]
df['restaurant_id_new'] = pd.factorize(df['restaurantId'])[0]

# Mezclar aleatoriamente todas las reseñas
df = df.sample(frac=1, random_state=10).reset_index(drop=True)

# ------------------------------------------
# Fase 1: garantizar al menos una reseña por usuario en train
# ------------------------------------------
train_indices = []
holdout_indices = []

for user_id, group in df.groupby('user_id_new'):
    indices = group.index.tolist()
    if len(indices) == 1:
        train_indices.extend(indices)
    else:
        pick = np.random.choice(indices, 1, replace=False)
        train_indices.extend(pick)
        holdout_indices.extend(list(set(indices) - set(pick)))

# Crear conjuntos preliminares
train_df = df.loc[train_indices].copy()
holdout_df = df.loc[holdout_indices].copy()

# ------------------------------------------
# Fase 2: asegurar que todos los restaurantes aparecen en train
# ------------------------------------------
train_restaurants = set(train_df['restaurant_id_new'].unique())
missing_restaurants = set(holdout_df['restaurant_id_new'].unique()) - train_restaurants

for rest_id in sorted(missing_restaurants):
    candidates = holdout_df[holdout_df['restaurant_id_new'] == rest_id]
    if not candidates.empty:
        chosen_idx = np.random.choice(candidates.index, 1, replace=False)
        train_df = pd.concat([train_df, holdout_df.loc[chosen_idx]], ignore_index=True)
        holdout_df = holdout_df.drop(index=chosen_idx)

# ------------------------------------------
# Fase 3: completar train hasta el 80% del total
# ------------------------------------------
train_target_size = int(0.80 * len(df))
if len(train_df) < train_target_size:
    extra_needed = train_target_size - len(train_df)
    extra_samples = holdout_df.sample(n=extra_needed, random_state=10)
    train_df = pd.concat([train_df, extra_samples], ignore_index=True)
    holdout_df = holdout_df.drop(index=extra_samples.index)

# ------------------------------------------
# Fase 4: dividir el resto en validación y test (10% cada uno)
# ------------------------------------------
holdout_df = holdout_df.sample(frac=1, random_state=10).reset_index(drop=True)
val_size = int(0.5 * len(holdout_df))
val_df = holdout_df.iloc[:val_size].copy()
test_df = holdout_df.iloc[val_size:].copy()

# ------------------------------------------
# Selección final de columnas a guardar
# ------------------------------------------
cols_to_keep = ['images', 'rating', 'restaurantId', 'reviewId', 'text', 'userId', 'sel_image_url', 'image_emb', 'text_emb']
train_df = train_df[cols_to_keep + ['user_id_new', 'restaurant_id_new']]
val_df   = val_df[cols_to_keep + ['user_id_new', 'restaurant_id_new']]
test_df  = test_df[cols_to_keep + ['user_id_new', 'restaurant_id_new']]

# ------------------------------------------
# Guardar los datasets particionados
# ------------------------------------------
train_df.to_pickle("/kaggle/working/train_v80.pkl")
val_df.to_pickle("/kaggle/working/val_v10.pkl")
test_df.to_pickle("/kaggle/working/test_v10.pkl")

# ------------------------------------------
# Mostrar resumen de las particiones
# ------------------------------------------
print("\nDatos guardados con éxito.")
print(f"Train: {train_df.shape}")
print(f"Val:   {val_df.shape}")
print(f"Test:  {test_df.shape}")
total = len(df)
print(f"\nPorcentajes aproximados:")
print(f"Train: {len(train_df)/total:.2%}")
print(f"Val:   {len(val_df)/total:.2%}")
print(f"Test:  {len(test_df)/total:.2%}")
