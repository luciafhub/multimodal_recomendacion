# Visualización y evaluación de baseline con media global

# -------------------------------
# Importación de librerías necesarias
# -------------------------------
import pandas as pd
import numpy as np
import sys
import numpy.core.numeric as _num
sys.modules['numpy._core.numeric'] = _num  # Solución para compatibilidad con ciertos entornos

from sklearn.metrics import mean_squared_error

# -------------------------------
# Carga de los datasets
# -------------------------------
train_obj = pd.read_pickle("/kaggle/input/ny8010/train_v80.pkl")
val_obj   = pd.read_pickle("/kaggle/input/ny8010/val_v10.pkl")
test_obj  = pd.read_pickle("/kaggle/input/ny8010/test_v10.pkl")

# Asegurar que los objetos cargados sean DataFrames (por compatibilidad)
train_df = pd.DataFrame(train_obj) if isinstance(train_obj, list) else train_obj
val_df   = pd.DataFrame(val_obj) if isinstance(val_obj, list) else val_obj
test_df  = pd.DataFrame(test_obj) if isinstance(test_obj, list) else test_obj

# Mostrar número de filas de cada conjunto
print("Número de datos")
print("\nTrain:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))
print("\nTotal:", len(train_df) + len(val_df) + len(test_df))

# Mostrar las columnas disponibles
print("\nColumnas")
print(train_df.columns)

# Alias para simplificar el uso posterior
train = train_df
val = val_df
test = test_df

# -------------------------------
# Evaluación de un modelo baseline
# -------------------------------

# Asegurar que la columna 'rating' sea tipo float
for df in (train, val, test):
    df['rating'] = df['rating'].astype(float)

# Definición de la función de evaluación RMSE
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Baseline: media global de los ratings en el conjunto de entrenamiento
global_mean = train['rating'].mean()

# Evaluación del baseline en train, val y test
results = []
for name, df in [("Train", train), ("Val", val), ("Test", test)]:
    preds_global = np.full(len(df), global_mean)
    results.append({
        'Baseline': 'Global Mean',
        'Dataset':  name,
        'RMSE':     rmse(df['rating'], preds_global)
    })

# Mostrar resultados finales
df_results = pd.DataFrame(results)
print("\nRMSE del baseline (media global):")
print(df_results.pivot(index='Baseline', columns='Dataset', values='RMSE').round(4))





################
# ARQUITECTURA 1_FM
################


import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================
# CONFIGURACIÓN
# ============================
start_total = time.time()
cudnn.benchmark = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = DEVICE == "cuda"

EMB_DIM = 50
PATIENCE = 10
MAX_EPOCHS = 1000
BATCH_SIZE = 128
LR_LIST = [1e-5, 1e-4, 5e-4, 1e-3]
WEIGHT_DECAY = 1e-6
RESULTS_XLSX = "Arquitectura_1_FM_NY.xlsx"

# ============================
# Dataset ya cargado: train, val, test
# ============================
class ReviewsDataset(Dataset):
    def __init__(self, df):
        self.u = torch.tensor(df['user_id_new'].values, dtype=torch.long)
        self.i = torch.tensor(df['restaurant_id_new'].values, dtype=torch.long)
        self.r = torch.tensor(df['rating'].values, dtype=torch.float32)
    def __len__(self):
        return len(self.r)
    def __getitem__(self, idx):
        return self.u[idx], self.i[idx], self.r[idx]

train_loader = DataLoader(ReviewsDataset(train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(ReviewsDataset(val), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(ReviewsDataset(test), batch_size=BATCH_SIZE, shuffle=False)

# ============================
# Modelo FM
# ============================
class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, emb_dim=EMB_DIM):
        super().__init__()
        self.Eu = nn.Embedding(n_users, emb_dim)
        self.Ei = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.Eu.weight, mean=1.0, std=0.01)
        nn.init.normal_(self.Ei.weight, mean=1.0, std=0.01)
    def forward(self, u, i):
        return (self.Eu(u) * self.Ei(i)).sum(dim=1)

# ============================
# RMSE
# ============================
def compute_rmse(model, loader, criterion):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for u, i, r in loader:
            u, i, r = u.to(DEVICE), i.to(DEVICE), r.to(DEVICE)
            pred = model(u, i)
            loss = criterion(pred, r)
            total += loss.item() * r.size(0)
            n += r.size(0)
    return np.sqrt(total / n)

# ============================
# Entrenamiento
# ============================
n_users = train['user_id_new'].nunique()
n_items = train['restaurant_id_new'].nunique()

results = []
criterion = nn.MSELoss()

for lr in LR_LIST:
    print(f"\nEntrenando con learning rate = {lr}")
    start_lr_time = time.time()

    model = MatrixFactorization(n_users, n_items).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    best_val, best_state, best_epoch = float('inf'), None, 0
    epochs_no_improve = 0
    train_hist, val_hist = [], []

    for epoch in tqdm(range(1, MAX_EPOCHS + 1), desc=f"LR {lr}"):
        model.train()
        running, count = 0.0, 0
        for u, i, r in train_loader:
            u, i, r = u.to(DEVICE), i.to(DEVICE), r.to(DEVICE)
            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    pred = model(u, i)
                    loss = criterion(pred, r)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(u, i)
                loss = criterion(pred, r)
                loss.backward()
                optimizer.step()

            running += loss.item() * r.size(0)
            count += r.size(0)

        train_rmse = np.sqrt(running / count)
        val_rmse = compute_rmse(model, val_loader, criterion)
        train_hist.append(train_rmse)
        val_hist.append(val_rmse)

        if val_rmse < best_val:
            best_val, best_state, best_epoch = val_rmse, model.state_dict(), epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping en epoch {epoch} (mejor: {best_epoch})")
                break

    model.load_state_dict(best_state)
    test_rmse = compute_rmse(model, test_loader, criterion)

    time_lr_min = round((time.time() - start_lr_time) / 60, 2)

    results.append({
        'Arquitectura': '1-FM-NY',
        'learning_rate': lr,
        'train_rmse': train_hist[best_epoch - 1],
        'val_rmse': best_val,
        'test_rmse': test_rmse,
        'epochs': best_epoch,
        'train_time_min': time_lr_min
    })

    # Guardar curva en PDF
    plt.figure()
    plt.plot(train_hist, label='Train RMSE')
    plt.plot(val_hist, label='Val RMSE')
    plt.axvline(best_epoch, linestyle='--', color='gray', label='Best Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title(f'1-FM @ lr={lr}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'curve_1FM_lr_{lr}.pdf', format='pdf')
    plt.close()

# ============================
# Guardar resultados en Excel
# ============================
df_results = pd.DataFrame(results)
df_results.to_excel(RESULTS_XLSX, index=False)

total_time_min = round((time.time() - start_total) / 60, 2)
print(f"\nResultados guardados en `{RESULTS_XLSX}`")
print(f"Tiempo total de entrenamiento: {total_time_min} minutos")
