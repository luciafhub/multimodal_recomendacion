# ==============================
# Entrenamiento unificado - Modelos de Recomendación con CLIP
# ==============================

import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")

# ==============================
# Configuración general
# ==============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = DEVICE == "cuda"
BATCH_SIZE = 128
LR_LIST = [1e-5, 1e-4, 5e-4, 1e-3]
MAX_EPOCHS = 1000
PATIENCE = 10
WEIGHT_DECAY = 1e-6

# ==============================
# Helpers comunes
# ==============================

def is_valid(x): return isinstance(x, (list, np.ndarray)) and len(x) == 512

def compute_rmse(model, loader, criterion):
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                x, y = batch
                x, y = x.to(DEVICE), y.to(DEVICE)
                preds = model(x)
            elif len(batch) == 4:
                u, i, x, y = batch
                u, i, x, y = u.to(DEVICE), i.to(DEVICE), x.to(DEVICE), y.to(DEVICE)
                preds = model(u, i, x)
            else:
                raise ValueError("Formato de batch desconocido.")
            loss = criterion(preds, y)
            total += loss.item() * y.size(0)
            n += y.size(0)
    return np.sqrt(total / n)

def entrenar_modelo(model, train_loader, val_loader, test_loader, modelo_nombre, lr, output_excel):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_val, best_epoch, best_state = float('inf'), 0, None
    train_hist, val_hist = [], []
    epochs_no_improve = 0

    for epoch in tqdm(range(1, MAX_EPOCHS + 1), desc=f"{modelo_nombre} lr={lr}"):
        model.train()
        running_loss, count = 0.0, 0

        for batch in train_loader:
            if len(batch) == 2:
                x, y = batch
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                if use_amp:
                    with torch.cuda.amp.autocast():
                        pred = model(x)
                        loss = criterion(pred, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred = model(x)
                    loss = criterion(pred, y)
                    loss.backward()
                    optimizer.step()
            elif len(batch) == 4:
                u, i, x, y = batch
                u, i, x, y = u.to(DEVICE), i.to(DEVICE), x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                pred = model(u, i, x)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
            else:
                raise ValueError("Formato de batch desconocido.")

            running_loss += loss.item() * y.size(0)
            count += y.size(0)

        train_rmse = np.sqrt(running_loss / count)
        val_rmse = compute_rmse(model, val_loader, criterion)
        train_hist.append(train_rmse)
        val_hist.append(val_rmse)

        if val_rmse < best_val:
            best_val, best_epoch, best_state = val_rmse, epoch, model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping en epoch {epoch} (mejor: {best_epoch})")
                break

    model.load_state_dict(best_state)
    test_rmse = compute_rmse(model, test_loader, criterion)

    # Curva de aprendizaje
    plt.figure()
    plt.plot(train_hist, label='Train RMSE')
    plt.plot(val_hist, label='Val RMSE')
    plt.axvline(best_epoch, linestyle='--', color='gray', label='Best Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title(f'{modelo_nombre} @ lr={lr}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'curve_{modelo_nombre}_lr_{lr}.pdf')
    plt.close()

    return {
        'Arquitectura': modelo_nombre,
        'learning_rate': lr,
        'train_rmse': train_hist[best_epoch - 1],
        'val_rmse': best_val,
        'test_rmse': test_rmse,
        'epochs': best_epoch
    }

# ==============================
# Arquitectura 5: Imagen sola (C2)
# ==============================

class CLIPImageDataset(Dataset):
    def __init__(self, df):
        self.x = torch.from_numpy(np.stack(df['image_emb'].values)).float()
        self.y = torch.tensor(df['rating'].astype(float).values, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

class CLIPImageRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.mlp(x).squeeze(1)

# ==============================
# Arquitectura 6: Imagen + Texto
# ==============================

class CLIPMixedDataset(Dataset):
    def __init__(self, df):
        img_emb = np.stack(df['image_emb'].values)
        txt_emb = np.stack(df['text_emb'].values)
        self.x = torch.from_numpy(np.concatenate([img_emb, txt_emb], axis=1)).float()
        self.y = torch.tensor(df['rating'].astype(float).values, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

class CLIPMixedRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.mlp(x).squeeze(1)

# ==============================
# Arquitectura 7: Full (Imagen + Texto + IDs)
# ==============================

class FullDataset(Dataset):
    def __init__(self, df):
        self.u = torch.tensor(df['user_enc'].values, dtype=torch.long)
        self.i = torch.tensor(df['item_enc'].values, dtype=torch.long)
        img_emb = np.stack(df['image_emb'].values)
        txt_emb = np.stack(df['text_emb'].values)
        self.x = torch.from_numpy(np.concatenate([img_emb, txt_emb], axis=1)).float()
        self.y = torch.tensor(df['rating'].astype(float).values, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.u[idx], self.i[idx], self.x[idx], self.y[idx]

class FullModel(nn.Module):
    def __init__(self, n_users, n_items, clip_dim=1024, emb_dim=50):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        nn.init.normal_(self.user_emb.weight, mean=1.0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=1.0, std=0.01)
        self.mlp = nn.Sequential(
