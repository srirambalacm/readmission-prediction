"""PyTorch MLP with categorical embeddings, for comparison against LightGBM."""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

from src.features import RANDOM_STATE


def _embedding_dim(cardinality: int) -> int:
    """fastai heuristic, half the cardinality, capped at 50."""
    return int(min(50, (cardinality + 1) // 2))


class TabularMLP(nn.Module):
    def __init__(self, cardinalities: list[int], n_numeric: int,
                 hidden: tuple[int, ...] = (128, 64), dropout: float = 0.3):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(c, _embedding_dim(c)) for c in cardinalities
        ])
        emb_total = sum(_embedding_dim(c) for c in cardinalities)

        layers, in_dim = [], emb_total + n_numeric
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [x_num], dim=1)
        return self.net(x).squeeze(1)
def encode_for_mlp(X_train: pd.DataFrame, X_valid: pd.DataFrame):
    """Integer-encode categoricals and standardize numerics.

    Category vocabularies are fit on training data only. Unseen levels in
    validation map to index 0, so an extra slot is reserved per feature.
    """
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()

    cardinalities, tr_cat, va_cat = [], [], []
    for col in cat_cols:
        levels = pd.Index(sorted(X_train[col].astype(str).unique()))
        mapping = {v: i + 1 for i, v in enumerate(levels)}  # 0 reserved for unseen
        tr_cat.append(X_train[col].astype(str).map(mapping).fillna(0).to_numpy())
        va_cat.append(X_valid[col].astype(str).map(mapping).fillna(0).to_numpy())
        cardinalities.append(len(levels) + 1)

    scaler = StandardScaler().fit(X_train[num_cols])

    return (
        np.stack(tr_cat, axis=1).astype(np.int64),
        scaler.transform(X_train[num_cols]).astype(np.float32),
        np.stack(va_cat, axis=1).astype(np.int64),
        scaler.transform(X_valid[num_cols]).astype(np.float32),
        cardinalities,
    )
def train_mlp(X_train, y_train, X_valid, y_valid, epochs: int = 40,
              batch_size: int = 512, lr: float = 1e-3, patience: int = 5,
              verbose: bool = True):
    """Train with early stopping on validation AUPRC."""
    torch.manual_seed(RANDOM_STATE)

    tr_cat, tr_num, va_cat, va_num, cards = encode_for_mlp(X_train, X_valid)

    tr_cat_t = torch.from_numpy(tr_cat)
    tr_num_t = torch.from_numpy(tr_num)
    tr_y_t = torch.from_numpy(y_train.to_numpy().astype(np.float32))
    va_cat_t = torch.from_numpy(va_cat)
    va_num_t = torch.from_numpy(va_num)

    model = TabularMLP(cards, tr_num.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    n = len(tr_y_t)
    best_auprc, best_state, bad_epochs = -1.0, None, 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(tr_cat_t[idx], tr_num_t[idx]), tr_y_t[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(va_cat_t, va_num_t)).numpy()
        auprc = average_precision_score(y_valid, scores)

        if auprc > best_auprc:
            best_auprc, bad_epochs = auprc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1

        if verbose:
            print(f"epoch {epoch+1:2d}  val AUPRC {auprc:.4f}"
                  f"{'  *' if bad_epochs == 0 else ''}")

        if bad_epochs >= patience:
            if verbose:
                print(f"early stop at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final = torch.sigmoid(model(va_cat_t, va_num_t)).numpy()

    return model, final