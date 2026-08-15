#!/usr/bin/env python3
"""Small PyTorch MLP regressor for candidate throughput.

Deliberately small (two hidden layers, dropout, standardized inputs): the
dataset is a few hundred rows of ~20 tabular features, which is squarely
gradient-boosting territory. This exists as the neural baseline to compare
against models/baseline.py - and as the starting point if the feature set
later grows into something sequential (per-beacon RSSI time series rather
than summary statistics), where a net would start to earn its keep.

Run directly to train and report metrics:
    python -m models.mlp data/dataset.csv --epochs 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import feature_columns, impute_features, load_dataset, split, to_xy


class ThroughputMLP(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...] = (64, 32), dropout: float = 0.1):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class Standardizer:
    """Fit on train only, applied to all splits - avoids leaking eval
    statistics into the model's input scaling."""

    def __init__(self, X: np.ndarray):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std


def train(model, X_train, y_train, X_val, y_val, epochs: int, lr: float,
          batch_size: int, weight_decay: float, seed: int, verbose: bool = True):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    Xtr = torch.from_numpy(X_train)
    ytr = torch.from_numpy(y_train)
    Xva = torch.from_numpy(X_val)
    yva = torch.from_numpy(y_val)

    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva), yva).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if verbose and (epoch + 1) % 25 == 0:
            print(f"  epoch {epoch + 1:>4}  val_mse={val_loss:.4f}  best={best_val:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def metrics(model, X, y) -> dict:
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X)).numpy()
    resid = y - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true",
                        help="run 2 epochs just to verify the forward/backward path works")
    args = parser.parse_args()

    df = impute_features(load_dataset(args.dataset))
    feats = feature_columns(df)
    train_df, val_df, test_df = split(df, seed=args.seed)

    X_train, y_train = to_xy(train_df, feats)
    X_val, y_val = to_xy(val_df, feats)

    scaler = Standardizer(X_train)
    X_train_s = scaler(X_train).astype(np.float32)
    X_val_s = scaler(X_val).astype(np.float32)

    print(f"dataset={args.dataset} rows={len(df)} features={len(feats)}")
    print(f"split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    model = ThroughputMLP(len(feats), dropout=args.dropout)
    epochs = 2 if args.smoke_test else args.epochs
    model, best_val = train(model, X_train_s, y_train, X_val_s, y_val,
                           epochs=epochs, lr=args.lr, batch_size=args.batch_size,
                           weight_decay=args.weight_decay, seed=args.seed)

    m = metrics(model, X_val_s, y_val)
    print(f"\nval MAE={m['mae']:.3f} RMSE={m['rmse']:.3f} R2={m['r2']:.3f}")
    if args.smoke_test:
        print("smoke test OK: forward/backward/eval path runs end to end")


if __name__ == "__main__":
    main()
