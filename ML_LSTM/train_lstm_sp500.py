"""
train_lstm_sp500.py
Sophisticated LSTM forecaster for S&P‑500 'Close' prices with MLflow tracking.
Run:  python train_lstm_sp500.py --file stocks_data.parquet --seq_len 40 --epochs 50
"""

import argparse, itertools, os, math, time

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# ---------- 0. CLI ---------- #
p = argparse.ArgumentParser()
p.add_argument("--file", default="stocks_data.parquet")
p.add_argument("--seq_len", type=int, default=40)
p.add_argument("--hidden", type=int, default=128)
p.add_argument("--layers", type=int, default=2)
p.add_argument("--dropout", type=float, default=0.3)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--patience", type=int, default=5, help="early‑stop patience")
args = p.parse_args()

# ---------- 1. Load + feature engineering ---------- #
def load_prices(path: str) -> pd.DataFrame:
    df = (
        pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    )
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    df = df.sort_index()

    # technical features
    df["return"] = df["Close"].pct_change()
    df["sma20"] = df["Close"].rolling(20).mean()
    df["momentum10"] = df["Close"] / df["Close"].shift(10) - 1
    df = df.dropna()

    return df

df = load_prices(args.file)
feature_cols = ["Close", "return", "sma20", "momentum10"]

# ---------- 2. Scale & window ---------- #
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df[feature_cols])

seq = args.seq_len
X, y = [], []
for i in range(len(scaled) - seq):
    X.append(scaled[i : i + seq])
    # predict next close price (index 0 in feature vector)
    y.append(scaled[i + seq, 0])
X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

# train 70 %, val 15 %, test 15 %
n = len(X)
train_end, val_end = int(0.7 * n), int(0.85 * n)
splits = (slice(0, train_end), slice(train_end, val_end), slice(val_end, n))
datasets = [
    TensorDataset(torch.tensor(X[s]), torch.tensor(y[s])) for s in splits
]
loaders = [
    DataLoader(ds, batch_size=args.batch, shuffle=i == 0) for i, ds in enumerate(datasets)
]

# ---------- 3. Model ---------- #
class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden, layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        # use last time‑step’s hidden state
        return self.fc(out[:, -1])

device = "cuda" if torch.cuda.is_available() else "cpu"
model = LSTMForecaster(len(feature_cols), args.hidden, args.layers, args.dropout).to(
    device
)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# ---------- 4. Training loop with MLflow ---------- #
mlflow.set_experiment("sp500-lstm")
mlflow.pytorch.autolog(log_models=False)  # we’ll log manually at the end
with mlflow.start_run():
    mlflow.log_params(vars(args))
    best_val = math.inf
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # ---- train ---- #
        model.train()
        for xb, yb in loaders[0]:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        # ---- evaluate ---- #
        model.eval()
        with torch.no_grad():
            val_pred = []
            val_true = []
            for xb, yb in loaders[1]:
                xb, yb = xb.to(device), yb.to(device)
                val_pred.append(model(xb))
                val_true.append(yb)
            val_pred = torch.cat(val_pred).cpu().numpy()
            val_true = torch.cat(val_true).cpu().numpy()
            val_mae = mean_absolute_error(val_true, val_pred)
            mlflow.log_metric("val_mae", val_mae, step=epoch)

        print(
            f"epoch {epoch:02d}  val MAE={val_mae:7.5f}  "
            f"time={time.time()-t0:4.1f}s"
        )

        # early stopping
        if val_mae < best_val:
            best_val, patience_counter = val_mae, 0
            torch.save(model.state_dict(), "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("→ early stop")
                break

    # ---------- 5. Test set & artefacts ---------- #
    model.load_state_dict(torch.load("best.pt"))
    model.eval()
    test_pred, test_true = [], []
    for xb, yb in loaders[2]:
        xb, yb = xb.to(device), yb.to(device)
        test_pred.append(model(xb))
        test_true.append(yb)
    test_pred = torch.cat(test_pred).cpu().numpy()
    test_true = torch.cat(test_true).cpu().numpy()

    test_rmse = math.sqrt(mean_squared_error(test_true, test_pred))
    mlflow.log_metric("test_rmse", test_rmse)

    # Denormalise predictions back to price space
    inv = scaler.inverse_transform(
        np.hstack([test_pred, np.zeros((len(test_pred), len(feature_cols) - 1))])
    )[:, 0]
    actual = scaler.inverse_transform(
        np.hstack([test_true, np.zeros((len(test_true), len(feature_cols) - 1))])
    )[:, 0]

    # align dates for test period
    test_dates = df.index[-len(actual) :]
    pred_df = pd.DataFrame(
        {"date": test_dates, "actual_close": actual, "pred_close": inv}
    )
    pred_df.to_parquet("predictions.parquet", index=False)
    mlflow.log_artifact("predictions.parquet")

    # log full model + scaler
    torch.save({"model_state": model.state_dict(), "scaler": scaler}, "model.pt")
    mlflow.log_artifact("model.pt")
    mlflow.pytorch.log_model(
        model, artifact_path="pytorch-model", input_example=torch.randn(1, seq, len(feature_cols))
    )
