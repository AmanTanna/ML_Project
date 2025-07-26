"""
train_lstm_sp500.py
Sophisticated LSTM forecaster for S&P‑500 'close' prices with MLflow tracking.
Run:  python train_lstm_sp500.py --file ../data_analysis/stocks_data.parquet --seq_len 40 --epochs 50
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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# ---------- MLflow Setup ---------- #
# Set tracking URI to local directory
mlflow.set_tracking_uri("file:./mlruns")
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

# Enable autolog for PyTorch
mlflow.pytorch.autolog(
    log_every_n_epoch=1,
    log_models=True,
    disable_for_unsupported_versions=False,
    exclusive=False,
    disable=False,
    silent=False
)

# ---------- 0. CLI ---------- #
p = argparse.ArgumentParser()
p.add_argument("--file", default="../data_analysis/stocks_data.parquet")
p.add_argument("--seq_len", type=int, default=40)
p.add_argument("--hidden", type=int, default=128)
p.add_argument("--layers", type=int, default=2)
p.add_argument("--dropout", type=float, default=0.3)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--epochs", type=int, default=30)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--patience", type=int, default=5, help="early‑stop patience")
args = p.parse_args()

print(f"Loading data from: {args.file}")

# ---------- 1. Load + feature engineering ---------- #
def load_prices(path: str) -> pd.DataFrame:
    # Check if file exists
    if not os.path.exists(path):
        print(f"ERROR: Data file not found: {path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
        raise FileNotFoundError(f"Data file not found: {path}")
    
    print(f"Loading data from {path}...")
    
    df = (
        pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    )
    
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    
    # Handle different date column names - your data uses 'date' not 'Date'
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        print("Using 'date' column as index")
    elif "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        print("Using 'Date' column as index")
    else:
        print(f"ERROR: No date column found. Available columns: {list(df.columns)}")
        raise ValueError("No date column found")
    
    # Check if we have the required columns
    required_cols = ["close"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Filter for one symbol to start (you can modify this later)
    if "symbol" in df.columns:
        symbols = df["symbol"].unique()
        print(f"Found {len(symbols)} symbols: {symbols[:10]}...")  # Show first 10
        # Use AAPL as example - you can change this
        df = df[df["symbol"] == "AAPL"].copy()
        print(f"Using AAPL data: {len(df)} rows")
    
    df = df.sort_index()

    # technical features
    print("Creating technical features...")
    df["return"] = df["close"].pct_change()
    df["sma20"] = df["close"].rolling(20).mean()
    df["momentum10"] = df["close"] / df["close"].shift(10) - 1
    df = df.dropna()
    
    print(f"After feature engineering: {len(df)} rows")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    return df

try:
    df = load_prices(args.file)
    feature_cols = ["close", "return", "sma20", "momentum10"]
    print(f"Using features: {feature_cols}")
    
except Exception as e:
    print(f"ERROR loading data: {e}")
    print("\nTrying alternative paths...")
    
    # Try different possible paths
    possible_paths = [
        "stocks_data.parquet",
        "../data_analysis/stocks_data.parquet",
        "../../data_analysis/stocks_data.parquet",
        "/Users/jineshshah/Desktop/aman_proj/ML_Project/data_analysis/stocks_data.parquet"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found data at: {path}")
            args.file = path
            df = load_prices(args.file)
            feature_cols = ["close", "return", "sma20", "momentum10"]
            break
    else:
        print("No data file found in any expected location!")
        exit(1)

# ---------- 2. Scale & window ---------- #
print("Scaling and creating sequences...")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df[feature_cols])

seq = args.seq_len
X, y = [], []
for i in range(len(scaled) - seq):
    X.append(scaled[i : i + seq])
    # predict next close price (index 0 in feature vector)
    y.append(scaled[i + seq, 0])
X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)

print(f"Created {len(X)} sequences of length {seq}")

# train 70 %, val 15 %, test 15 %
n = len(X)
train_end, val_end = int(0.7 * n), int(0.85 * n)
splits = (slice(0, train_end), slice(train_end, val_end), slice(val_end, n))
datasets = [
    TensorDataset(torch.tensor(X[s]), torch.tensor(y[s])) for s in splits
]
loaders = [
    DataLoader(ds, batch_size=args.batch, shuffle=i == 0) for i, ds in enumerate(datasets)
]

print(f"Train samples: {len(datasets[0])}, Val samples: {len(datasets[1])}, Test samples: {len(datasets[2])}")

# ---------- 3. Model ---------- #
class LSTMForecaster(nn.Module):
    def __init__(self, input_size: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden, layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        # use last time‑step's hidden state
        return self.fc(out[:, -1])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = LSTMForecaster(len(feature_cols), args.hidden, args.layers, args.dropout).to(
    device
)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

# ---------- 4. Training loop with MLflow ---------- #
print("Starting training...")

# Create/set experiment
experiment_name = "sp500-lstm"
try:
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
except mlflow.exceptions.MlflowException:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")

mlflow.set_experiment(experiment_name)

with mlflow.start_run() as run:
    print(f"MLflow run ID: {run.info.run_id}")
    print(f"MLflow run URI: {mlflow.get_artifact_uri()}")
    
    # Log parameters
    mlflow.log_params(vars(args))
    mlflow.log_param("model_type", "LSTM")
    mlflow.log_param("device", device)
    mlflow.log_param("total_params", sum(p.numel() for p in model.parameters()))
    mlflow.log_param("data_shape", f"{len(df)} rows, {len(feature_cols)} features")
    
    best_val = math.inf
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        # ---- train ---- #
        model.train()
        train_loss = 0
        for xb, yb in loaders[0]:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

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
            val_mse = mean_squared_error(val_true, val_pred)

        # Log metrics
        avg_train_loss = train_loss / len(loaders[0])
        mlflow.log_metrics({
            "train_loss": avg_train_loss,
            "val_mae": val_mae,
            "val_mse": val_mse,
            "epoch_time": time.time() - t0
        }, step=epoch)

        print(
            f"epoch {epoch:02d}  train_loss={avg_train_loss:.5f}  val_mae={val_mae:.5f}  "
            f"time={time.time()-t0:.1f}s"
        )

        # early stopping
        if val_mae < best_val:
            best_val, patience_counter = val_mae, 0
            torch.save(model.state_dict(), "best.pt")
            mlflow.log_metric("best_val_mae", best_val)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("→ early stop")
                mlflow.log_param("early_stopped", True)
                mlflow.log_param("stopped_at_epoch", epoch)
                break

    # ---------- 5. Test set & artefacts ---------- #
    print("Evaluating on test set...")
    model.load_state_dict(torch.load("best.pt", weights_only=True))
    model.eval()
    test_pred, test_true = [], []
    with torch.no_grad():
        for xb, yb in loaders[2]:
            xb, yb = xb.to(device), yb.to(device)
            test_pred.append(model(xb))
            test_true.append(yb)
    test_pred = torch.cat(test_pred).cpu().numpy()
    test_true = torch.cat(test_true).cpu().numpy()

    test_rmse = math.sqrt(mean_squared_error(test_true, test_pred))
    test_mae = mean_absolute_error(test_true, test_pred)
    
    # Log final test metrics
    mlflow.log_metrics({
        "test_rmse": test_rmse,
        "test_mae": test_mae
    })
    print(f"Test RMSE: {test_rmse:.5f}, Test MAE: {test_mae:.5f}")

    # Denormalise predictions back to price space
    inv = scaler.inverse_transform(
        np.hstack([test_pred, np.zeros((len(test_pred), len(feature_cols) - 1))])
    )[:, 0]
    actual = scaler.inverse_transform(
        np.hstack([test_true, np.zeros((len(test_true), len(feature_cols) - 1))])
    )[:, 0]

    # Calculate price-based metrics
    price_mae = mean_absolute_error(actual, inv)
    price_mape = np.mean(np.abs((actual - inv) / actual)) * 100
    
    mlflow.log_metrics({
        "price_mae": price_mae,
        "price_mape": price_mape
    })

    # align dates for test period
    test_dates = df.index[-len(actual) :]
    pred_df = pd.DataFrame(
        {"date": test_dates, "actual_close": actual, "pred_close": inv}
    )
    
    # Save both parquet and CSV
    pred_df.to_parquet("predictions.parquet", index=False)
    pred_df.to_csv("predictions.csv", index=False)
    mlflow.log_artifact("predictions.parquet")
    mlflow.log_artifact("predictions.csv")
    
    # Create visualization
    plt.figure(figsize=(15, 8))
    
    # Plot actual vs predicted prices
    plt.subplot(2, 1, 1)
    plt.plot(test_dates, actual, label='Actual Close Price', color='blue', linewidth=2)
    plt.plot(test_dates, inv, label='Predicted Close Price', color='red', linewidth=2, alpha=0.8)
    plt.title('AAPL Stock Price: Actual vs Predicted', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Close Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Plot prediction error
    plt.subplot(2, 1, 2)
    error = actual - inv
    plt.plot(test_dates, error, label='Prediction Error', color='green', linewidth=1)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.title('Prediction Error Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Error ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("predictions_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create metrics summary plot
    plt.figure(figsize=(12, 6))
    
    # Training metrics over time
    epochs_range = range(1, len(mlflow.get_run(run.info.run_id).data.metrics) // 4 + 1)
    
    plt.subplot(1, 2, 1)
    # Note: You'd need to collect these during training for a proper plot
    # This is a simplified version
    plt.bar(['Train Loss', 'Val MAE', 'Test RMSE', 'Test MAE'], 
            [avg_train_loss, val_mae, test_rmse, test_mae],
            color=['skyblue', 'lightcoral', 'lightgreen', 'orange'])
    plt.title('Final Model Metrics', fontweight='bold')
    plt.ylabel('Error Value')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    # Price-based metrics
    plt.bar(['Price MAE ($)', 'Price MAPE (%)'], 
            [price_mae, price_mape],
            color=['purple', 'gold'])
    plt.title('Price-based Performance', fontweight='bold')
    plt.ylabel('Error Value')
    
    plt.tight_layout()
    plt.savefig("metrics_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Log visualizations
    mlflow.log_artifact("predictions_plot.png")
    mlflow.log_artifact("metrics_summary.png")
    
    print(f"Predictions saved to: predictions.csv and predictions.parquet")
    print(f"Visualizations saved as: predictions_plot.png and metrics_summary.png")

print("Training complete!")
print(f"Check MLflow UI at: http://localhost:5000")
print(f"Run 'mlflow ui' in the directory: {os.getcwd()}")
