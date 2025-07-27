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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# ---------- MLflow Setup ---------- #
# Set tracking URI to local directory
mlflow_dir = os.path.join(os.getcwd(), "mlruns")
mlflow.set_tracking_uri(f"file://{mlflow_dir}")
print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

# Disable autolog to avoid parameter conflicts
# mlflow.pytorch.autolog(
#     log_every_n_epoch=1,
#     log_models=True,
#     disable_for_unsupported_versions=False,
#     exclusive=False,
#     disable=False,
#     silent=False
# )

# ---------- 0. CLI ---------- #
p = argparse.ArgumentParser()
p.add_argument("--file", default="../data_analysis/stocks_data.parquet")
p.add_argument("--symbols", nargs="+", default=["AAPL"], help="List of stock symbols to train on")
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
    
    # Filter for symbols
    if "symbol" in df.columns:
        symbols = df["symbol"].unique()
        print(f"Found {len(symbols)} symbols: {symbols[:10]}...")
        required_symbols = args.symbols
        df = df[df["symbol"].isin(required_symbols)].copy()
        print(f"Filtered to {len(df)} rows for symbols: {required_symbols}")
    else:
        print("No 'symbol' column found, assuming single symbol data")
    
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

# Load data
try:
    df = load_prices(args.file)
    feature_cols = ["close", "return", "sma20", "momentum10"]
    print(f"Using features: {feature_cols}")
    
except Exception as e:
    print(f"ERROR loading data: {e}")
    print("\nTrying alternative paths...")
    
    possible_paths = [
        "stocks_data.parquet",
        "../../data_analysis/stocks_data.parquet",
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

# Create sequences by symbol
seq = args.seq_len
X, y = [], []
symbol_info = {}

for symbol, df_group in df.groupby("symbol"):
    print(f"Creating sequences for symbol: {symbol}")
    scaled = scaler.fit_transform(df_group[feature_cols])
    
    # Store symbol info for later
    symbol_info[symbol] = {
        'start_date': df_group.index.min(),
        'end_date': df_group.index.max(),
        'num_days': len(df_group),
        'price_range': {
            'min': df_group['close'].min(),
            'max': df_group['close'].max(),
            'mean': df_group['close'].mean(),
            'std': df_group['close'].std()
        }
    }
    
    for i in range(len(scaled) - seq):
        X.append(scaled[i : i + seq])
        y.append(scaled[i + seq, 0])

X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)
print(f"Created {len(X)} total sequences from all symbols")

# Data splits
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
        return self.fc(out[:, -1])

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = LSTMForecaster(len(feature_cols), args.hidden, args.layers, args.dropout).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

# ---------- 4. Training loop with comprehensive MLflow tracking ---------- #
print("Starting training...")

experiment_name = "sp500-lstm-comprehensive"
try:
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
except mlflow.exceptions.MlflowException:
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")

mlflow.set_experiment(experiment_name)

with mlflow.start_run() as run:
    print(f"MLflow run ID: {run.info.run_id}")
    
    # ---------- Log comprehensive parameters (fixed) ---------- #
    # Log args parameters one by one to avoid conflicts
    mlflow.log_param("file", args.file)
    mlflow.log_param("symbols_list", str(args.symbols))  # Changed key name
    mlflow.log_param("seq_len", args.seq_len)
    mlflow.log_param("hidden", args.hidden)
    mlflow.log_param("layers", args.layers)
    mlflow.log_param("dropout", args.dropout)
    mlflow.log_param("batch", args.batch)
    mlflow.log_param("epochs", args.epochs)
    mlflow.log_param("lr", args.lr)
    mlflow.log_param("patience", args.patience)
    
    # Additional parameters
    mlflow.log_param("model_type", "LSTM")
    mlflow.log_param("device", device)
    mlflow.log_param("total_params", sum(p.numel() for p in model.parameters()))
    mlflow.log_param("data_shape", f"{len(df)} rows, {len(feature_cols)} features")
    mlflow.log_param("train_samples", len(datasets[0]))
    mlflow.log_param("val_samples", len(datasets[1]))
    mlflow.log_param("test_samples", len(datasets[2]))
    mlflow.log_param("symbols_count", len(args.symbols))  # Changed key name
    mlflow.log_param("symbols_joined", ",".join(args.symbols))  # Changed key name
    
    # Log data statistics
    for symbol, info in symbol_info.items():
        mlflow.log_param(f"{symbol}_start_date", str(info['start_date'].date()))
        mlflow.log_param(f"{symbol}_end_date", str(info['end_date'].date()))
        mlflow.log_param(f"{symbol}_num_days", info['num_days'])
        mlflow.log_param(f"{symbol}_price_min", round(info['price_range']['min'], 2))
        mlflow.log_param(f"{symbol}_price_max", round(info['price_range']['max'], 2))
        mlflow.log_param(f"{symbol}_price_mean", round(info['price_range']['mean'], 2))
        mlflow.log_param(f"{symbol}_price_std", round(info['price_range']['std'], 2))
    
    # Training tracking
    best_val = math.inf
    patience_counter = 0
    training_history = {
        'train_loss': [],
        'val_mae': [],
        'val_mse': [],
        'epoch_time': []
    }

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        # Training
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

        # Validation
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
            val_rmse = math.sqrt(val_mse)

        # Store history
        avg_train_loss = train_loss / len(loaders[0])
        epoch_time = time.time() - t0
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_mae'].append(val_mae)
        training_history['val_mse'].append(val_mse)
        training_history['epoch_time'].append(epoch_time)

        # Early stopping check
        is_best_epoch = val_mae < best_val
        
        # Log metrics
        mlflow.log_metrics({
            "train_loss": avg_train_loss,
            "val_mae": val_mae,
            "val_mse": val_mse,
            "val_rmse": val_rmse,
            "epoch_time": epoch_time,
            "learning_rate": args.lr,
            "patience_counter": patience_counter,
            "is_best_epoch": int(is_best_epoch),  # 1 if best, 0 if not
            "current_best_val_mae": best_val if not is_best_epoch else val_mae
        }, step=epoch)

        print(f"epoch {epoch:02d}  train_loss={avg_train_loss:.5f}  val_mae={val_mae:.5f}  time={epoch_time:.1f}s")

        # Early stopping
        if is_best_epoch:
            best_val, patience_counter = val_mae, 0
            torch.save(model.state_dict(), "best.pt")
            # Log the epoch number as a metric (can be updated)
            mlflow.log_metric("best_epoch_number", epoch, step=epoch)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("→ early stop")
                break

    # Log final parameters after training is complete
    final_params = {
        "early_stopped": patience_counter >= args.patience,
        "stopped_at_epoch": epoch,
        "total_epochs": epoch,
        "final_patience_counter": patience_counter,
        "training_completed": True
    }
    
    for key, value in final_params.items():
        mlflow.log_param(key, value)
    
    # ---------- 5. Comprehensive Test Evaluation ---------- #
    print("Comprehensive test evaluation...")
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

    # Calculate comprehensive metrics
    test_mse = mean_squared_error(test_true, test_pred)
    test_rmse = math.sqrt(test_mse)
    test_mae = mean_absolute_error(test_true, test_pred)
    
    # R-squared (coefficient of determination)
    test_r2 = r2_score(test_true, test_pred)
    
    # Denormalize for price-based metrics
    inv_pred = scaler.inverse_transform(
        np.hstack([test_pred, np.zeros((len(test_pred), len(feature_cols) - 1))])
    )[:, 0]
    inv_true = scaler.inverse_transform(
        np.hstack([test_true, np.zeros((len(test_true), len(feature_cols) - 1))])
    )[:, 0]

    # Price-based metrics
    price_mae = mean_absolute_error(inv_true, inv_pred)
    price_mse = mean_squared_error(inv_true, inv_pred)
    price_rmse = math.sqrt(price_mse)
    price_mape = np.mean(np.abs((inv_true - inv_pred) / inv_true)) * 100
    price_r2 = r2_score(inv_true, inv_pred)
    
    # Accuracy metrics (% predictions within threshold)
    threshold_1pct = np.mean(np.abs((inv_true - inv_pred) / inv_true) <= 0.01) * 100
    threshold_2pct = np.mean(np.abs((inv_true - inv_pred) / inv_true) <= 0.02) * 100  
    threshold_5pct = np.mean(np.abs((inv_true - inv_pred) / inv_true) <= 0.05) * 100
    
    # Direction accuracy (up/down prediction)
    true_direction = np.diff(inv_true) > 0
    pred_direction = np.diff(inv_pred) > 0
    direction_accuracy = np.mean(true_direction == pred_direction) * 100
    
    # Volatility metrics
    true_volatility = np.std(inv_true)
    pred_volatility = np.std(inv_pred)
    volatility_ratio = pred_volatility / true_volatility
    
    # Max/Min prediction accuracy
    max_error = np.max(np.abs(inv_true - inv_pred))
    min_error = np.min(np.abs(inv_true - inv_pred))
    
    # Log all comprehensive metrics
    comprehensive_metrics = {
        # Normalized metrics
        "test_mse": test_mse,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_r2": test_r2,
        
        # Price-based metrics
        "price_mae": price_mae,
        "price_mse": price_mse,
        "price_rmse": price_rmse,
        "price_mape": price_mape,
        "price_r2": price_r2,
        
        # Accuracy metrics
        "accuracy_1pct": threshold_1pct,
        "accuracy_2pct": threshold_2pct,
        "accuracy_5pct": threshold_5pct,
        "direction_accuracy": direction_accuracy,
        
        # Volatility and range metrics
        "true_volatility": true_volatility,
        "pred_volatility": pred_volatility,
        "volatility_ratio": volatility_ratio,
        "max_error": max_error,
        "min_error": min_error,
        
        # Training summary metrics
        "avg_train_loss": np.mean(training_history['train_loss']),
        "final_train_loss": training_history['train_loss'][-1],
        "avg_epoch_time": np.mean(training_history['epoch_time']),
        "total_training_time": sum(training_history['epoch_time']),
        
        # Model performance score (custom composite metric)
        "performance_score": (100 - price_mape) * (price_r2) * (direction_accuracy / 100)
    }
    
    mlflow.log_metrics(comprehensive_metrics)
    
    # Log additional summary stats
    mlflow.log_param("mean_price_actual", round(np.mean(inv_true), 2))
    mlflow.log_param("mean_price_predicted", round(np.mean(inv_pred), 2))
    mlflow.log_param("price_prediction_bias", round(np.mean(inv_pred - inv_true), 2))
    
    # ---------- 6. Create comprehensive visualizations ---------- #
    print("Creating comprehensive visualizations...")
    
    # 1. Training history plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(training_history['train_loss']) + 1)
    ax1.plot(epochs, training_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.set_title('Training Loss Over Time', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, training_history['val_mae'], 'r-', label='Validation MAE', linewidth=2)
    ax2.set_title('Validation MAE Over Time', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(epochs, training_history['epoch_time'], 'g-', label='Epoch Time', linewidth=2)
    ax3.set_title('Training Time per Epoch', fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Time (seconds)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Metrics summary
    metrics_names = ['RMSE', 'MAE', 'MAPE (%)', 'R²', 'Dir. Acc. (%)']
    metrics_values = [price_rmse, price_mae, price_mape, price_r2, direction_accuracy]
    bars = ax4.bar(metrics_names, metrics_values, color=['skyblue', 'lightcoral', 'lightgreen', 'orange', 'purple'])
    ax4.set_title('Test Performance Metrics', fontweight='bold')
    ax4.set_ylabel('Value')
    plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig("training_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact("training_analysis.png")
    
    # 2. Prediction accuracy visualization
    test_dates = df.index[-len(inv_true):]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Actual vs Predicted
    ax1.plot(test_dates, inv_true, label='Actual', color='blue', linewidth=2, alpha=0.8)
    ax1.plot(test_dates, inv_pred, label='Predicted', color='red', linewidth=2, alpha=0.8)
    ax1.set_title(f'Stock Price Prediction - MAPE: {price_mape:.2f}%', fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.get_xticklabels(), rotation=45)
    
    # Prediction errors
    errors = inv_true - inv_pred
    ax2.plot(test_dates, errors, color='green', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.fill_between(test_dates, errors, alpha=0.3, color='green')
    ax2.set_title(f'Prediction Errors - MAE: ${price_mae:.2f}', fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Error ($)')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=45)
    
    # Percentage errors
    pct_errors = (inv_true - inv_pred) / inv_true * 100
    ax3.plot(test_dates, pct_errors, color='purple', linewidth=1.5, alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax3.fill_between(test_dates, pct_errors, alpha=0.3, color='purple')
    ax3.set_title(f'Percentage Errors - Avg: {np.mean(np.abs(pct_errors)):.2f}%', fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Error (%)')
    ax3.grid(True, alpha=0.3)
    plt.setp(ax3.get_xticklabels(), rotation=45)
    
    # Scatter plot
    ax4.scatter(inv_true, inv_pred, alpha=0.6, color='darkblue')
    ax4.plot([inv_true.min(), inv_true.max()], [inv_true.min(), inv_true.max()], 'r--', lw=2)
    ax4.set_xlabel('Actual Price ($)')
    ax4.set_ylabel('Predicted Price ($)')
    ax4.set_title(f'Actual vs Predicted - R²: {price_r2:.3f}', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("prediction_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact("prediction_analysis.png")
    
    # 3. Save detailed predictions with comprehensive info
    test_dates = df.index[-len(inv_true):]
    detailed_predictions = pd.DataFrame({
        'date': test_dates,
        'actual_close': inv_true,
        'predicted_close': inv_pred,
        'absolute_error': np.abs(inv_true - inv_pred),
        'percentage_error': np.abs((inv_true - inv_pred) / inv_true * 100),
        'prediction_within_1pct': np.abs((inv_true - inv_pred) / inv_true) <= 0.01,
        'prediction_within_2pct': np.abs((inv_true - inv_pred) / inv_true) <= 0.02,
        'prediction_within_5pct': np.abs((inv_true - inv_pred) / inv_true) <= 0.05
    })
    
    detailed_predictions.to_csv("detailed_predictions.csv", index=False)
    detailed_predictions.to_parquet("detailed_predictions.parquet", index=False)
    mlflow.log_artifact("detailed_predictions.csv")
    mlflow.log_artifact("detailed_predictions.parquet")
    
    # 4. Create and log model summary
    model_summary = {
        'model_architecture': f"LSTM({len(feature_cols)}, {args.hidden}, {args.layers})",
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'training_time': sum(training_history['epoch_time']),
        'best_epoch': mlflow.get_run(run.info.run_id).data.params.get('best_epoch', 'N/A'),
        'final_performance': {
            'price_mape': price_mape,
            'direction_accuracy': direction_accuracy,
            'r2_score': price_r2,
            'accuracy_within_2pct': threshold_2pct
        }
    }
    
    with open("model_summary.txt", "w") as f:
        f.write("LSTM Stock Price Prediction Model Summary\n")
        f.write("=" * 50 + "\n\n")
        for key, value in model_summary.items():
            f.write(f"{key}: {value}\n")
    
    mlflow.log_artifact("model_summary.txt")
    
    # Log model artifacts
    model_data = {
        "model_state": model.state_dict(),
        "scaler": scaler,
        "feature_cols": feature_cols,
        "seq_len": seq,
        "model_config": vars(args),
        "performance_metrics": comprehensive_metrics
    }
    torch.save(model_data, "complete_model.pt")
    mlflow.log_artifact("complete_model.pt")
    
    # Log model with MLflow
    try:
        input_example = X[val_end:val_end+1]
        mlflow.pytorch.log_model(
            model, 
            artifact_path="lstm-model",
            input_example=input_example
        )
        print("Model logged successfully to MLflow")
    except Exception as e:
        print(f"Warning: Could not log model to MLflow: {e}")

    print("\n" + "="*60)
    print("COMPREHENSIVE TRAINING RESULTS")
    print("="*60)
    print(f"Model Performance Score: {comprehensive_metrics['performance_score']:.2f}")
    print(f"Price MAPE: {price_mape:.2f}%")
    print(f"Direction Accuracy: {direction_accuracy:.2f}%")
    print(f"R² Score: {price_r2:.3f}")
    print(f"Predictions within 2%: {threshold_2pct:.1f}%")
    print(f"Mean Absolute Error: ${price_mae:.2f}")
    print("="*60)

print("\nTraining complete!")
print(f"MLflow tracking directory: {mlflow_dir}")
print(f"Check if directory exists: {os.path.exists(mlflow_dir)}")
print(f"Start MLflow UI with: cd {os.getcwd()} && mlflow ui --port 5000")
print(f"Then visit: http://localhost:5000")
