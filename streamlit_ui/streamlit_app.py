# streamlit_app.py
"""
Streamlit UI for train_lstm_sp500.py
------------------------------------
▪ Lets you pick / upload a dataset, choose symbols & hyper‑parameters
▪ Calls the training script via subprocess and streams logs live
▪ After completion, surfaces MLflow run metadata, metrics, plots & CSV/Parquet
"""

import json, os, subprocess, sys, tempfile, uuid
from pathlib import Path
import pandas as pd
import streamlit as st
import mlflow
import plotly.express as px

# ────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ────────────────────────────────────────────────────────────────────────────
SCRIPT_PATH = Path(__file__).parent.parent / "ML_LSTM" / "train_lstm_sp500.py"
DEFAULT_PARQUET = Path(__file__).parent.parent / "ML_LSTM" / "stocks_data.parquet"
MLFLOW_URI = f"file://{(Path(__file__).parent.parent / 'mlruns').resolve()}"

# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR – EXPERIMENT CONFIG
# ────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️  Experiment Settings")

# 1️⃣  Data source ───────────────────────────────────────────────
source = st.sidebar.radio("Data source", ("Sample parquet", "Upload file"))
if source == "Upload file":
    user_file = st.sidebar.file_uploader("Parquet or CSV", ["parquet", "csv"])
    if user_file:
        tmp = Path(tempfile.gettempdir()) / f"{uuid.uuid4()}_{user_file.name}"
        tmp.write_bytes(user_file.read())
        data_path = tmp
    else:
        st.stop()
else:
    data_path = DEFAULT_PARQUET
    if not Path(data_path).exists():
        st.error("Sample parquet not found – put one in ./samples/")
        st.stop()

# 2️⃣  Symbols & hyper‑parameters ───────────────────────────────
symbols = st.sidebar.text_input("Stock Ticker", "AAPL")
seq_len  = st.sidebar.number_input("Sequence length", 10, 200, 40, 5)
epochs   = st.sidebar.number_input("Epochs", 1, 300, 50, 5)
hidden   = st.sidebar.number_input("Hidden units", 16, 512, 128, 16)
layers   = st.sidebar.slider("LSTM layers", 1, 4, 2)
dropout  = st.sidebar.slider("Drop‑out", 0.0, 0.9, 0.3, 0.05)
batch    = st.sidebar.number_input("Batch size", 8, 512, 64, 8)
lr       = st.sidebar.select_slider("Learning rate",
                                    options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                                    value=1e-3)
patience = st.sidebar.number_input("Early‑stop patience", 1, 30, 5)

# ────────────────────────────────────────────────────────────────────────────
# MAIN AREA
# ────────────────────────────────────────────────────────────────────────────
st.title("📈 Streamlit UI – S&P‑500 LSTM Forecaster")

st.markdown(
"""Choose parameters in the sidebar and click **_Train model_**.  
Progress and logs appear in real time. When training finishes you’ll see:
* Key metrics & hyper‑parameters
* Plots and prediction files logged to MLflow
* Direct links to the full MLflow run
"""
)

# ────────────────────────────────────────────────────────────────────────────
# LAUNCH TRAINING
# ────────────────────────────────────────────────────────────────────────────
if st.button("🚀 Train model"):
    if not SCRIPT_PATH.exists():
        st.error(f"Could not find training script at {SCRIPT_PATH}")
        st.stop()

    # --- assemble CLI args --------------------------------------------------
    sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
    cli = [
        sys.executable, str(SCRIPT_PATH),
        "--file", str(data_path),
        "--symbols", *sym_list,
        "--seq_len", str(seq_len),
        "--epochs",  str(epochs),
        "--hidden",  str(hidden),
        "--layers",  str(layers),
        "--dropout", str(dropout),
        "--batch",   str(batch),
        "--lr",      str(lr),
        "--patience", str(patience),
    ]

    # --- run training and collect logs --------------------------------------
    log_text = ""
    proc = subprocess.Popen(cli, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  text=True)
    for line in proc.stdout:
        log_text += line
    proc.wait()

    # --- handle result ------------------------------------------------------
    if proc.returncode != 0:
        st.error("💥 Training script exited with error – see logs below.")
    else:
        st.success("✅ Training finished!")

    # ────────────────────────────────────────────────────────────────────
    #  FETCH & DISPLAY LATEST MLFLOW RUN
    # ────────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name("sp500-lstm-comprehensive")
    if exp is None:
        st.error("No MLflow experiment found – did the script log correctly?")
        st.stop()

    run = client.search_runs(exp.experiment_id,
                             order_by=["attributes.start_time DESC"],
                             max_results=1)[0]
    run_id = run.info.run_id
    st.markdown(f"### 🗂 Latest MLflow run – ID `{run_id}`")

    # --- metrics table --------------------------------------------------
    metrics_df = (pd.DataFrame(run.data.metrics.items(),
                               columns=["metric", "value"])
                  .sort_values("metric"))
    st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    # --- params accordion ------------------------------------------------
    with st.expander("📋 Hyper‑parameters / Run Params"):
        st.json(run.data.params, expanded=False)

    # --- show images & data artifacts -----------------------------------
    art_dir = Path(MLFLOW_URI.replace("file://", "")) / exp.experiment_id / run_id / "artifacts"
    img_exts = {".png", ".jpg", ".jpeg"}

    # --- Plotly chart for detailed_predictions.parquet if it exists ---
    detailed_parquet = art_dir / "detailed_predictions.parquet"
    if detailed_parquet.exists():
        st.subheader("📊 Interactive Actual vs Predicted Chart (from detailed_predictions.parquet)")
        df_detail = pd.read_parquet(detailed_parquet)
        if all(col in df_detail.columns for col in ["actual_close", "predicted_close", "date"]):
            fig = px.line(
                df_detail,
                x="date",
                y=["actual_close", "predicted_close"],
                labels={"value": "Price", "variable": "Legend"},
                title="Actual vs Predicted Close Prices (Detailed)"
            )
            fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    type="date"
                )
            )
            st.plotly_chart(fig, use_container_width=True)

    for p in art_dir.glob("**/*"):
        if p.suffix.lower() in img_exts:
            st.image(str(p), caption=p.relative_to(art_dir.parent))
        elif p.name.endswith("_predictions.csv") or p.name == "detailed_predictions.csv":
            st.subheader(f"📈 {p.name}")
            df_pred = pd.read_csv(p)
            st.dataframe(df_pred.head(), use_container_width=True)
            # Plotly interactive chart: Actual vs Predicted
            if all(col in df_pred.columns for col in ["actual_close", "pred_close", "date"]):
                fig = px.line(
                    df_pred,
                    x="date",
                    y=["actual_close", "pred_close"],
                    labels={"value": "Price", "variable": "Legend"},
                    title=f"Actual vs Predicted Close Prices: {p.stem.replace('_predictions','')}"
                )
                fig.update_layout(
                    xaxis=dict(
                        rangeslider=dict(visible=True),
                        type="date"
                    )
                )
                st.plotly_chart(fig, use_container_width=True)
            # Optionally, keep the single actual_close chart as well
            if "actual_close" in df_pred.columns and "date" in df_pred.columns:
                fig = px.line(
                    df_pred,
                    x="date",
                    y="actual_close",
                    title=f"Actual Close Prices: {p.stem.replace('_predictions','')}"
                )
                fig.update_layout(
                    xaxis=dict(
                        rangeslider=dict(visible=True),
                        type="date"
                    )
                )
                st.plotly_chart(fig, use_container_width=True)

    # --- Show logs at the bottom in an expander ---------------------------
    with st.expander("📝 Show Training Logs"):
        st.code(log_text, language="bash")

    st.markdown(
f"""➡ **Open full MLflow UI**  
```bash
mlflow ui --port 5000 --backend-store-uri "{MLFLOW_URI.replace('file://','')}"
and browse to http://localhost:5000
""")
