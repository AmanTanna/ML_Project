# streamlit_app.py
"""
Streamlit UI for train_lstm_sp500.py
------------------------------------
▪ Lets you pick / upload a dataset, choose symbols & hyper‑parameters
▪ Calls the training script via subprocess and streams logs live
▪ After completion, surfaces MLflow run metadata, metrics, plots & CSV/Parquet
"""

import json, os, subprocess, sys, tempfile, uuid, time
from pathlib import Path
import pandas as pd
import streamlit as st
import mlflow
import plotly.express as px
import yfinance as yf

# ────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ────────────────────────────────────────────────────────────────────────────
SCRIPT_PATH = Path(__file__).parent.parent / "ML_LSTM" / "train_lstm_sp500.py"
DEFAULT_PARQUET = Path(__file__).parent.parent / "ML_LSTM" / "stocks_data.parquet"
MLFLOW_URI = f"file://{(Path(__file__).parent.parent / 'mlruns').resolve()}"

# ────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_company_info(symbol):
    """Fetch company information using yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Extract relevant information
        company_data = {
            'name': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'country': info.get('country', 'N/A'),
            'website': info.get('website', 'N/A'),
            'market_cap': info.get('marketCap', 0),
            'employees': info.get('fullTimeEmployees', 'N/A'),
            'description': info.get('longBusinessSummary', 'N/A'),
            'current_price': info.get('currentPrice', 0),
            'previous_close': info.get('previousClose', 0),
            'day_high': info.get('dayHigh', 0),
            'day_low': info.get('dayLow', 0),
            'volume': info.get('volume', 0),
            'avg_volume': info.get('averageVolume', 0),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'dividend_yield': info.get('dividendYield', 'N/A'),
            'beta': info.get('beta', 'N/A'),
            '52_week_high': info.get('fiftyTwoWeekHigh', 0),
            '52_week_low': info.get('fiftyTwoWeekLow', 0),
        }
        
        return company_data
    except Exception as e:
        st.error(f"Error fetching company info for {symbol}: {e}")
        return None

def format_number(num):
    """Format large numbers for display"""
    if pd.isna(num) or num == 0:
        return "N/A"
    
    if num >= 1e12:
        return f"${num/1e12:.2f}T"
    elif num >= 1e9:
        return f"${num/1e9:.2f}B"
    elif num >= 1e6:
        return f"${num/1e6:.2f}M"
    elif num >= 1e3:
        return f"${num/1e3:.2f}K"
    else:
        return f"${num:.2f}"

def display_company_info(symbol):
    """Display company information in a nice format"""
    with st.spinner(f"Fetching company information for {symbol}..."):
        company_info = get_company_info(symbol)
    
    if company_info:
        st.subheader(f"📊 {company_info['name']} ({symbol})")
        
        # Company overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Price", f"${company_info['current_price']:.2f}")
            st.metric("Previous Close", f"${company_info['previous_close']:.2f}")
        
        with col2:
            st.metric("Market Cap", format_number(company_info['market_cap']))
            st.metric("P/E Ratio", f"{company_info['pe_ratio']:.2f}" if company_info['pe_ratio'] != 'N/A' else 'N/A')
        
        with col3:
            st.metric("Beta", f"{company_info['beta']:.2f}" if company_info['beta'] != 'N/A' else 'N/A')
            dividend_yield = company_info['dividend_yield']
            if dividend_yield != 'N/A' and dividend_yield is not None:
                st.metric("Dividend Yield", f"{dividend_yield*100:.2f}%")
            else:
                st.metric("Dividend Yield", "N/A")
        
        # Company details
        with st.expander("📋 Company Details"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Sector:** {company_info['sector']}")
                st.write(f"**Industry:** {company_info['industry']}")
                st.write(f"**Country:** {company_info['country']}")
                st.write(f"**Employees:** {company_info['employees']:,}" if company_info['employees'] != 'N/A' else "**Employees:** N/A")
            
            with col2:
                st.write(f"**52 Week High:** ${company_info['52_week_high']:.2f}")
                st.write(f"**52 Week Low:** ${company_info['52_week_low']:.2f}")
                if company_info['website'] != 'N/A':
                    st.write(f"**Website:** [Link]({company_info['website']})")
            
            if company_info['description'] != 'N/A' and len(company_info['description']) > 0:
                st.write("**Business Summary:**")
                st.write(company_info['description'][:500] + "..." if len(company_info['description']) > 500 else company_info['description'])

def create_loading_animation():
    """Create a spinning wheel loading animation"""
    return st.empty(), st.empty(), st.empty()

def update_loading_display(loading_container, progress_container, status_container, message, progress=None):
    """Update the loading display with spinning wheel and message"""
    # Spinning wheel using CSS animation
    loading_html = f"""
    <div style="display: flex; align-items: center; justify-content: center; padding: 20px;">
        <div style="
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin-right: 15px;
        "></div>
        <div style="
            font-size: 18px;
            color: #3498db;
            font-weight: bold;
        ">{message}</div>
    </div>
    <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
    """
    
    loading_container.markdown(loading_html, unsafe_allow_html=True)
    
    if progress is not None:
        progress_container.progress(progress)

# ────────────────────────────────────────────────────────────────────────────
# SIDEBAR – EXPERIMENT CONFIG
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
        st.error("Sample parquet not found – put one in ./samples/")
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
st.title("📈 Streamlit UI – S&P‑500 LSTM Forecaster")

st.markdown(
"""Choose parameters in the sidebar and click **_Train model_**.  
Progress and logs appear in real time. When training finishes you'll see:
* Key metrics & hyper‑parameters
* Plots and prediction files logged to MLflow
* Direct links to the full MLflow run
"""
)

# Display company information if symbol is provided
if symbols.strip():
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if len(symbol_list) == 1:
        display_company_info(symbol_list[0])
    elif len(symbol_list) > 1:
        st.subheader("📊 Selected Stocks")
        for symbol in symbol_list:
            with st.expander(f"{symbol} - Company Info"):
                display_company_info(symbol)

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

    st.info(f"🔧 **Command:** `{' '.join(cli)}`")

    # Create enhanced loading containers
    loading_container, progress_container, status_container = create_loading_animation()
    log_expander = st.expander("📝 View Training Logs", expanded=False)
    log_container = log_expander.empty()

    # Training phases for better UX
    training_phases = [
        "🚀 Initializing training environment...",
        "📊 Loading and validating dataset...",
        "🔧 Preprocessing stock data...",
        "🧠 Building LSTM neural network...",
        "⚡ Training model on historical data...",
        "📈 Generating predictions...",
        "💾 Saving model checkpoints...",
        "📋 Logging results to MLflow...",
        "🎯 Evaluating model performance...",
        "✨ Finalizing training process..."
    ]

    # --- run training and collect logs --------------------------------------
    log_text = ""
    phase_idx = 0
    lines_processed = 0
    epoch_pattern = r"Epoch (\d+)/(\d+)"
    current_epoch = 0
    total_epochs = epochs
    
    # Initial loading display
    update_loading_display(loading_container, progress_container, status_container, 
                          training_phases[0], 0.0)
    
    try:
        proc = subprocess.Popen(cli, 
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              text=True, 
                              bufsize=1, 
                              universal_newlines=True)
        
        # Real-time log processing with enhanced loading experience
        for line in proc.stdout:
            log_text += line
            lines_processed += 1
            
            # Check for epoch progress
            import re
            epoch_match = re.search(epoch_pattern, line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                total_epochs = int(epoch_match.group(2))
                epoch_progress = current_epoch / total_epochs
                
                # Update loading display with epoch progress
                update_loading_display(
                    loading_container, progress_container, status_container,
                    f"⚡ Training model - Epoch {current_epoch}/{total_epochs}",
                    epoch_progress * 0.8  # 80% of progress bar for training
                )
            
            # Update phase based on log content
            elif "Loading data" in line or "Reading" in line:
                phase_idx = 1
                update_loading_display(loading_container, progress_container, status_container,
                                     training_phases[phase_idx], 0.1)
            
            elif "Preprocessing" in line or "Creating features" in line:
                phase_idx = 2
                update_loading_display(loading_container, progress_container, status_container,
                                     training_phases[phase_idx], 0.2)
            
            elif "Model created" in line or "LSTM" in line:
                phase_idx = 3
                update_loading_display(loading_container, progress_container, status_container,
                                     training_phases[phase_idx], 0.3)
            
            elif "Starting training" in line:
                phase_idx = 4
                update_loading_display(loading_container, progress_container, status_container,
                                     training_phases[phase_idx], 0.4)
            
            elif "Generating predictions" in line or "Making predictions" in line:
                phase_idx = 5
                update_loading_display(loading_container, progress_container, status_container,
                                     training_phases[phase_idx], 0.85)
            
            elif "Saving" in line:
                phase_idx = 6
                update_loading_display(loading_container, progress_container, status_container,
                                     training_phases[phase_idx], 0.9)
            
            elif "MLflow" in line or "Logging" in line:
                phase_idx = 7
                update_loading_display(loading_container, progress_container, status_container,
                                     training_phases[phase_idx], 0.95)
            
            # Update logs every 10 lines
            if lines_processed % 10 == 0:
                log_container.code(log_text[-2000:], language="bash")
            
            # Check for completion indicators
            if "Training finished" in line or "Model saved" in line or "Training complete" in line:
                update_loading_display(loading_container, progress_container, status_container,
                                     "✅ Training completed successfully!", 1.0)
                time.sleep(1)  # Brief pause to show completion
        
        proc.wait()
        
    except Exception as e:
        loading_container.error(f"❌ Error during training: {e}")
        st.stop()

    # Clear loading animation
    loading_container.empty()
    progress_container.empty()
    status_container.empty()

    # Final log update
    log_container.code(log_text, language="bash")

    # --- handle result ------------------------------------------------------
    if proc.returncode != 0:
        st.error("💥 Training script exited with error – see logs above.")
    else:
        # Success message with animation
        success_html = """
        <div style="
            display: flex; 
            align-items: center; 
            justify-content: center; 
            padding: 20px;
            background: linear-gradient(90deg, #4CAF50, #45a049);
            border-radius: 10px;
            color: white;
            font-size: 20px;
            font-weight: bold;
            margin: 20px 0;
            animation: fadeIn 0.5s ease-in;
        ">
            <span style="font-size: 30px; margin-right: 15px;">🎉</span>
            Training Completed Successfully!
            <span style="font-size: 30px; margin-left: 15px;">🎉</span>
        </div>
        <style>
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(-20px); }
                to { opacity: 1; transform: translateY(0); }
            }
        </style>
        """
        st.markdown(success_html, unsafe_allow_html=True)

        # ────────────────────────────────────────────────────────────────────
        #  FETCH & DISPLAY LATEST MLFLOW RUN
        # ────────────────────────────────────────────────────────────────────
        with st.spinner("📊 Loading MLflow results..."):
            try:
                mlflow.set_tracking_uri(MLFLOW_URI)
                client = mlflow.MlflowClient()
                exp = client.get_experiment_by_name("sp500-lstm-comprehensive")
                if exp is None:
                    st.error("No MLflow experiment found – did the script log correctly?")
                    st.stop()

                runs = client.search_runs(exp.experiment_id,
                                        order_by=["attributes.start_time DESC"],
                                        max_results=1)
                if not runs:
                    st.error("No runs found in the experiment")
                    st.stop()
                
                run = runs[0]
                run_id = run.info.run_id
                st.markdown(f"### 🗂 Latest MLflow run – ID `{run_id}`")

                # --- metrics table --------------------------------------------------
                if run.data.metrics:
                    metrics_df = (pd.DataFrame(run.data.metrics.items(),
                                            columns=["metric", "value"])
                                .sort_values("metric"))
                    st.dataframe(metrics_df, hide_index=True, use_container_width=True)
                else:
                    st.warning("No metrics found in the run")

                # --- params accordion ------------------------------------------------
                with st.expander("📋 Hyper‑parameters / Run Params"):
                    if run.data.params:
                        st.json(run.data.params, expanded=False)
                    else:
                        st.warning("No parameters found in the run")

                # --- show images & data artifacts -----------------------------------
                art_dir = Path(MLFLOW_URI.replace("file://", "")) / exp.experiment_id / run_id / "artifacts"
                
                if not art_dir.exists():
                    st.warning(f"Artifacts directory not found: {art_dir}")
                else:
                    img_exts = {".png", ".jpg", ".jpeg"}

                    # --- Plotly chart for detailed_predictions.parquet if it exists ---
                    detailed_parquet = art_dir / "detailed_predictions.parquet"
                    if detailed_parquet.exists():
                        st.subheader("📊 Interactive Actual vs Predicted Chart")
                        try:
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
                        except Exception as e:
                            st.error(f"Error loading detailed predictions: {e}")

                    # Show other artifacts
                    artifact_count = 0
                    for p in art_dir.glob("**/*"):
                        if p.is_file():
                            artifact_count += 1
                            if p.suffix.lower() in img_exts:
                                st.image(str(p), caption=p.name)
                            elif p.name.endswith("_predictions.csv") or p.name == "detailed_predictions.csv":
                                st.subheader(f"📈 {p.name}")
                                try:
                                    df_pred = pd.read_csv(p)
                                    st.dataframe(df_pred.head(), use_container_width=True)
                                    
                                    # Interactive chart
                                    if all(col in df_pred.columns for col in ["actual_close", "predicted_close", "date"]):
                                        fig = px.line(
                                            df_pred,
                                            x="date",
                                            y=["actual_close", "predicted_close"],
                                            labels={"value": "Price", "variable": "Legend"},
                                            title=f"Predictions from {p.name}"
                                        )
                                        fig.update_layout(
                                            xaxis=dict(
                                                rangeslider=dict(visible=True),
                                                type="date"
                                            )
                                        )
                                        st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error loading {p.name}: {e}")
                    
                    if artifact_count == 0:
                        st.warning("No artifacts found in the run")

            except Exception as e:
                st.error(f"Error accessing MLflow data: {e}")
                import traceback
                st.code(traceback.format_exc())

    # MLflow UI link
    st.markdown(
f"""➡ **Open full MLflow UI**  
```bash
mlflow ui --port 5000 --backend-store-uri "{MLFLOW_URI.replace('file://','')}"
```
Then browse to http://localhost:5000
""")
