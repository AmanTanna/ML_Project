# LSTM Stock Price Predictor
A comprehensive machine learning project for time series forecasting of stock prices using LSTM neural networks. The project includes data preprocessing, model training, experiment tracking with MLflow, and an interactive Streamlit UI for visualization and analysis.

---

## 📁 Project Structure

```
ML_Project/
│
├── ML_LSTM/
│   ├── train_lstm_sp500.py        # Main training script for LSTM model
│   ├── stocks_data.parquet        # Sample stock data (parquet format)
│   └── ...                       # Other scripts and data files
│
├── streamlit_ui/
│   └── streamlit_app.py           # Streamlit UI for running experiments and viewing results
│
├── mlruns/                        # MLflow tracking directory (auto-generated)
│
├── README.md                      # Project documentation
└── requirements.txt               # Python dependencies
```

---

## 🚀 Features

- **LSTM-based stock price prediction** for one or more tickers
- **Flexible hyperparameter configuration** via Streamlit UI
- **Experiment tracking** with MLflow (metrics, parameters, artifacts)
- **Interactive visualizations** (Plotly) for actual vs. predicted prices
- **Per-symbol CSV and Parquet outputs** for predictions
- **Easy-to-use Streamlit interface** for running and analyzing experiments

---

## ⚙️ Setup

### 1. Clone the repository

```sh
git clone <your-repo-url>
cd ML_Project
```

### 2. Create and activate a virtual environment

```sh
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```sh
pip install -r requirements.txt
```

### 4. Prepare data

- Place your stock data in `ML_LSTM/stocks_data.parquet` or use the provided sample.
- You can also upload your own CSV/Parquet file via the Streamlit UI.

---

## 🏃‍♂️ Usage

### 1. **Start the Streamlit UI**

- From the project root
```sh
streamlit run streamlit_ui/streamlit_app.py
```

- Configure experiment parameters in the sidebar.
- Choose a data source (sample or upload).
- Click "Train model" to start a new experiment.
- View metrics, interactive charts, and logs in the UI.

### 2. **View MLflow UI (optional)**

From the project root:

```sh
mlflow ui --backend-store-uri ./mlruns
```

Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 📊 Outputs

- **MLflow**: All runs, metrics, parameters, and artifacts are tracked in the `mlruns/` directory.
- **Artifacts**: For each run, you get:
  - Interactive Plotly charts (actual vs. predicted)
  - Per-symbol CSV and Parquet files with predictions
  - Training logs and metrics

---

## 📝 Customization

- **Model & Features**: Edit `ML_LSTM/train_lstm_sp500.py` to change model architecture or feature engineering.
- **UI**: Edit `streamlit_ui/streamlit_app.py` to customize the Streamlit interface or add new visualizations.

---

## 🛠️ Troubleshooting

- **No runs in MLflow UI**: Ensure both the training script and Streamlit UI use the same `MLFLOW_URI` path.
- **File not found errors**: Check that all paths are correct relative to your working directory.
- **Dependency issues**: Reinstall requirements or recreate your virtual environment.

---

## 📚 Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies (including `streamlit`, `mlflow`, `pandas`, `plotly`, `torch`, etc.)

---

## 📄 License

MIT License

---

## 🙏 Acknowledgements

- [Streamlit](https://streamlit.io/)
- [MLflow](https://mlflow.org/)
- [PyTorch](https://pytorch.org/)
- [Plotly](https://plotly.com/python/)

---

## ✨ Contributions

Contributors:
Aman Tanna
Param Shah