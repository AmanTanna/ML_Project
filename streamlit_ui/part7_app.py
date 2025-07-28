"""
Part 7: Advanced Analytics & Monitoring Dashboard
=================================================

Complete Streamlit UI integrating:
- LSTM Stock Prediction (existing)
- RAG Financial Document Analysis (new)
- API Server Monitoring (new)
- OpenAI Integration (new)
- Real-time Analytics (new)
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
import mlflow
from streamlit_autorefresh import st_autorefresh

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our modules
try:
    from RAG.layer5_api.models import QueryRequest, RAGChatRequest
    from RAG.layer5_api.auth import auth_manager
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    st.warning("RAG components not available. Some features will be disabled.")

# ────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_ENDPOINTS = {
    "health": f"{API_BASE_URL}/api/v1/health/",
    "info": f"{API_BASE_URL}/api/v1/info",
    "login": f"{API_BASE_URL}/api/v1/auth/login",
    "query": f"{API_BASE_URL}/api/v1/query/search",
    "openai_chat": f"{API_BASE_URL}/api/v1/openai/chat",
    "rag_chat": f"{API_BASE_URL}/api/v1/openai/rag-chat",
    "models": f"{API_BASE_URL}/api/v1/openai/models",
    "stats": f"{API_BASE_URL}/api/v1/health/stats"
}

# LSTM Configuration
SCRIPT_PATH = Path(__file__).parent.parent / "ML_LSTM" / "train_lstm_sp500.py"
DEFAULT_PARQUET = Path(__file__).parent.parent / "ML_LSTM" / "stocks_data.parquet"
MLFLOW_URI = f"file://{(Path(__file__).parent.parent / 'mlruns').resolve()}"

# ────────────────────────────────────────────────────────────────────────────
# STREAMLIT CONFIGURATION
# ────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Financial AI Analytics Suite",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────────────────────────────────────────────────
# API CLIENT FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────

class APIClient:
    """Client for interacting with the RAG API"""
    
    def __init__(self):
        self.token = None
        self.base_url = API_BASE_URL
        
    def login(self, username: str, password: str) -> bool:
        """Login to the API and store token"""
        try:
            response = requests.post(
                API_ENDPOINTS["login"],
                json={"username": username, "password": password},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                self.token = data.get("access_token")
                return True
        except Exception as e:
            st.error(f"Login failed: {e}")
        return False
    
    def get_headers(self):
        """Get headers with authorization"""
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
    
    def check_health(self) -> Dict:
        """Check API server health"""
        try:
            response = requests.get(API_ENDPOINTS["health"], timeout=5)
            if response.status_code in [200, 500]:  # 500 might have health info
                return response.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}
        return {"status": "unavailable"}
    
    def get_info(self) -> Dict:
        """Get API information"""
        try:
            response = requests.get(API_ENDPOINTS["info"], timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception:
            pass
        return {}
    
    def query_documents(self, query: str, query_type: str = "semantic") -> Dict:
        """Query documents using RAG"""
        try:
            response = requests.post(
                API_ENDPOINTS["query"],
                headers=self.get_headers(),
                json={
                    "query_text": query,
                    "query_type": query_type,
                    "max_results": 10,
                    "include_sources": True
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API returned {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def rag_chat(self, query: str, model: str = "gpt-3.5-turbo") -> Dict:
        """RAG-enhanced chat"""
        try:
            response = requests.post(
                API_ENDPOINTS["rag_chat"],
                headers=self.get_headers(),
                json={
                    "query": query,
                    "model": model,
                    "temperature": 0.7,
                    "include_sources": True
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API returned {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def openai_chat(self, message: str, model: str = "gpt-3.5-turbo") -> Dict:
        """Direct OpenAI chat without RAG"""
        try:
            response = requests.post(
                API_ENDPOINTS["openai_chat"],
                headers=self.get_headers(),
                json={
                    "messages": [{"role": "user", "content": message}],
                    "model": model,
                    "temperature": 0.7
                },
                timeout=60
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API returned {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        try:
            response = requests.get(
                API_ENDPOINTS["stats"],
                headers=self.get_headers(),
                timeout=10
            )
            if response.status_code in [200, 500]:
                return response.json()
        except Exception as e:
            return {"error": str(e)}
        return {}

# Initialize API client
@st.cache_resource
def get_api_client():
    return APIClient()

api_client = get_api_client()

def check_api_health():
    """Check API server health and capabilities"""
    try:
        response = requests.get(API_ENDPOINTS["health"], timeout=5)
        if response.status_code in [200, 500]:  # 500 might contain useful error info
            data = response.json()
            return {
                "available": True,
                "status": data.get("status", "unknown"),
                "has_rag": False,  # Currently disabled due to import issues
                "has_openai": False,  # Currently has import issues
                "message": "API server is running in limited mode due to import issues",
                "issues": data.get("issues", [])
            }
    except Exception as e:
        pass
    
    return {"available": False, "has_rag": False, "has_openai": False, "message": "API server not responding"}

# Check API capabilities
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_api_status():
    return check_api_health()

api_status = get_api_status()

# ────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_company_info(symbol):
    """Fetch company information using yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'name': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'current_price': info.get('currentPrice', 0),
            'market_cap': info.get('marketCap', 0),
            'description': info.get('longBusinessSummary', 'N/A')[:200] + "..."
        }
    except Exception:
        return {'name': 'Unknown Company', 'sector': 'N/A', 'industry': 'N/A'}

def check_api_server():
    """Check if API server is running"""
    try:
        response = requests.get(API_ENDPOINTS["info"], timeout=2)
        return response.status_code == 200
    except:
        return False

def start_api_server_background():
    """Start API server in background"""
    try:
        subprocess.Popen([
            sys.executable, "start_api_server.py"
        ], cwd=Path(__file__).parent.parent)
        return True
    except:
        return False

# ────────────────────────────────────────────────────────────────────────────
# PAGE COMPONENTS
# ────────────────────────────────────────────────────────────────────────────

def render_sidebar():
    """Render sidebar navigation"""
    st.sidebar.title("🏦 Financial AI Suite")
    
    # API Server Status
    with st.sidebar.expander("🔧 API Server Status", expanded=True):
        if check_api_server():
            st.success("✅ API Server Online")
            health = api_client.check_health()
            if health.get("status") == "healthy":
                st.success("🟢 System Healthy")
            elif health.get("status") == "degraded":
                st.warning("🟡 System Degraded")
            else:
                st.error("🔴 System Issues")
        else:
            st.error("❌ API Server Offline")
            if st.button("🚀 Start API Server"):
                with st.spinner("Starting API server..."):
                    if start_api_server_background():
                        st.success("Server starting...")
                        st.rerun()
                    else:
                        st.error("Failed to start server")
    
    # Navigation
    page = st.sidebar.selectbox(
        "📋 Select Page",
        [
            "🏠 Dashboard",
            "💬 RAG Chat Assistant", 
            "📊 LSTM Stock Prediction",
            "🔍 Document Search",
            "📈 System Analytics",
            "⚙️ Settings"
        ]
    )
    
    return page.split(" ", 1)[1]  # Remove emoji

def render_dashboard():
    """Render main dashboard"""
    st.title("🏠 Financial AI Analytics Dashboard")
    
    # Auto-refresh every 30 seconds
    st_autorefresh(interval=30000, key="dashboard_refresh")
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        health = api_client.check_health()
        status = health.get("status", "unknown")
        color = {"healthy": "green", "degraded": "orange", "error": "red"}.get(status, "gray")
        st.metric("🏥 System Health", status.title(), delta=None)
    
    with col2:
        info = api_client.get_info()
        features = len(info.get("features", []))
        st.metric("⚡ Features", features, delta=None)
    
    with col3:
        # Get current stock price for demo
        try:
            ticker = yf.Ticker("AAPL")
            price = ticker.history(period="1d")["Close"].iloc[-1]
            st.metric("📈 AAPL Price", f"${price:.2f}", delta=None)
        except:
            st.metric("📈 Market", "Loading...", delta=None)
    
    with col4:
        st.metric("🤖 AI Models", "GPT-3.5", delta=None)
    
    # Main Content Sections
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🚀 Quick Actions")
        
        # Quick Chat
        with st.expander("💬 Ask AI Assistant", expanded=True):
            if not api_status["available"]:
                st.warning("🔌 API Server is not available")
            else:
                quick_query = st.text_input(
                    "Ask about financial data:",
                    placeholder="What are Apple's main revenue streams?" if api_status["has_rag"] else "Ask me anything about finance..."
                )
                if st.button("🔍 Search") and quick_query:
                    if not api_status["has_openai"] and not api_status["has_rag"]:
                        st.error("⚠️ AI chat features are currently unavailable due to API issues")
                    else:
                        with st.spinner("Searching..."):
                            # Use appropriate chat method based on RAG availability
                            if api_status["has_rag"]:
                                result = api_client.rag_chat(quick_query)
                            elif api_status["has_openai"]:
                                result = api_client.openai_chat(quick_query)
                            else:
                                result = {"error": "No AI services available"}
                                
                            if "error" in result:
                                st.error(f"Error: {result['error']}")
                            else:
                                st.success("✅ Response:")
                                st.write(result.get("response", "No response"))
        
        # Recent Activity (Mock data for demo)
        st.subheader("📊 Recent Activity")
        activity_data = pd.DataFrame({
            "Time": pd.date_range(start="2025-01-20", periods=5, freq="h"),
            "Action": ["RAG Query", "LSTM Training", "API Call", "Document Upload", "Chat Request"],
            "Status": ["✅ Success", "🔄 Running", "✅ Success", "✅ Success", "✅ Success"],
            "User": ["admin", "user", "admin", "user", "admin"]
        })
        st.dataframe(activity_data, use_container_width=True)
    
    with col2:
        st.subheader("🎯 System Overview")
        
        # API Endpoints Status
        with st.expander("🔗 API Endpoints"):
            endpoints_status = []
            for name, url in API_ENDPOINTS.items():
                try:
                    resp = requests.get(url, timeout=2)
                    status = "🟢" if resp.status_code in [200, 307] else "🔴"
                except:
                    status = "🔴"
                endpoints_status.append({"Endpoint": name, "Status": status})
            
            st.dataframe(pd.DataFrame(endpoints_status), use_container_width=True)
        
        # Quick Stats
        with st.expander("📈 Quick Stats"):
            stats = api_client.get_stats()
            if stats and "error" not in stats:
                st.write("**System Stats Available**")
                st.json(stats, expanded=False)
            else:
                st.info("Stats loading...")

def render_rag_chat():
    """Render RAG chat interface"""
    st.title("💬 RAG Chat Assistant")
    
    # Check API status first
    if not api_status["available"]:
        st.error("🔌 API Server is not available. Please ensure the API server is running.")
        st.info("To start the API server, run: `python start_api_server.py`")
        return
    
    if not api_status["has_rag"]:
        st.warning("⚠️ RAG functionality is limited")
        st.info("The API server is running in limited mode without RAG components.")
    
    if not api_status["has_openai"]:
        st.warning("⚠️ OpenAI functionality currently has issues") 
        st.info("OpenAI integration has import issues. Only LSTM and analytics features are available.")
    
    # Authentication
    if not api_client.token:
        st.warning("🔐 Please login to access the chat assistant")
        
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("Username", value="admin")
        with col2:
            password = st.text_input("Password", type="password", value="admin123")
        
        if st.button("🔑 Login"):
            if api_client.login(username, password):
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Login failed")
        return
    
    # Chat Interface
    st.success("🔑 Authenticated as admin")
    
    # Chat History
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for i, (user_msg, ai_msg, timestamp) in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(f"**You ({timestamp}):**")
            st.write(user_msg)
        
        with st.chat_message("assistant"):
            st.write(f"**AI Assistant:**")
            st.write(ai_msg)
    
    # Chat input
    user_input = st.chat_input("Ask me about financial documents...")
    
    if user_input:
        # Add user message
        timestamp = datetime.now().strftime("%H:%M")
        
        with st.chat_message("user"):
            st.write(f"**You ({timestamp}):**")
            st.write(user_input)
        
        # Get AI response
        with st.chat_message("assistant"):
            st.write("**AI Assistant:**")
            
            if not api_status["has_openai"] and not api_status["has_rag"]:
                ai_response = "I apologize, but AI chat features are currently unavailable due to API server issues. However, you can still use the LSTM Stock Prediction feature!"
                st.write(ai_response)
            else:
                with st.spinner("Thinking..."):
                    # Use RAG chat if available, otherwise use regular OpenAI chat
                    if api_status["has_rag"]:
                        result = api_client.rag_chat(user_input)
                    elif api_status["has_openai"]:
                        result = api_client.openai_chat(user_input)
                    else:
                        result = {"error": "No AI services available"}
                    
                    if "error" in result:
                        ai_response = f"I apologize, but I encountered an error: {result['error']}"
                    else:
                        ai_response = result.get("response", "I couldn't generate a response.")
                        
                        # Show sources if available (only for RAG)
                        if api_status["has_rag"]:
                            sources = result.get("sources", [])
                            if sources:
                                ai_response += "\\n\\n**Sources:**\\n"
                                for source in sources[:3]:  # Show top 3 sources
                                    ai_response += f"- {source.get('source_document', 'Unknown')}\\n"
                    
                    st.write(ai_response)
        
        # Add to chat history
        st.session_state.chat_history.append((user_input, ai_response, timestamp))
        
        # Keep only last 10 exchanges
        if len(st.session_state.chat_history) > 10:
            st.session_state.chat_history = st.session_state.chat_history[-10:]

def render_lstm_prediction():
    """Render full LSTM stock prediction interface with MLflow integration"""
    st.title("📊 LSTM Stock Prediction")
    
    # Helper functions
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

    # SIDEBAR – EXPERIMENT CONFIG
    st.sidebar.header("⚙️ LSTM Experiment Settings")
    
    # 1️⃣ Data source
    source = st.sidebar.radio("Data source", ("Sample parquet", "Upload file"))
    if source == "Upload file":
        user_file = st.sidebar.file_uploader("Parquet or CSV", ["parquet", "csv"])
        if user_file:
            import tempfile
            tmp = Path(tempfile.gettempdir()) / f"{uuid.uuid4().hex}_{user_file.name}"
            tmp.write_bytes(user_file.read())
            data_path = tmp
        else:
            st.sidebar.warning("Please upload a file to continue")
            data_path = DEFAULT_PARQUET
    else:
        data_path = DEFAULT_PARQUET
        if not Path(data_path).exists():
            st.error("Sample parquet not found")
            return
    
    # 2️⃣ Symbols & hyper-parameters
    symbols = st.sidebar.text_input("Stock Ticker", "AAPL")
    seq_len = st.sidebar.number_input("Sequence length", 10, 200, 40, 5)
    epochs = st.sidebar.number_input("Epochs", 1, 300, 50, 5)
    hidden = st.sidebar.number_input("Hidden units", 16, 512, 128, 16)
    layers = st.sidebar.slider("LSTM layers", 1, 4, 2)
    dropout = st.sidebar.slider("Drop-out", 0.0, 0.9, 0.3, 0.05)
    batch = st.sidebar.number_input("Batch size", 8, 512, 64, 8)
    lr = st.sidebar.select_slider("Learning rate",
                                options=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                                value=1e-3)
    patience = st.sidebar.number_input("Early-stop patience", 1, 30, 5)
    
    # Main content
    st.markdown(
    """Choose parameters in the sidebar and click **_Train model_**.  
    Progress and logs appear in real time. When training finishes you'll see:
    * Key metrics & hyper-parameters
    * Plots and prediction files logged to MLflow
    * Direct links to the full MLflow run
    """
    )
    
    # Display company information if symbol is provided
    if symbols.strip():
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
        if len(symbol_list) == 1:
            company_info = get_company_info(symbol_list[0])
            with st.expander(f"📊 {symbol_list[0]} Company Information", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Company:** {company_info['name']}")
                    st.write(f"**Sector:** {company_info['sector']}")
                    st.write(f"**Industry:** {company_info['industry']}")
                with col2:
                    if company_info.get('current_price'):
                        st.metric("Current Price", f"${company_info['current_price']:.2f}")
                    if company_info.get('market_cap'):
                        st.metric("Market Cap", f"${company_info['market_cap']:,.0f}")
                
                if company_info.get('description') and company_info['description'] != 'N/A':
                    st.write("**Description:**")
                    st.write(company_info['description'])
        
        elif len(symbol_list) > 1:
            st.subheader("📊 Selected Stocks")
            for symbol in symbol_list:
                with st.expander(f"{symbol} - Company Info"):
                    company_info = get_company_info(symbol)
                    st.write(f"**Company:** {company_info['name']}")
                    st.write(f"**Sector:** {company_info['sector']}")
    
    # LAUNCH TRAINING
    if st.button("🚀 Train LSTM Model"):
        if not SCRIPT_PATH.exists():
            st.error(f"Could not find training script at {SCRIPT_PATH}")
            return
        
        # Assemble CLI args
        sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
        cli = [
            sys.executable, str(SCRIPT_PATH),
            "--file", str(data_path),
            "--symbols", *sym_list,
            "--seq_len", str(seq_len),
            "--epochs", str(epochs),
            "--hidden", str(hidden),
            "--layers", str(layers),
            "--dropout", str(dropout),
            "--batch", str(batch),
            "--lr", str(lr),
            "--patience", str(patience),
        ]
        
        st.info(f"🔧 **Command:** `{' '.join(cli)}`")
        
        # Create enhanced loading containers
        loading_container, progress_container, status_container = create_loading_animation()
        log_expander = st.expander("📝 View Training Logs", expanded=False)
        log_container = log_expander.empty()
        
        # Training phases
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
        
        # Run training and collect logs
        log_text = ""
        phase_idx = 0
        lines_processed = 0
        epoch_pattern = r"Epoch (\\d+)/(\\d+)"
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
            
            # Real-time log processing
            import re
            for line in proc.stdout:
                log_text += line
                lines_processed += 1
                
                # Check for epoch progress
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
            return
        
        # Clear loading animation
        loading_container.empty()
        progress_container.empty()
        status_container.empty()
        
        # Final log update
        log_container.code(log_text, language="bash")
        
        # Handle result
        if proc.returncode != 0:
            st.error("💥 Training script exited with error – see logs above.")
        else:
            # Success message
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
            
            # FETCH & DISPLAY LATEST MLFLOW RUN
            with st.spinner("📊 Loading MLflow results..."):
                try:
                    mlflow.set_tracking_uri(MLFLOW_URI)
                    client = mlflow.MlflowClient()
                    exp = client.get_experiment_by_name("sp500-lstm-comprehensive")
                    if exp is None:
                        st.error("No MLflow experiment found – did the script log correctly?")
                        return
                    
                    runs = client.search_runs(exp.experiment_id,
                                            order_by=["attributes.start_time DESC"],
                                            max_results=1)
                    if not runs:
                        st.error("No runs found in the experiment")
                        return
                    
                    run = runs[0]
                    run_id = run.info.run_id
                    st.markdown(f"### 🗂 Latest MLflow run – ID `{run_id}`")
                    
                    # Metrics table
                    if run.data.metrics:
                        metrics_df = (pd.DataFrame(run.data.metrics.items(),
                                                columns=["metric", "value"])
                                    .sort_values("metric"))
                        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
                    else:
                        st.warning("No metrics found in the run")
                    
                    # Params accordion
                    with st.expander("📋 Hyper-parameters / Run Params"):
                        if run.data.params:
                            st.json(run.data.params, expanded=False)
                        else:
                            st.warning("No parameters found in the run") 
                    
                    # Show images & data artifacts
                    art_dir = Path(MLFLOW_URI.replace("file://", "")) / exp.experiment_id / run_id / "artifacts"
                    
                    if not art_dir.exists():
                        st.warning(f"Artifacts directory not found: {art_dir}")
                    else:
                        img_exts = {".png", ".jpg", ".jpeg"}
                        
                        # Plotly chart for detailed_predictions.parquet if it exists
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

def render_document_search():
    """Render document search interface"""
    st.title("🔍 Document Search")
    
    # Check API status first
    if not api_status["available"]:
        st.error("🔌 API Server is not available. Please ensure the API server is running.")
        st.info("To start the API server, run: `python start_api_server.py`")
        return
    
    if not api_status["has_rag"]:
        st.error("⚠️ Document search is not available")
        st.info("The API server is running in limited mode without RAG components. Document search functionality requires the full RAG system to be operational.")
        
        # Show what's available instead
        st.markdown("### 💬 Available Features:")
        st.markdown("- **OpenAI Chat**: Direct access to GPT models")
        st.markdown("- **LSTM Stock Prediction**: Machine learning stock forecasting")
        st.markdown("- **System Analytics**: Dashboard and monitoring")
        return
    
    if not api_client.token:
        st.warning("🔐 Please login to access document search")
        return
    
    # Search Interface
    search_query = st.text_area(
        "Enter your search query:",
        placeholder="What are Apple's main business segments?",
        height=100
    )
    
    col1, col2, col3 = st.columns(3)
    with col1:
        query_type = st.selectbox(
            "Query Type",
            ["semantic", "structured", "raw", "hybrid"],
            index=0
        )
    
    with col2:
        max_results = st.slider("Max Results", 1, 20, 5)
    
    with col3:
        include_sources = st.checkbox("Include Sources", value=True)
    
    if st.button("🔍 Search Documents") and search_query:
        with st.spinner("Searching documents..."):
            result = api_client.query_documents(
                search_query, 
                query_type=query_type
            )
            
            if "error" in result:
                st.error(f"Search failed: {result['error']}")
            else:
                st.success(f"✅ Found {result.get('total_results', 0)} results")
                
                # Display results
                results = result.get('results', [])
                for i, res in enumerate(results):
                    with st.expander(f"Result {i+1} - Score: {res.get('relevance_score', 0):.3f}"):
                        st.write("**Content:**")
                        st.write(res.get('content', ''))
                        
                        if include_sources and res.get('source_document'):
                            st.write(f"**Source:** {res.get('source_document')}")
                        
                        if res.get('company'):
                            st.write(f"**Company:** {res.get('company')}")

def render_system_analytics():
    """Render system analytics and monitoring"""
    st.title("📈 System Analytics & Monitoring")
    
    # Auto-refresh every 10 seconds
    st_autorefresh(interval=10000, key="analytics_refresh")
    
    # Get system stats
    stats = api_client.get_stats()
    
    if "error" in stats:
        st.error(f"Unable to load analytics: {stats['error']}")
        return
    
    # Overview Metrics
    st.subheader("📊 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🚀 System Status", "Operational")
    
    with col2:
        st.metric("⚡ Active Sessions", "3")
    
    with col3:
        st.metric("📈 Queries Today", "127")
    
    with col4:
        st.metric("🔄 Uptime", "2h 34m")
    
    # Performance Charts
    st.subheader("📈 Performance Metrics")
    
    # Mock performance data
    time_range = pd.date_range(start="2025-01-27 17:00", periods=20, freq="5min")
    performance_data = pd.DataFrame({
        "Time": time_range,
        "Response Time (ms)": np.random.normal(200, 50, 20),
        "CPU Usage (%)": np.random.normal(45, 15, 20),
        "Memory Usage (%)": np.random.normal(60, 10, 20),
        "Active Requests": np.random.poisson(5, 20)
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(
            performance_data,
            x="Time",
            y="Response Time (ms)",
            title="API Response Time",
            color_discrete_sequence=["#1f77b4"]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.line(
            performance_data,
            x="Time",
            y=["CPU Usage (%)", "Memory Usage (%)"],
            title="System Resources",
            color_discrete_sequence=["#ff7f0e", "#2ca02c"]
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Activity Log
    st.subheader("📋 Recent Activity")
    
    # Mock activity data
    activity_log = pd.DataFrame({
        "Timestamp": pd.date_range(start="2025-01-27 19:00", periods=10, freq="-5min"),
        "Event": np.random.choice([
            "RAG Query Processed", 
            "User Login", 
            "Document Indexed", 
            "API Health Check",
            "LSTM Model Training",
            "OpenAI API Call"
        ], 10),
        "Status": np.random.choice(["✅ Success", "⚠️ Warning", "❌ Error"], 10, p=[0.8, 0.15, 0.05]),
        "Duration": np.random.normal(150, 50, 10).astype(int).astype(str) + "ms"
    })
    
    st.dataframe(activity_log, use_container_width=True)

def render_settings():
    """Render settings page"""
    st.title("⚙️ Settings & Configuration")
    
    # API Configuration
    st.subheader("🔧 API Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("API Base URL", value=API_BASE_URL, disabled=True)
    with col2:
        st.selectbox("Default Model", ["gpt-3.5-turbo", "gpt-4"], index=0)
    
    # System Settings
    st.subheader("⚙️ System Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        st.slider("Auto-refresh Interval (seconds)", 5, 60, 30)
        st.checkbox("Enable Notifications", value=True)
    
    with col2:
        st.slider("Max Chat History", 5, 50, 10)
        st.checkbox("Dark Mode", value=False)
    
    # Status Information
    st.subheader("ℹ️ System Information")
    
    info = api_client.get_info()
    if info:
        st.json(info, expanded=False)
    
    # Test Connections
    st.subheader("🔍 Connection Tests")
    
    if st.button("🧪 Test API Connection"):
        with st.spinner("Testing connections..."):
            results = {}
            for name, url in API_ENDPOINTS.items():
                try:
                    resp = requests.get(url, timeout=5)
                    results[name] = f"✅ {resp.status_code}"
                except Exception as e:
                    results[name] = f"❌ {str(e)[:50]}"
            
            for name, status in results.items():
                st.write(f"**{name}:** {status}")

# ────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ────────────────────────────────────────────────────────────────────────────

def main():
    """Main Streamlit application"""
    
    # Render sidebar and get selected page
    selected_page = render_sidebar()
    
    # Route to appropriate page
    if selected_page == "Dashboard":
        render_dashboard()
    elif selected_page == "RAG Chat Assistant":
        render_rag_chat()
    elif selected_page == "LSTM Stock Prediction":
        render_lstm_prediction()
    elif selected_page == "Document Search":
        render_document_search()
    elif selected_page == "System Analytics":
        render_system_analytics()
    elif selected_page == "Settings":
        render_settings()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "🏦 **Financial AI Analytics Suite** | "
        "Part 7: Advanced Analytics & Monitoring | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

if __name__ == "__main__":
    main()
