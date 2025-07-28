#!/usr/bin/env python3
"""
Part 7: Complete System Launcher
================================

This script launches the complete Financial AI Analytics Suite:
1. API Server (RAG + OpenAI Integration)
2. Streamlit Dashboard
3. Health monitoring
4. All integrated components
"""

import os
import sys
import time
import subprocess
import signal
import threading
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

class SystemLauncher:
    """Manages launching and monitoring all system components"""
    
    def __init__(self):
        self.processes = {}
        self.project_root = Path(__file__).parent
        self.running = True
        
    def log(self, message, level="INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
    
    def start_api_server(self):
        """Start the RAG API server"""
        self.log("🚀 Starting RAG API Server...")
        
        try:
            # Change to project directory
            os.chdir(self.project_root)
            
            # Start the API server
            process = subprocess.Popen([
                sys.executable, "start_api_server.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            self.processes["api_server"] = process
            self.log("✅ API Server started successfully")
            
            # Wait a moment for server to initialize
            time.sleep(3)
            
            return True
            
        except Exception as e:
            self.log(f"❌ Failed to start API server: {e}", "ERROR")
            return False
    
    def start_streamlit_app(self):
        """Start the Streamlit dashboard"""
        self.log("🎨 Starting Streamlit Dashboard...")
        
        try:
            # Start Streamlit app
            process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", 
                "streamlit_ui/part7_app.py",
                "--server.port", "8501",
                "--server.headless", "true",
                "--browser.gatherUsageStats", "false"
            ], cwd=self.project_root)
            
            self.processes["streamlit"] = process
            self.log("✅ Streamlit Dashboard started successfully")
            
            return True
            
        except Exception as e:
            self.log(f"❌ Failed to start Streamlit: {e}", "ERROR")
            return False
    
    def check_health(self):
        """Check health of all services"""
        import requests
        
        services = {
            "API Server": "http://localhost:8000/api/v1/info",
            "Streamlit": "http://localhost:8501"
        }
        
        for service, url in services.items():
            try:
                response = requests.get(url, timeout=5)
                status = "🟢 Healthy" if response.status_code == 200 else f"🟡 Status {response.status_code}"
            except:
                status = "🔴 Unhealthy"
            
            self.log(f"{service}: {status}")
    
    def monitor_processes(self):
        """Monitor all processes and restart if needed"""
        while self.running:
            time.sleep(10)  # Check every 10 seconds
            
            for name, process in self.processes.items():
                if process.poll() is not None:  # Process has terminated
                    self.log(f"⚠️ {name} process terminated, restarting...", "WARNING")
                    
                    if name == "api_server":
                        self.start_api_server()
                    elif name == "streamlit":
                        self.start_streamlit_app()
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.log("🛑 Shutdown signal received, stopping all services...")
        self.shutdown()
    
    def shutdown(self):
        """Shutdown all services gracefully"""
        self.running = False
        
        for name, process in self.processes.items():
            if process.poll() is None:  # Process is still running
                self.log(f"🛑 Stopping {name}...")
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.log(f"⚡ Force killing {name}...")
                    process.kill()
        
        self.log("✅ All services stopped")
    
    def launch(self):
        """Launch the complete system"""
        self.log("🏁 Starting Financial AI Analytics Suite - Part 7")
        self.log("=" * 60)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Start services
        success = True
        
        # 1. Start API Server
        if not self.start_api_server():
            success = False
        
        # 2. Start Streamlit Dashboard
        if success and not self.start_streamlit_app():
            success = False
        
        if not success:
            self.log("❌ Failed to start some services", "ERROR")
            self.shutdown()
            return False
        
        # Wait for services to be fully ready
        self.log("⏳ Waiting for services to initialize...")
        time.sleep(5)
        
        # Check health
        self.log("🏥 Checking service health...")
        self.check_health()
        
        # Print access information
        self.log("=" * 60)
        self.log("🎉 FINANCIAL AI ANALYTICS SUITE IS READY!")
        self.log("=" * 60)
        self.log("🌐 Streamlit Dashboard: http://localhost:8501")
        self.log("🔗 API Server: http://localhost:8000")
        self.log("📚 API Documentation: http://localhost:8000/docs")
        self.log("=" * 60)
        self.log("Features Available:")
        self.log("  💬 RAG Chat Assistant with OpenAI Integration")
        self.log("  📊 LSTM Stock Prediction")
        self.log("  🔍 Advanced Document Search")
        self.log("  📈 Real-time System Analytics")
        self.log("  ⚙️ Comprehensive Settings & Configuration")
        self.log("=" * 60)
        self.log("Press Ctrl+C to stop all services")
        
        # Start monitoring in background
        monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
        monitor_thread.start()
        
        # Keep main thread alive
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.signal_handler(signal.SIGINT, None)
        
        return True

def main():
    """Main entry point"""
    launcher = SystemLauncher()
    
    try:
        launcher.launch()
    except Exception as e:
        launcher.log(f"💥 Fatal error: {e}", "ERROR")
        launcher.shutdown()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
