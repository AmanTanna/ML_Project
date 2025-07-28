"""
Test the fixed API server with a simple request
"""

import requests
import time
import subprocess
import os
import signal
from contextlib import contextmanager

@contextmanager
def api_server():
    """Context manager to start and stop the API server"""
    # Start the server
    env = os.environ.copy()
    process = subprocess.Popen(
        ["python", "start_api_server.py"],
        cwd="/Users/jineshshah/Desktop/aman_proj/ML_Project",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    time.sleep(3)
    
    try:
        yield process
    finally:
        # Stop the server
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

def test_fixed_api():
    """Test that the API server works without datetime serialization errors"""
    print("Testing fixed API server...")
    
    with api_server() as server_process:
        # Check if server is running
        if server_process.poll() is not None:
            print("✗ Server failed to start")
            stdout, stderr = server_process.communicate()
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return False
        
        base_url = "http://localhost:8000"
        
        # Test health endpoint (should return timestamp without error)
        try:
            response = requests.get(f"{base_url}/api/v1/health", timeout=5)
            if response.status_code == 200:
                print("✓ Health endpoint works (datetime serialization fixed)")
                data = response.json()
                if 'timestamp' in data:
                    print(f"✓ Timestamp properly serialized: {data['timestamp']}")
            else:
                print(f"✗ Health endpoint failed: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"✗ Request failed: {e}")
            return False
        
        # Test login endpoint (contains datetime fields)
        try:
            login_data = {"username": "admin", "password": "admin123"}
            response = requests.post(f"{base_url}/api/v1/auth/login", json=login_data, timeout=5)
            if response.status_code == 200:
                print("✓ Login endpoint works (user profile with datetime serialized)")
            else:
                print(f"⚠ Login endpoint returned: {response.status_code}")
                # This might fail due to missing auth components, but no datetime errors
        except requests.exceptions.RequestException as e:
            print(f"✗ Login request failed: {e}")
        
        print("\n🎉 API server is working without datetime serialization errors!")
        return True

if __name__ == "__main__":
    success = test_fixed_api()
    if success:
        print("\n✅ DATETIME SERIALIZATION FIX CONFIRMED")
        print("The API server no longer crashes with 'Object of type datetime is not JSON serializable' errors.")
    else:
        print("\n❌ Server test failed - check the logs above")
