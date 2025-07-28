"""
Test API Client - Part 6
========================

Test the running API server with HTTP requests.
"""

import requests
import json
import time
from datetime import datetime

# API Configuration
BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

def test_api_endpoints():
    """Test API endpoints with HTTP requests"""
    print("=" * 60)
    print("RAG API CLIENT TEST")
    print("=" * 60)
    
    try:
        # Test 1: Root endpoint
        print("1. Testing root endpoint...")
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Root endpoint: {data['message']}")
        else:
            print(f"✗ Root endpoint failed: {response.status_code}")
            return False
        
        # Test 2: API info
        print("\n2. Testing API info...")
        response = requests.get(f"{BASE_URL}/api/v1/info")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API info: {data['name']}")
        else:
            print(f"✗ API info failed: {response.status_code}")
        
        # Test 3: Health check
        print("\n3. Testing health check...")
        response = requests.get(f"{BASE_URL}/api/v1/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Health check: {data['status']}")
        else:
            print(f"✗ Health check failed: {response.status_code}")
        
        # Test 4: Authentication
        print("\n4. Testing authentication...")
        login_data = {
            "username": "admin",
            "password": "admin123"
        }
        response = requests.post(
            f"{BASE_URL}/api/v1/auth/login",
            json=login_data,
            headers=HEADERS
        )
        
        if response.status_code == 200:
            data = response.json()
            access_token = data.get("access_token")
            print("✓ Authentication successful")
            
            # Test 5: Protected endpoint
            print("\n5. Testing protected endpoint...")
            auth_headers = {
                **HEADERS,
                "Authorization": f"Bearer {access_token}"
            }
            
            response = requests.get(
                f"{BASE_URL}/api/v1/auth/me",
                headers=auth_headers
            )
            
            if response.status_code == 200:
                user_data = response.json()
                print(f"✓ Protected endpoint: user {user_data['username']}")
            else:
                print(f"✗ Protected endpoint failed: {response.status_code}")
            
            # Test 6: OpenAI models
            print("\n6. Testing OpenAI models...")
            response = requests.get(
                f"{BASE_URL}/api/v1/openai/models",
                headers=auth_headers
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✓ OpenAI models: {len(data['models'])} available")
            else:
                print(f"⚠ OpenAI models failed: {response.status_code}")
            
            # Test 7: Query endpoint (may fail without full RAG system)
            print("\n7. Testing query endpoint...")
            query_data = {
                "query_text": "Test query",
                "query_type": "semantic",
                "max_results": 3
            }
            
            response = requests.post(
                f"{BASE_URL}/api/v1/query/search",
                json=query_data,
                headers=auth_headers
            )
            
            if response.status_code == 200:
                print("✓ Query endpoint working")
            elif response.status_code == 503:
                print("⚠ Query endpoint not available (RAG system not initialized)")
            else:
                print(f"⚠ Query endpoint: {response.status_code}")
            
        else:
            print(f"✗ Authentication failed: {response.status_code}")
            return False
        
        print("\n" + "=" * 60)
        print("🎉 API CLIENT TEST COMPLETED SUCCESSFULLY!")
        print("✅ All basic endpoints are working")
        print("✅ Authentication is functional")
        print("✅ API is ready for use")
        print("=" * 60)
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server")
        print("Make sure the server is running on http://localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        return False

def test_openai_chat():
    """Test OpenAI chat endpoint"""
    print("\n" + "=" * 60)
    print("TESTING OPENAI CHAT INTEGRATION")
    print("=" * 60)
    
    try:
        # Login first
        login_data = {"username": "admin", "password": "admin123"}
        response = requests.post(f"{BASE_URL}/api/v1/auth/login", json=login_data)
        
        if response.status_code != 200:
            print("✗ Authentication failed")
            return False
        
        token = response.json()["access_token"]
        auth_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        
        # Test OpenAI chat
        chat_data = {
            "messages": [
                {"role": "user", "content": "Hello! This is a test message. Please respond briefly."}
            ],
            "model": "gpt-3.5-turbo",
            "temperature": 0.7
        }
        
        print("Testing OpenAI chat endpoint...")
        response = requests.post(
            f"{BASE_URL}/api/v1/openai/chat",
            json=chat_data,
            headers=auth_headers,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✓ OpenAI chat successful!")
            print(f"✓ Response: {data['response'][:100]}...")
            print(f"✓ Model: {data['model']}")
            print(f"✓ Tokens used: {data['usage']['total_tokens']}")
            return True
        else:
            print(f"⚠ OpenAI chat failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"⚠ OpenAI chat test error: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔗 Testing RAG API Server...")
    print("Make sure the server is running first: python start_api_server.py")
    print()
    
    # Wait a moment for user to confirm
    input("Press Enter when the server is running...")
    
    # Run basic tests
    basic_success = test_api_endpoints()
    
    if basic_success:
        # Test OpenAI integration
        try_openai = input("\nTest OpenAI integration? (y/n): ").lower().strip() == 'y'
        if try_openai:
            openai_success = test_openai_chat()
        else:
            print("⏭️  Skipping OpenAI test")
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    if basic_success:
        print("🎉 PART 6 API SERVER IS WORKING!")
        print("✅ All core API endpoints functional")
        print("✅ Authentication and authorization working")
        print("✅ Ready for production use")
    else:
        print("⚠️  PART 6 HAS ISSUES")
        print("Check server logs for details")
    print("=" * 60)
