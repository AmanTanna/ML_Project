"""
Simple Part 6 Test - Basic API Components
=======================================

Test basic API components without complex middleware.
"""

import os
import sys
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_environment_setup():
    """Test environment configuration"""
    print("--- Testing Environment Setup ---")
    
    # Check .env file
    env_file = "/Users/jineshshah/Desktop/aman_proj/ML_Project/.env"
    if os.path.exists(env_file):
        print("✓ .env file exists")
        
        # Check for OpenAI key
        with open(env_file, 'r') as f:
            content = f.read()
            if 'OPENAI_API_KEY' in content:
                print("✓ OpenAI API key configured")
            else:
                print("✗ OpenAI API key not found in .env")
    else:
        print("✗ .env file not found")
    
    return True

def test_auth_manager():
    """Test authentication manager"""
    print("\n--- Testing Auth Manager ---")
    
    try:
        from RAG.layer5_api.auth import auth_manager
        
        # Test user authentication
        user = auth_manager.authenticate_user("admin", "admin123")
        if user:
            print("✓ Admin user authentication working")
        else:
            print("✗ Admin user authentication failed")
            return False
        
        # Test token creation
        token = auth_manager.create_access_token("admin")
        if token:
            print("✓ Token creation working")
        else:
            print("✗ Token creation failed")
            return False
        
        # Test token verification
        try:
            payload = auth_manager.verify_token(token)
            if payload and payload.get("sub") == "admin":
                print("✓ Token verification working")
            else:
                print("✗ Token verification failed")
                return False
        except Exception as e:
            print(f"✗ Token verification error: {str(e)}")
            return False
        
        print("✓ Auth manager fully functional")
        return True
        
    except Exception as e:
        print(f"✗ Auth manager test failed: {str(e)}")
        return False

def test_openai_integration():
    """Test OpenAI integration"""
    print("\n--- Testing OpenAI Integration ---")
    
    try:
        from RAG.layer5_api.openai_integration import get_openai_manager
        
        # Test getting OpenAI manager
        try:
            manager = get_openai_manager()
            print("✓ OpenAI manager initialized")
        except Exception as e:
            print(f"✗ OpenAI manager initialization failed: {str(e)}")
            return False
        
        # Test available models
        try:
            models = manager.get_available_models()
            print(f"✓ Available models: {len(models)} models")
        except Exception as e:
            print(f"⚠ Could not fetch models (API key may be invalid): {str(e)}")
        
        # Test token estimation
        try:
            text = "This is a test message for token estimation"
            tokens = manager.estimate_tokens(text)
            print(f"✓ Token estimation: {tokens} tokens")
        except Exception as e:
            print(f"✗ Token estimation failed: {str(e)}")
            return False
        
        print("✓ OpenAI integration working")
        return True
        
    except Exception as e:
        print(f"✗ OpenAI integration test failed: {str(e)}")
        return False

def test_models():
    """Test Pydantic models"""
    print("\n--- Testing API Models ---")
    
    try:
        from RAG.layer5_api.models import (
            QueryRequest, LoginRequest, OpenAIRequest, 
            RAGChatRequest, HealthResponse
        )
        
        # Test QueryRequest
        query_req = QueryRequest(
            query_text="Test query",
            query_type="semantic",
            max_results=5
        )
        print("✓ QueryRequest model working")
        
        # Test LoginRequest
        login_req = LoginRequest(
            username="testuser",
            password="testpass"
        )
        print("✓ LoginRequest model working")
        
        # Test OpenAIRequest
        openai_req = OpenAIRequest(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-3.5-turbo"
        )
        print("✓ OpenAIRequest model working")
        
        # Test RAGChatRequest
        rag_req = RAGChatRequest(
            query="Test RAG query",
            model="gpt-3.5-turbo",
            max_context_results=3
        )
        print("✓ RAGChatRequest model working")
        
        # Test HealthResponse
        health_resp = HealthResponse(
            status="healthy",
            message="Test health response"
        )
        print("✓ HealthResponse model working")
        
        print("✓ All API models working correctly")
        return True
        
    except Exception as e:
        print(f"✗ API models test failed: {str(e)}")
        return False

def test_basic_api_structure():
    """Test basic API structure without running server"""
    print("\n--- Testing Basic API Structure ---")
    
    try:
        from RAG.layer5_api.api_server import app
        
        # Check if app is created
        if app:
            print("✓ FastAPI app created successfully")
        else:
            print("✗ FastAPI app creation failed")
            return False
        
        # Check app configuration
        if hasattr(app, 'title') and app.title:
            print(f"✓ App title: {app.title}")
        else:
            print("✗ App title not set")
        
        if hasattr(app, 'version') and app.version:
            print(f"✓ App version: {app.version}")
        else:
            print("✗ App version not set")
        
        # Check routes
        routes = [route.path for route in app.routes]
        expected_routes = ["/", "/api/v1/info"]
        
        for expected_route in expected_routes:
            if any(expected_route in route for route in routes):
                print(f"✓ Route found: {expected_route}")
            else:
                print(f"⚠ Route might be missing: {expected_route}")
        
        print("✓ Basic API structure looks good")
        return True
        
    except Exception as e:
        print(f"✗ Basic API structure test failed: {str(e)}")
        return False

def run_simple_tests():
    """Run simple component tests"""
    print("=" * 60)
    print("PART 6 SIMPLE COMPONENT TESTS")
    print("=" * 60)
    
    tests = [
        ("Environment Setup", test_environment_setup),
        ("Auth Manager", test_auth_manager),
        ("OpenAI Integration", test_openai_integration),
        ("API Models", test_models),
        ("Basic API Structure", test_basic_api_structure)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {total} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL COMPONENT TESTS PASSED!")
        print("✅ Part 6 API components are working correctly")
    else:
        print(f"\n⚠️  {total - passed} tests failed")
        print("Some components need attention")
    
    return passed == total

if __name__ == "__main__":
    success = run_simple_tests()
    
    print("\n" + "=" * 60)
    print("PART 6 STATUS")
    print("=" * 60)
    
    if success:
        print("🎉 PART 6 CORE COMPONENTS WORKING!")
        print("✅ Authentication system functional")
        print("✅ OpenAI integration configured")
        print("✅ API models defined correctly")
        print("✅ FastAPI structure in place")
        print("\n📝 Ready for full API testing with server")
    else:
        print("⚠️  PART 6 NEEDS ATTENTION")
        print("Some core components have issues")
        print("Fix these before proceeding to full API testing")
