"""
Start RAG API Server - Part 6
=============================

Simple script to start the RAG API server for testing.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def start_server():
    """Start the RAG API server"""
    print("=" * 60)
    print("STARTING RAG API SERVER - PART 6")
    print("=" * 60)
    
    try:
        from RAG.layer5_api.api_server import APIServer
        
        print("✓ Importing API server components...")
        print("✓ Environment variables loaded")
        print(f"✓ OpenAI API Key: {'Configured' if os.getenv('OPENAI_API_KEY') else 'Not found'}")
        
        # Create and run server
        server = APIServer()
        print(f"✓ Starting server on {server.host}:{server.port}")
        print("\n📝 API Documentation available at:")
        print(f"   - Swagger UI: http://{server.host}:{server.port}/docs")
        print(f"   - ReDoc: http://{server.host}:{server.port}/redoc")
        print("\n🔑 Default credentials:")
        print("   - Admin: username=admin, password=admin123")
        print("   - User: username=user, password=user123")
        print("\n🚀 Server starting...")
        print("=" * 60)
        
        server.run()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server failed to start: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    start_server()
