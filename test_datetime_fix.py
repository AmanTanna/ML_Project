"""
Quick test for datetime serialization fix
"""

import json
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RAG.layer5_api.models import ErrorResponse, QueryResponse, OpenAIResponse

def test_datetime_serialization():
    """Test that datetime objects are properly serialized"""
    print("Testing datetime serialization in API models...")
    
    # Test ErrorResponse
    error = ErrorResponse(
        error="Test error",
        status_code=500,
        path="/test",
        details="Test details"
    )
    
    try:
        error_dict = error.dict()
        error_json = json.dumps(error_dict)  # This should not fail
        print("✓ ErrorResponse datetime serialization works")
    except TypeError as e:
        print(f"✗ ErrorResponse datetime serialization failed: {e}")
        return False
    
    # Test that timestamp is properly formatted
    if 'timestamp' in error_dict and isinstance(error_dict['timestamp'], str):
        print("✓ Timestamp converted to ISO string format")
    else:
        print("✗ Timestamp not properly converted")
        return False
    
    print(f"Sample serialized error: {error_json[:100]}...")
    return True

if __name__ == "__main__":
    success = test_datetime_serialization()
    if success:
        print("\n🎉 Datetime serialization fix is working correctly!")
        print("The API server should no longer crash with 'datetime is not JSON serializable' errors.")
    else:
        print("\n❌ Datetime serialization still has issues.")
