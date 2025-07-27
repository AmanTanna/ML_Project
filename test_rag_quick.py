#!/usr/bin/env python3
"""
Quick RAG System Test
====================

Quick verification that all RAG components are properly installed and configured.
"""

import sys
from pathlib import Path

# Add RAG to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all RAG components can be imported"""
    print("🧪 Testing RAG System Imports...")
    
    try:
        from RAG import RAGConfig, StorageManager
        print("✅ Core RAG components imported successfully")
        
        from RAG.layer1_raw import RawSourceManager
        print("✅ Layer 1 (Raw Source) imported successfully")
        
        from RAG.layer2_structured import StructuredDataManager
        print("✅ Layer 2 (Structured Data) imported successfully")
        
        # Test configuration
        config = RAGConfig()
        print(f"✅ Configuration loaded - Base dir: {config.BASE_DIR}")
        
        # Test storage manager
        storage = StorageManager(config)
        print("✅ Storage manager initialized successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_dependencies():
    """Test that all required dependencies are available"""
    print("\n🔧 Testing Dependencies...")
    
    dependencies = [
        ("requests", "SEC EDGAR API calls"),
        ("duckdb", "Columnar database"),
        ("pyarrow", "Parquet file format"),
        ("pandas", "Data manipulation"),
        ("tqdm", "Progress bars"),
        ("jsonlines", "JSONL file handling")
    ]
    
    missing_deps = []
    
    for dep, description in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep} - {description}")
        except ImportError:
            print(f"❌ {dep} - {description} (MISSING)")
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing_deps)}")
        print("📦 Install with: pip install " + " ".join(missing_deps))
        return False
    
    return True

def main():
    """Run quick system test"""
    print("🚀 RAG SYSTEM QUICK TEST")
    print("=" * 40)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test dependencies  
    deps_ok = test_dependencies()
    
    if imports_ok and deps_ok:
        print("\n✅ ALL TESTS PASSED!")
        print("🎉 Your RAG system is ready to run!")
        print("\n💡 Next step: Run 'python rag_demo.py' for full demonstration")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("🔧 Please fix the issues above before running the full demo")

if __name__ == "__main__":
    main()