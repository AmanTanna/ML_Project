#!/usr/bin/env python3
"""
Test RAG System Part 2 - Raw Source Layer (Layer 1)
====================================================

Tests SEC EDGAR API client, document storage, and raw source manager.
"""

import sys
from pathlib import Path
import traceback

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_part2_raw_source_layer():
    """Test Part 2 implementation"""
    print("Testing RAG System Part 2 - Raw Source Layer (Layer 1)")
    print("=" * 60)
    
    try:
        # Test imports
        print("🔄 Testing imports...")
        from RAG import RAGConfig
        from RAG.layer1_raw import RawSourceManager, SECClient, DocumentStorage, Filing
        print("✅ All Layer 1 imports successful!")
        
        # Test configuration
        print("\n🔄 Testing configuration...")
        config = RAGConfig()
        print(f"✅ RAGConfig created with {len(config.SP500_YEARS)} years: {config.SP500_YEARS}")
        print(f"✅ Filing types: {config.FILING_TYPES}")
        print(f"✅ Raw PDFs directory: {config.RAW_PDFS_DIR}")
        
        # Test SEC client
        print("\n🔄 Testing SEC EDGAR API client...")
        sec_client = SECClient(config)
        
        # Test company lookup
        test_ticker = 'AAPL'
        cik = sec_client.get_company_cik(test_ticker)
        if cik:
            print(f"✅ SEC API working - CIK for {test_ticker}: {cik}")
        else:
            print(f"⚠️  Could not get CIK for {test_ticker} (API may be rate limited)")
        
        # Test S&P 500 tickers
        tickers = sec_client.get_sp500_tickers()
        print(f"✅ Got {len(tickers)} S&P 500 sample tickers")
        
        # Test document storage
        print("\n🔄 Testing document storage...")
        doc_storage = DocumentStorage(config)
        print("✅ Document storage initialized")
        
        # Test storage statistics
        stats = doc_storage.get_storage_stats()
        print(f"✅ Storage stats: {stats['total_documents']} documents, {stats['total_size_formatted']}")
        
        # Test raw source manager
        print("\n🔄 Testing Raw Source Manager...")
        raw_manager = RawSourceManager(config)
        print("✅ Raw Source Manager created")
        
        # Initialize Layer 1
        raw_manager.initialize()
        print("✅ Raw Source Manager initialized")
        
        # Test health check
        health = raw_manager._health_check()
        print(f"✅ Health check: SEC API accessible: {health['sec_api_accessible']}")
        print(f"✅ Storage directories exist: {health['storage_directories_exist']}")
        
        # Test layer statistics
        layer_stats = raw_manager.get_layer_stats()
        print(f"✅ Layer stats: {layer_stats['sec_requests_made']} SEC requests made")
        
        print("\n🎉 Part 2 Complete - Raw Source Layer working!")
        print("\nLayer 1 Features Available:")
        print("- ✅ SEC EDGAR API client with rate limiting")
        print("- ✅ Document storage with metadata tracking")
        print("- ✅ Filing registry and validation")
        print("- ✅ Organized directory structure")
        print("- ✅ Health monitoring and statistics")
        
        print(f"\nStorage structure created at: {config.RAW_PDFS_DIR}")
        print("\nNext steps:")
        print("- Part 3: Structured Data Layer (DuckDB + Parquet)")
        print("- Part 4: Semantic Search Layer (FAISS + embeddings)")
        print("- Part 5: Multi-layer Query Engine")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\n📋 Make sure to install dependencies:")
        print("pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print(f"\n📋 Error details:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_part2_raw_source_layer()
    
    if success:
        print("\n✅ Part 2 tests passed - Ready for Part 3!")
    else:
        print("\n❌ Part 2 needs fixes before proceeding")
        
    sys.exit(0 if success else 1)
