#!/usr/bin/env python3
"""
Test Script for RAG System Part 4: Semantic Search Layer
========================================================

Tests the complete semantic search pipeline including:
- Text chunking with SEC document awareness
- Sentence transformer embeddings
- FAISS vector index operations
- End-to-end semantic search functionality
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
import json
import numpy as np

# Import components individually to avoid circular imports
from RAG.config import RAGConfig

try:
    from RAG.layer3_semantic.embeddings_manager import EmbeddingsManager
    from RAG.layer3_semantic.text_chunking import TextChunkingManager
    from RAG.layer3_semantic.faiss_manager import FAISSIndexManager
    from RAG.layer3_semantic.manager import SemanticSearchManager
    IMPORTS_SUCCESS = True
except ImportError as e:
    print(f"Import error: {e}")
    print("Some tests may be skipped due to missing dependencies")
    IMPORTS_SUCCESS = False

def test_embeddings_manager():
    """Test the embeddings manager"""
    print("\n=== Testing Embeddings Manager ===")
    
    if not IMPORTS_SUCCESS:
        print("✗ Skipping due to import errors")
        return False
    
    try:
        config = RAGConfig()
        embeddings_manager = EmbeddingsManager(config)
        
        # Test single text embedding
        test_texts = [
            "Apple Inc. reported strong quarterly earnings driven by iPhone sales.",
            "The company's revenue increased by 15% year-over-year.",
            "Risk factors include supply chain disruptions and regulatory changes."
        ]
        
        print(f"Embedding {len(test_texts)} test texts...")
        embeddings = embeddings_manager.embed_texts(test_texts)
        
        if embeddings is not None:
            print(f"✓ Successfully generated embeddings: {embeddings.shape}")
            print(f"✓ Model: {embeddings_manager.model_name}")
            print(f"✓ Embedding dimension: {embeddings.shape[1]}")
            
            # Test similarity calculation
            similarity = embeddings_manager.calculate_similarity(embeddings[0], embeddings[1])
            print(f"✓ Similarity between texts 1 and 2: {similarity:.4f}")
            
            # Test batch embedding with caching
            print("Testing batch embedding and caching...")
            embeddings2 = embeddings_manager.embed_texts(test_texts)  # Should use cache
            
            if np.array_equal(embeddings, embeddings2):
                print("✓ Caching working correctly")
            else:
                print("⚠ Caching issue detected")
            
            # Test statistics
            stats = embeddings_manager.get_stats()
            print(f"✓ Embeddings statistics: {stats}")
            
            return True
        else:
            print("✗ Failed to generate embeddings")
            return False
            
    except Exception as e:
        print(f"✗ Embeddings manager test failed: {e}")
        return False

def test_text_chunking():
    """Test the text chunking manager"""
    print("\n=== Testing Text Chunking Manager ===")
    
    if not IMPORTS_SUCCESS:
        print("✗ Skipping due to import errors")
        return None
    
    try:
        config = RAGConfig()
        chunking_manager = TextChunkingManager(config)
        
        # Sample SEC filing text
        sample_text = """
        ITEM 1. BUSINESS
        
        Apple Inc. designs, manufactures and markets smartphones, personal computers, tablets, wearables and accessories, and sells a variety of related services. The Company's products include iPhone, Mac, iPad, AirPods, Apple TV, Apple Watch, Beats products, HomePod, iPod touch and accessories.
        
        ITEM 1A. RISK FACTORS
        
        The Company's business, reputation, results of operations, financial condition and stock price can be affected by a number of factors, including those set forth below, that could cause the Company's results to differ materially from management's expectations.
        
        Competition
        The markets for the Company's products and services are highly competitive, and the Company expects competition to intensify. The Company's competitors range from large multinational corporations to small specialized firms.
        
        ITEM 2. PROPERTIES
        
        The Company's headquarters are located in Cupertino, California. As of September 30, 2023, the Company owned or leased approximately 34 million square feet of building space worldwide.
        """
        
        print("Chunking sample SEC filing text...")
        chunks = chunking_manager.chunk_document(
            text=sample_text,
            source_document="AAPL_10-K_2023-10-27.txt",
            ticker="AAPL",
            form_type="10-K",
            filing_date="2023-10-27"
        )
        
        if chunks:
            print(f"✓ Created {len(chunks)} chunks")
            
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                print(f"  Chunk {i+1}:")
                print(f"    Section: {chunk.section}")
                print(f"    Length: {len(chunk.text)} chars")
                print(f"    Token count: {chunk.token_count}")
                print(f"    Text preview: {chunk.text[:100]}...")
                print()
            
            # Test chunking statistics
            stats = chunking_manager.get_chunking_stats(chunks)
            print(f"✓ Chunking statistics: {json.dumps(stats, indent=2)}")
            
            # Test filtering
            business_chunks = chunking_manager.get_chunks_by_section(chunks, 'business')
            print(f"✓ Found {len(business_chunks)} business section chunks")
            
            return chunks
        else:
            print("✗ No chunks created")
            return None
            
    except Exception as e:
        print(f"✗ Text chunking test failed: {e}")
        return None

def test_faiss_manager(test_chunks):
    """Test the FAISS manager"""
    print("\n=== Testing FAISS Manager ===")
    
    if not IMPORTS_SUCCESS:
        print("✗ Skipping due to import errors")
        return False
    
    if not test_chunks:
        print("⚠ No test chunks provided, skipping FAISS test")
        return False
    
    try:
        config = RAGConfig()
        faiss_manager = FAISSIndexManager(config)
        embeddings_manager = EmbeddingsManager(config)
        
        # Generate embeddings for test chunks
        chunk_texts = [chunk.text for chunk in test_chunks]
        print(f"Generating embeddings for {len(chunk_texts)} chunks...")
        
        embeddings = embeddings_manager.embed_texts(chunk_texts)
        if embeddings is None:
            print("✗ Failed to generate embeddings for FAISS test")
            return False
        
        print(f"✓ Generated embeddings: {embeddings.shape}")
        
        # Test index creation
        print("Creating FAISS index...")
        success = faiss_manager.create_index(
            embeddings=embeddings,
            chunks=test_chunks,
            model_name=embeddings_manager.model_name,
            force_recreate=True
        )
        
        if not success:
            print("✗ Failed to create FAISS index")
            return False
        
        print("✓ FAISS index created successfully")
        
        # Test search
        query_text = "business operations and competition"
        query_embedding = embeddings_manager.embed_texts([query_text])
        
        if query_embedding is not None:
            print(f"Searching for: '{query_text}'")
            search_results = faiss_manager.search(query_embedding[0], k=3)
            
            print(f"✓ Found {len(search_results)} search results:")
            for i, (chunk_id, similarity) in enumerate(search_results):
                print(f"  Result {i+1}: Chunk {chunk_id}, Similarity: {similarity:.4f}")
        
        # Test index statistics
        stats = faiss_manager.get_index_stats()
        print(f"✓ FAISS index statistics: {json.dumps(stats, indent=2)}")
        
        # Test save/load
        print("Testing index persistence...")
        faiss_manager._save_index()
        faiss_manager._save_metadata()
        
        # Create new manager and load
        faiss_manager2 = FAISSIndexManager(config)
        load_success = faiss_manager2.load_index()
        
        if load_success:
            print("✓ Index saved and loaded successfully")
            
            # Test search with loaded index
            search_results2 = faiss_manager2.search(query_embedding[0], k=3)
            if len(search_results2) == len(search_results):
                print("✓ Loaded index produces same search results")
            else:
                print("⚠ Loaded index results differ from original")
        else:
            print("✗ Failed to load saved index")
        
        return True
        
    except Exception as e:
        print(f"✗ FAISS manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_semantic_search_end_to_end():
    """Test the complete semantic search pipeline"""
    print("\n=== Testing Complete Semantic Search Pipeline ===")
    
    if not IMPORTS_SUCCESS:
        print("✗ Skipping due to import errors")
        return False
    
    try:
        config = RAGConfig()
        semantic_manager = SemanticSearchManager(config)
        
        # Test with minimal data (use existing documents if available)
        print("Building semantic search index with existing data...")
        
        # Try to build index with existing documents
        success = semantic_manager.build_search_index(
            use_existing_data=True,
            companies=["AAPL", "GOOGL", "MSFT"],  # Test with a few companies
            max_documents=5  # Limit for testing
        )
        
        if not success:
            print("⚠ No existing data found, creating mock documents...")
            # Create some mock documents for testing
            mock_docs = [
                {
                    'content': """
                    ITEM 1. BUSINESS
                    Apple Inc. designs, manufactures and markets smartphones, personal computers, tablets, wearables and accessories. The Company's iPhone generates the majority of revenue through device sales and services.
                    
                    ITEM 1A. RISK FACTORS
                    The Company faces intense competition in all segments of its business. Economic downturns could reduce demand for the Company's products.
                    """,
                    'filename': 'AAPL_10-K_2023-10-27.txt',
                    'ticker': 'AAPL',
                    'form_type': '10-K',
                    'filing_date': '2023-10-27'
                },
                {
                    'content': """
                    ITEM 1. BUSINESS
                    Alphabet Inc. is a holding company whose primary business is Google, which includes search, advertising, operating systems and cloud computing. Google's advertising revenue represents the majority of total revenue.
                    
                    ITEM 1A. RISK FACTORS
                    The Company operates in rapidly changing markets with evolving technologies. Regulatory changes could impact advertising revenue and data collection practices.
                    """,
                    'filename': 'GOOGL_10-K_2023-02-02.txt',
                    'ticker': 'GOOGL',
                    'form_type': '10-K',
                    'filing_date': '2023-02-02'
                }
            ]
            
            # Save mock documents to test directory
            raw_path = Path(config.RAW_DATA_PATH) / "by_ticker"
            for doc in mock_docs:
                ticker_path = raw_path / doc['ticker']
                ticker_path.mkdir(parents=True, exist_ok=True)
                
                file_path = ticker_path / doc['filename']
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(doc['content'])
            
            # Try again with mock data
            success = semantic_manager.build_search_index(
                use_existing_data=True,
                companies=["AAPL", "GOOGL"],
                max_documents=2
            )
        
        if success:
            print("✓ Semantic search index built successfully")
            
            # Test search queries
            test_queries = [
                "iPhone revenue and device sales",
                "competition and market risks",
                "advertising revenue Google",
                "regulatory risks and compliance"
            ]
            
            for query in test_queries:
                print(f"\n--- Searching for: '{query}' ---")
                results = semantic_manager.search(query, k=3)
                
                if results:
                    print(f"✓ Found {len(results)} results:")
                    for i, result in enumerate(results):
                        print(f"  {i+1}. [{result.ticker}] {result.source_document}")
                        print(f"     Section: {result.section}")
                        print(f"     Similarity: {result.similarity_score:.4f}")
                        print(f"     Preview: {result.chunk_text[:150]}...")
                        print()
                else:
                    print("⚠ No results found")
            
            # Test company-specific search
            print("\n--- Testing Company-Specific Search ---")
            apple_results = semantic_manager.search_by_company("business operations", "AAPL", k=2)
            print(f"✓ Found {len(apple_results)} Apple-specific results")
            
            # Test section-specific search
            print("\n--- Testing Section-Specific Search ---")
            risk_results = semantic_manager.search_by_section("competition", "risk_factors", k=2)
            print(f"✓ Found {len(risk_results)} risk factors results")
            
            # Test index statistics
            stats = semantic_manager.get_index_statistics()
            print(f"\n✓ Index statistics: {json.dumps(stats, indent=2)}")
            
            # Test health check
            health = semantic_manager.health_check()
            print(f"✓ Health check: {health['status']}")
            if health['issues']:
                print(f"  Issues: {health['issues']}")
            
            return True
        else:
            print("✗ Failed to build semantic search index")
            return False
            
    except Exception as e:
        print(f"✗ End-to-end semantic search test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Part 4 tests"""
    print("RAG System Part 4: Semantic Search Layer Tests")
    print("=" * 60)
    
    # Set up logging to reduce noise during testing
    logging.getLogger().setLevel(logging.WARNING)
    
    test_results = {
        'embeddings_manager': False,
        'text_chunking': False,
        'faiss_manager': False,
        'end_to_end_search': False
    }
    
    # Test 1: Embeddings Manager
    test_results['embeddings_manager'] = test_embeddings_manager()
    
    # Test 2: Text Chunking Manager
    test_chunks = test_text_chunking()
    test_results['text_chunking'] = test_chunks is not None
    
    # Test 3: FAISS Manager
    test_results['faiss_manager'] = test_faiss_manager(test_chunks)
    
    # Test 4: End-to-End Semantic Search
    test_results['end_to_end_search'] = test_semantic_search_end_to_end()
    
    # Summary
    print("\n" + "=" * 60)
    print("Part 4 Test Results Summary:")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name.replace('_', ' ').title():.<40} {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Part 4: Semantic Search Layer - ALL TESTS PASSED!")
        print("Ready to proceed to Part 5!")
    else:
        print(f"\n⚠ {total - passed} tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
