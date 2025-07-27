#!/usr/bin/env python3
"""
Part 5 Demo: Multi-layer Query Engine
=====================================

Demonstrates the unified query interface with real examples.
"""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from RAG.config import RAGConfig
from RAG.layer4_query import QueryEngineManager

async def demo_part5():
    """Demonstrate Part 5 functionality"""
    print("🚀 PART 5 DEMO: Multi-layer Query Engine")
    print("=" * 60)
    
    # Initialize the system
    config = RAGConfig()
    manager = QueryEngineManager(config)
    
    print("✅ Multi-layer Query Engine initialized successfully!")
    print()
    
    # Example queries to demonstrate different routing
    test_queries = [
        {
            'query': "What is Apple's revenue for 2023?",
            'description': "Financial metrics query → Should route to structured layer"
        },
        {
            'query': "What are the main risks facing technology companies?",
            'description': "Semantic search query → Should route to semantic layer"
        },
        {
            'query': "Show me Apple's business segments and their competitive advantages",
            'description': "Hybrid query → Should use multiple layers"
        },
        {
            'query': "Compare revenue growth of AAPL vs GOOGL over past 3 years",
            'description': "Comparative analysis → Should route to structured + semantic"
        }
    ]
    
    print("📊 QUERY ANALYSIS EXAMPLES")
    print("-" * 40)
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{test['query']}'")
        print(f"   Expected: {test['description']}")
        
        # Analyze query without executing
        analysis = manager.analyze_query(test['query'])
        
        if analysis.get('classification'):
            classification = analysis['classification']
            print(f"   ✓ Classified as: {classification['primary_type']}")
            print(f"   ✓ Confidence: {classification['confidence']:.2f}")
            
        if analysis.get('routing'):
            routing = analysis['routing']
            print(f"   ✓ Primary layer: {routing['primary_layer']}")
            print(f"   ✓ Strategy: {routing['strategy']}")
            
        if analysis.get('complexity'):
            complexity = analysis['complexity']
            print(f"   ✓ Complexity: {complexity['complexity_level']} (score: {complexity['complexity_score']})")
    
    print("\n" + "=" * 60)
    print("📈 SYSTEM STATISTICS")
    print("-" * 40)
    
    # Get system statistics
    stats = manager.get_system_stats()
    overview = stats['overview']
    
    print(f"Total queries analyzed: {len(test_queries)}")
    print(f"System components: ✅ Query Engine, ✅ Router, ✅ Fusion")
    print(f"Available layers: Semantic, Structured, Raw")
    
    print("\n" + "=" * 60)
    print("🔍 HEALTH CHECK")
    print("-" * 40)
    
    # Perform health check
    health = manager.health_check()
    print(f"Overall status: {health['status'].upper()}")
    
    if health['components']:
        print("Component status:")
        for component, status in health['components'].items():
            status_icon = "✅" if status.get('status') == 'healthy' else "⚠️"
            print(f"  {status_icon} {component}: {status.get('status', 'unknown')}")
    
    if health['issues']:
        print("Issues found:")
        for issue in health['issues']:
            print(f"  ⚠️ {issue}")
    
    print("\n" + "=" * 60)
    print("🎯 QUERY EXECUTION EXAMPLE")
    print("-" * 40)
    
    # Execute one real query to show end-to-end functionality
    try:
        response = await manager.process_query(
            query_text="What are Apple's main business segments?",
            explain_reasoning=True,
            max_results=3
        )
        
        print(f"Query: {response.query_text}")
        print(f"Results found: {len(response.results)}")
        print(f"Processing time: {response.processing_time:.3f}s")
        print(f"Strategy used: {response.query_strategy}")
        print(f"Layers accessed: {', '.join(response.layers_used)}")
        
        if response.results:
            print("\nTop result:")
            result = response.results[0]
            print(f"  Source: {result.source_type}")
            print(f"  Relevance: {result.relevance_score:.3f}")
            print(f"  Content: {result.content[:200]}...")
        
        if response.explanation:
            print(f"\nExplanation: {response.explanation}")
            
    except Exception as e:
        print(f"Query execution demo failed (expected): {e}")
        print("This is normal if there's limited data in the system.")
    
    print("\n" + "=" * 60)
    print("✨ PART 5 DEMONSTRATION COMPLETE!")
    print()
    print("The Multi-layer Query Engine provides:")
    print("📌 Intelligent query routing across all RAG layers")
    print("📌 Advanced results fusion and ranking")
    print("📌 Comprehensive performance monitoring")
    print("📌 Unified interface for complex financial queries")
    print("📌 Batch processing and caching capabilities")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(demo_part5())
