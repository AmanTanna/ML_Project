"""
Query Router
============

Intelligent routing system that determines the optimal RAG layer(s) 
for different types of financial queries.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from ..config import RAGConfig
from ..utils import setup_logging

class QueryType(Enum):
    """Types of queries the system can handle"""
    FINANCIAL_METRICS = "financial_metrics"
    SEMANTIC_SEARCH = "semantic_search"
    DOCUMENT_RETRIEVAL = "document_retrieval"
    COMPANY_ANALYSIS = "company_analysis"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    TEMPORAL_ANALYSIS = "temporal_analysis"
    REGULATORY_SEARCH = "regulatory_search"
    RISK_ANALYSIS = "risk_analysis"

class LayerType(Enum):
    """Available RAG layers"""
    SEMANTIC = "semantic"
    STRUCTURED = "structured"
    RAW = "raw"

@dataclass
class QueryClassification:
    """Result of query classification"""
    primary_type: QueryType
    secondary_types: List[QueryType]
    confidence: float
    keywords_found: List[str]
    entities_found: List[str]
    reasoning: str

@dataclass
class RoutingDecision:
    """Routing decision for a query"""
    primary_layer: LayerType
    secondary_layers: List[LayerType]
    layer_weights: Dict[LayerType, float]
    routing_strategy: str
    confidence: float
    reasoning: str

class QueryRouter:
    """Intelligent query routing system"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = setup_logging(config.LOG_LEVEL, config.LOG_FILE)
        
        # Initialize keyword patterns for different query types
        self._init_patterns()
        
        # Routing statistics
        self.routing_stats = {
            'total_routes': 0,
            'layer_usage': {layer.value: 0 for layer in LayerType},
            'query_type_distribution': {qtype.value: 0 for qtype in QueryType},
            'avg_confidence': 0.0
        }
        
        self.logger.info("Query Router initialized")
    
    def _init_patterns(self):
        """Initialize keyword patterns for query classification"""
        
        # Financial metrics patterns
        self.financial_patterns = {
            'revenue': ['revenue', 'sales', 'income statement', 'top line', 'turnover'],
            'profit': ['profit', 'net income', 'earnings', 'bottom line', 'margin'],
            'ratios': ['ratio', 'p/e', 'pe ratio', 'debt to equity', 'current ratio', 'roe', 'roa'],
            'balance_sheet': ['assets', 'liabilities', 'equity', 'balance sheet', 'debt', 'cash'],
            'cash_flow': ['cash flow', 'operating cash', 'free cash flow', 'capex', 'fcf'],
            'valuation': ['valuation', 'market cap', 'enterprise value', 'ev', 'multiple'],
            'growth': ['growth', 'yoy', 'year over year', 'cagr', 'growth rate']
        }
        
        # Semantic search patterns
        self.semantic_patterns = {
            'strategy': ['strategy', 'strategic', 'business model', 'approach', 'plan'],
            'risk': ['risk', 'risks', 'risk factor', 'uncertainty', 'threat', 'exposure'],
            'competition': ['competition', 'competitive', 'competitor', 'market share', 'rivalry'],
            'management': ['management', 'leadership', 'ceo', 'executive', 'board'],
            'market': ['market', 'industry', 'sector', 'market condition', 'outlook'],
            'regulatory': ['regulatory', 'regulation', 'compliance', 'legal', 'sec'],
            'technology': ['technology', 'innovation', 'digital', 'tech', 'r&d'],
            'operations': ['operations', 'operational', 'efficiency', 'process', 'supply chain']
        }
        
        # Document retrieval patterns
        self.document_patterns = {
            'filings': ['10-k', '10-q', '8-k', 'filing', 'sec filing', 'form'],
            'sections': ['md&a', 'business', 'risk factors', 'financial statements'],
            'reports': ['annual report', 'quarterly report', 'earnings report'],
            'specific_docs': ['proxy', 'registration', 'prospectus']
        }
        
        # Company and entity patterns
        self.entity_patterns = {
            'ticker': re.compile(r'\b[A-Z]{1,5}\b'),  # Stock tickers
            'company': re.compile(r'\b(?:Inc|Corp|Ltd|LLC|Company|Corporation)\b', re.IGNORECASE),
            'dates': re.compile(r'\b(?:Q[1-4]|20\d{2}|19\d{2}|\d{1,2}/\d{1,2}/\d{4})\b'),
            'financial_terms': re.compile(r'\$[\d,]+(?:\.\d{2})?[BMK]?', re.IGNORECASE)
        }
        
        # Comparative analysis patterns
        self.comparative_patterns = [
            'compare', 'versus', 'vs', 'compared to', 'relative to', 'against',
            'peer', 'competitor', 'industry average', 'benchmark'
        ]
        
        # Temporal analysis patterns
        self.temporal_patterns = [
            'trend', 'over time', 'historical', 'year over year', 'quarterly',
            'annual', 'growth', 'change', 'evolution', 'progression'
        ]
    
    def classify_query(self, query_text: str) -> QueryClassification:
        """Classify the type of query"""
        query_lower = query_text.lower()
        
        # Score different query types
        type_scores = {}
        keywords_found = []
        
        # Check financial metrics patterns
        financial_score = 0
        for category, patterns in self.financial_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    financial_score += 1
                    keywords_found.append(pattern)
        
        if financial_score > 0:
            type_scores[QueryType.FINANCIAL_METRICS] = financial_score / len(self.financial_patterns)
        
        # Check semantic patterns
        semantic_score = 0
        for category, patterns in self.semantic_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    semantic_score += 1
                    keywords_found.append(pattern)
        
        if semantic_score > 0:
            type_scores[QueryType.SEMANTIC_SEARCH] = semantic_score / len(self.semantic_patterns)
        
        # Check document patterns
        document_score = 0
        for category, patterns in self.document_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    document_score += 1
                    keywords_found.append(pattern)
        
        if document_score > 0:
            type_scores[QueryType.DOCUMENT_RETRIEVAL] = document_score / len(self.document_patterns)
        
        # Check for comparative analysis
        comparative_score = sum(1 for pattern in self.comparative_patterns if pattern in query_lower)
        if comparative_score > 0:
            type_scores[QueryType.COMPARATIVE_ANALYSIS] = comparative_score / len(self.comparative_patterns)
        
        # Check for temporal analysis
        temporal_score = sum(1 for pattern in self.temporal_patterns if pattern in query_lower)
        if temporal_score > 0:
            type_scores[QueryType.TEMPORAL_ANALYSIS] = temporal_score / len(self.temporal_patterns)
        
        # Extract entities
        entities_found = self._extract_entities(query_text)
        
        # Determine primary and secondary types
        if not type_scores:
            # Default to semantic search for unclassified queries
            primary_type = QueryType.SEMANTIC_SEARCH
            secondary_types = []
            confidence = 0.5
        else:
            sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
            primary_type = sorted_types[0][0]
            confidence = sorted_types[0][1]
            
            # Secondary types are those with scores > 0.1 but not primary
            secondary_types = [qtype for qtype, score in sorted_types[1:] if score > 0.1]
        
        # Generate reasoning
        reasoning = self._generate_classification_reasoning(primary_type, secondary_types, keywords_found, entities_found)
        
        # Update statistics
        self.routing_stats['query_type_distribution'][primary_type.value] += 1
        
        classification = QueryClassification(
            primary_type=primary_type,
            secondary_types=secondary_types,
            confidence=confidence,
            keywords_found=keywords_found,
            entities_found=entities_found,
            reasoning=reasoning
        )
        
        self.logger.debug(f"Query classified as {primary_type.value} with confidence {confidence:.2f}")
        return classification
    
    def route_query(self, query_text: str, classification: Optional[QueryClassification] = None) -> RoutingDecision:
        """Route query to appropriate RAG layers"""
        
        if classification is None:
            classification = self.classify_query(query_text)
        
        # Determine routing based on query type
        routing_rules = {
            QueryType.FINANCIAL_METRICS: {
                'primary': LayerType.STRUCTURED,
                'secondary': [LayerType.SEMANTIC],
                'weights': {LayerType.STRUCTURED: 0.7, LayerType.SEMANTIC: 0.3}
            },
            QueryType.SEMANTIC_SEARCH: {
                'primary': LayerType.SEMANTIC,
                'secondary': [LayerType.RAW],
                'weights': {LayerType.SEMANTIC: 0.8, LayerType.RAW: 0.2}
            },
            QueryType.DOCUMENT_RETRIEVAL: {
                'primary': LayerType.RAW,
                'secondary': [LayerType.SEMANTIC],
                'weights': {LayerType.RAW: 0.6, LayerType.SEMANTIC: 0.4}
            },
            QueryType.COMPANY_ANALYSIS: {
                'primary': LayerType.SEMANTIC,
                'secondary': [LayerType.STRUCTURED, LayerType.RAW],
                'weights': {LayerType.SEMANTIC: 0.5, LayerType.STRUCTURED: 0.3, LayerType.RAW: 0.2}
            },
            QueryType.COMPARATIVE_ANALYSIS: {
                'primary': LayerType.STRUCTURED,
                'secondary': [LayerType.SEMANTIC],
                'weights': {LayerType.STRUCTURED: 0.6, LayerType.SEMANTIC: 0.4}
            },
            QueryType.TEMPORAL_ANALYSIS: {
                'primary': LayerType.STRUCTURED,
                'secondary': [LayerType.SEMANTIC],
                'weights': {LayerType.STRUCTURED: 0.7, LayerType.SEMANTIC: 0.3}
            },
            QueryType.REGULATORY_SEARCH: {
                'primary': LayerType.SEMANTIC,
                'secondary': [LayerType.RAW],
                'weights': {LayerType.SEMANTIC: 0.7, LayerType.RAW: 0.3}
            },
            QueryType.RISK_ANALYSIS: {
                'primary': LayerType.SEMANTIC,
                'secondary': [LayerType.RAW],
                'weights': {LayerType.SEMANTIC: 0.8, LayerType.RAW: 0.2}
            }
        }
        
        # Get routing rule for primary type
        rule = routing_rules.get(classification.primary_type)
        if not rule:
            # Default routing
            rule = {
                'primary': LayerType.SEMANTIC,
                'secondary': [],
                'weights': {LayerType.SEMANTIC: 1.0}
            }
        
        # Adjust weights based on secondary types
        adjusted_weights = rule['weights'].copy()
        if classification.secondary_types:
            # Reduce primary weight and distribute to secondary types
            primary_weight = adjusted_weights[rule['primary']]
            reduction = min(0.2, 0.1 * len(classification.secondary_types))
            adjusted_weights[rule['primary']] = primary_weight - reduction
            
            # Add weight for secondary type layers
            weight_per_secondary = reduction / len(classification.secondary_types)
            for secondary_type in classification.secondary_types:
                secondary_rule = routing_rules.get(secondary_type)
                if secondary_rule:
                    secondary_layer = secondary_rule['primary']
                    if secondary_layer in adjusted_weights:
                        adjusted_weights[secondary_layer] += weight_per_secondary
                    else:
                        adjusted_weights[secondary_layer] = weight_per_secondary
        
        # Determine routing strategy
        if len(adjusted_weights) == 1:
            strategy = "single_layer"
        elif len(adjusted_weights) == 2:
            strategy = "dual_layer"
        else:
            strategy = "multi_layer"
        
        # Calculate routing confidence
        routing_confidence = min(1.0, classification.confidence + 0.2)
        
        # Generate reasoning
        reasoning = self._generate_routing_reasoning(
            classification, rule['primary'], rule['secondary'], strategy
        )
        
        # Update statistics
        self.routing_stats['total_routes'] += 1
        for layer in adjusted_weights.keys():
            self.routing_stats['layer_usage'][layer.value] += 1
        
        # Update average confidence
        total_routes = self.routing_stats['total_routes']
        current_avg = self.routing_stats['avg_confidence']
        self.routing_stats['avg_confidence'] = (
            (current_avg * (total_routes - 1) + routing_confidence) / total_routes
        )
        
        decision = RoutingDecision(
            primary_layer=rule['primary'],
            secondary_layers=rule['secondary'],
            layer_weights=adjusted_weights,
            routing_strategy=strategy,
            confidence=routing_confidence,
            reasoning=reasoning
        )
        
        self.logger.info(f"Query routed to {rule['primary'].value} (primary) with strategy: {strategy}")
        return decision
    
    def _extract_entities(self, query_text: str) -> List[str]:
        """Extract entities from query text"""
        entities = []
        
        # Extract tickers
        tickers = self.entity_patterns['ticker'].findall(query_text)
        entities.extend(f"TICKER:{ticker}" for ticker in tickers)
        
        # Extract company names
        if self.entity_patterns['company'].search(query_text):
            entities.append("ENTITY:company")
        
        # Extract dates
        dates = self.entity_patterns['dates'].findall(query_text)
        entities.extend(f"DATE:{date}" for date in dates)
        
        # Extract financial amounts
        amounts = self.entity_patterns['financial_terms'].findall(query_text)
        entities.extend(f"AMOUNT:{amount}" for amount in amounts)
        
        return entities
    
    def _generate_classification_reasoning(self, primary_type: QueryType, secondary_types: List[QueryType],
                                         keywords: List[str], entities: List[str]) -> str:
        """Generate reasoning for query classification"""
        reasoning = f"Classified as {primary_type.value} based on keywords: {', '.join(keywords[:5])}"
        
        if entities:
            reasoning += f". Entities found: {', '.join(entities[:3])}"
        
        if secondary_types:
            reasoning += f". Secondary types: {', '.join(t.value for t in secondary_types)}"
        
        return reasoning
    
    def _generate_routing_reasoning(self, classification: QueryClassification, primary_layer: LayerType,
                                  secondary_layers: List[LayerType], strategy: str) -> str:
        """Generate reasoning for routing decision"""
        reasoning = f"Routed to {primary_layer.value} layer based on {classification.primary_type.value} classification"
        
        if secondary_layers:
            reasoning += f". Secondary layers: {', '.join(l.value for l in secondary_layers)}"
        
        reasoning += f". Strategy: {strategy}"
        
        return reasoning
    
    def optimize_routing(self, query_feedback: List[Dict[str, Any]]):
        """Optimize routing based on feedback"""
        # This is a placeholder for learning-based routing optimization
        # In practice, you would analyze query success rates by routing decision
        # and adjust routing rules accordingly
        
        self.logger.info("Routing optimization placeholder - would analyze feedback and adjust rules")
        
        # Example of what optimization might look like:
        # - Track success rates by query type and routing decision
        # - Adjust layer weights based on performance
        # - Learn new patterns from successful queries
        
        pass
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        return {
            'routing_stats': self.routing_stats.copy(),
            'patterns_loaded': {
                'financial_patterns': len(self.financial_patterns),
                'semantic_patterns': len(self.semantic_patterns),
                'document_patterns': len(self.document_patterns)
            }
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of query router"""
        health = {
            'status': 'healthy',
            'issues': [],
            'components': {
                'patterns': 'loaded',
                'routing_rules': 'active',
                'statistics': 'tracking'
            }
        }
        
        # Check if patterns are loaded
        if not self.financial_patterns or not self.semantic_patterns:
            health['status'] = 'degraded'
            health['issues'].append('Some query patterns missing')
            health['components']['patterns'] = 'incomplete'
        
        return health
    
    def add_custom_pattern(self, query_type: str, category: str, patterns: List[str]):
        """Add custom patterns for query classification"""
        try:
            if query_type == 'financial':
                if category not in self.financial_patterns:
                    self.financial_patterns[category] = []
                self.financial_patterns[category].extend(patterns)
            elif query_type == 'semantic':
                if category not in self.semantic_patterns:
                    self.semantic_patterns[category] = []
                self.semantic_patterns[category].extend(patterns)
            elif query_type == 'document':
                if category not in self.document_patterns:
                    self.document_patterns[category] = []
                self.document_patterns[category].extend(patterns)
            
            self.logger.info(f"Added {len(patterns)} patterns to {query_type}.{category}")
            
        except Exception as e:
            self.logger.error(f"Failed to add custom patterns: {e}")
            raise
