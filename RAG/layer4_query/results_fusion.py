"""
Results Fusion
==============

Advanced system for combining and ranking results from multiple RAG layers.
Implements sophisticated fusion algorithms for optimal result quality.
"""

import logging
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

from ..config import RAGConfig
from ..utils import setup_logging

@dataclass
class FusionConfig:
    """Configuration for result fusion"""
    max_results: int = 20
    min_score_threshold: float = 0.1
    diversity_weight: float = 0.2
    recency_weight: float = 0.1
    source_diversity_bonus: float = 0.1
    company_diversity_bonus: float = 0.05
    section_diversity_bonus: float = 0.05
    
@dataclass
class ResultMetrics:
    """Metrics for a single result"""
    relevance_score: float
    diversity_score: float
    recency_score: float
    source_authority: float
    content_quality: float
    final_score: float
    ranking_factors: Dict[str, float] = field(default_factory=dict)

class ResultsFusion:
    """Advanced results fusion and ranking system"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.logger = setup_logging(config.LOG_LEVEL, config.LOG_FILE)
        
        # Fusion configuration
        self.fusion_config = FusionConfig()
        
        # Source authority weights
        self.source_weights = {
            'semantic': 1.0,
            'structured': 0.9,
            'raw': 0.8
        }
        
        # Section authority weights (for SEC filings)
        self.section_weights = {
            'risk factors': 0.95,
            'md&a': 0.9,
            'business': 0.85,
            'financial statements': 0.9,
            'notes to financial statements': 0.8,
            'default': 0.7
        }
        
        # Company tier weights (could be based on market cap, etc.)
        self.company_tiers = {
            'AAPL': 1.0, 'MSFT': 1.0, 'GOOGL': 1.0, 'AMZN': 1.0,
            'TSLA': 0.95, 'META': 0.95, 'NVDA': 0.95,
            'default': 0.8
        }
        
        # Fusion statistics
        self.fusion_stats = {
            'total_fusions': 0,
            'avg_input_results': 0.0,
            'avg_output_results': 0.0,
            'fusion_methods_used': defaultdict(int),
            'avg_score_improvement': 0.0
        }
        
        self.logger.info("Results Fusion system initialized")
    
    def fuse_results(self, layer_results: Dict[str, List[Any]], 
                    layer_weights: Dict[str, float],
                    query_text: str,
                    max_results: Optional[int] = None) -> List[Any]:
        """
        Fuse results from multiple layers using advanced ranking algorithms
        
        Args:
            layer_results: Dictionary mapping layer names to their results
            layer_weights: Weights for each layer
            query_text: Original query for relevance calculation
            max_results: Maximum number of results to return
        """
        start_time = datetime.now()
        max_results = max_results or self.fusion_config.max_results
        
        try:
            self.logger.info(f"Starting fusion with {sum(len(results) for results in layer_results.values())} total results")
            
            # Collect all results with source information
            all_results = []
            for layer_name, results in layer_results.items():
                layer_weight = layer_weights.get(layer_name, 1.0)
                for result in results:
                    # Enhance result with layer information
                    result.layer_source = layer_name
                    result.layer_weight = layer_weight
                    all_results.append(result)
            
            if not all_results:
                self.logger.warning("No results to fuse")
                return []
            
            # Remove duplicates
            unique_results = self._remove_duplicates(all_results)
            self.logger.debug(f"After deduplication: {len(unique_results)} results")
            
            # Calculate comprehensive metrics for each result
            results_with_metrics = []
            for result in unique_results:
                metrics = self._calculate_result_metrics(result, query_text, unique_results)
                result.fusion_metrics = metrics
                results_with_metrics.append(result)
            
            # Apply fusion algorithm
            fused_results = self._apply_fusion_algorithm(results_with_metrics, query_text)
            
            # Final ranking and selection
            final_results = self._final_ranking_and_selection(fused_results, max_results)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_fusion_stats(len(all_results), len(final_results), processing_time)
            
            self.logger.info(f"Fusion completed: {len(final_results)} results in {processing_time:.3f}s")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Fusion failed: {e}")
            # Return best effort - just combine and sort by original scores
            all_results = []
            for results in layer_results.values():
                all_results.extend(results)
            
            return sorted(all_results, key=lambda x: getattr(x, 'relevance_score', 0), reverse=True)[:max_results]
    
    def _remove_duplicates(self, results: List[Any]) -> List[Any]:
        """Remove duplicate results using content similarity"""
        if len(results) <= 1:
            return results
        
        unique_results = []
        seen_content = set()
        
        for result in results:
            # Create content signature
            content = getattr(result, 'content', '').strip()
            if not content:
                continue
            
            # Simple content similarity check
            content_signature = self._create_content_signature(content)
            
            # Check for duplicates
            is_duplicate = False
            for seen_sig in seen_content:
                if self._calculate_signature_similarity(content_signature, seen_sig) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_content.add(content_signature)
                unique_results.append(result)
        
        return unique_results
    
    def _create_content_signature(self, content: str) -> str:
        """Create a signature for content similarity comparison"""
        # Simple approach: use first 100 characters, normalized
        signature = content.lower().replace('\n', ' ').replace('\t', ' ')
        # Remove extra spaces
        signature = ' '.join(signature.split())
        return signature[:100]
    
    def _calculate_signature_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between content signatures"""
        if not sig1 or not sig2:
            return 0.0
        
        # Simple Jaccard similarity on words
        words1 = set(sig1.split())
        words2 = set(sig2.split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_result_metrics(self, result: Any, query_text: str, all_results: List[Any]) -> ResultMetrics:
        """Calculate comprehensive metrics for a result"""
        
        # Base relevance score
        relevance_score = getattr(result, 'relevance_score', 0.5)
        
        # Diversity score (how different this result is from others)
        diversity_score = self._calculate_diversity_score(result, all_results)
        
        # Recency score (based on document date or filing date)
        recency_score = self._calculate_recency_score(result)
        
        # Source authority score
        source_authority = self._calculate_source_authority(result)
        
        # Content quality score
        content_quality = self._calculate_content_quality(result)
        
        # Calculate final score using weighted combination
        layer_weight = getattr(result, 'layer_weight', 1.0)
        
        final_score = (
            relevance_score * layer_weight * 0.4 +
            diversity_score * self.fusion_config.diversity_weight +
            recency_score * self.fusion_config.recency_weight +
            source_authority * 0.2 +
            content_quality * 0.1
        )
        
        # Ranking factors for explanation
        ranking_factors = {
            'relevance': relevance_score,
            'layer_weight': layer_weight,
            'diversity': diversity_score,
            'recency': recency_score,
            'source_authority': source_authority,
            'content_quality': content_quality
        }
        
        return ResultMetrics(
            relevance_score=relevance_score,
            diversity_score=diversity_score,
            recency_score=recency_score,
            source_authority=source_authority,
            content_quality=content_quality,
            final_score=final_score,
            ranking_factors=ranking_factors
        )
    
    def _calculate_diversity_score(self, result: Any, all_results: List[Any]) -> float:
        """Calculate how diverse this result is compared to others"""
        try:
            # Factors for diversity:
            # 1. Source type diversity
            # 2. Company diversity  
            # 3. Section diversity
            # 4. Content diversity
            
            diversity_score = 1.0
            
            # Source diversity
            source_type = getattr(result, 'source_type', 'unknown')
            source_counts = defaultdict(int)
            for r in all_results:
                source_counts[getattr(r, 'source_type', 'unknown')] += 1
            
            total_results = len(all_results)
            source_frequency = source_counts[source_type] / total_results
            source_diversity = 1.0 - source_frequency
            diversity_score += source_diversity * self.fusion_config.source_diversity_bonus
            
            # Company diversity
            company = getattr(result, 'company', None)
            if company:
                company_counts = defaultdict(int)
                for r in all_results:
                    r_company = getattr(r, 'company', None) 
                    if r_company:
                        company_counts[r_company] += 1
                
                if company_counts:
                    company_frequency = company_counts[company] / sum(company_counts.values())
                    company_diversity = 1.0 - company_frequency
                    diversity_score += company_diversity * self.fusion_config.company_diversity_bonus
            
            # Section diversity
            section = getattr(result, 'section', None)
            if section:
                section_counts = defaultdict(int)
                for r in all_results:
                    r_section = getattr(r, 'section', None)
                    if r_section:
                        section_counts[r_section] += 1
                
                if section_counts:
                    section_frequency = section_counts[section] / sum(section_counts.values())
                    section_diversity = 1.0 - section_frequency
                    diversity_score += section_diversity * self.fusion_config.section_diversity_bonus
            
            return min(diversity_score, 2.0)  # Cap at 2.0
            
        except Exception as e:
            self.logger.warning(f"Diversity calculation failed: {e}")
            return 1.0
    
    def _calculate_recency_score(self, result: Any) -> float:
        """Calculate recency score based on document date"""
        try:
            # Try to extract date from metadata
            metadata = getattr(result, 'metadata', {})
            filing_date = metadata.get('filing_date')
            
            if not filing_date:
                return 0.5  # Neutral score for undated content
            
            # Parse date (assuming YYYY-MM-DD format)
            if isinstance(filing_date, str):
                try:
                    from datetime import datetime
                    doc_date = datetime.strptime(filing_date[:10], '%Y-%m-%d')
                    now = datetime.now()
                    
                    # Calculate days since filing
                    days_old = (now - doc_date).days
                    
                    # Recency score decreases with age
                    # Recent (< 90 days): score 1.0
                    # Older documents: exponential decay
                    if days_old < 90:
                        return 1.0
                    elif days_old < 365:
                        return 0.8
                    elif days_old < 730:  # 2 years
                        return 0.6
                    else:
                        return 0.4
                        
                except (ValueError, TypeError):
                    return 0.5
            
            return 0.5
            
        except Exception as e:
            self.logger.warning(f"Recency calculation failed: {e}")
            return 0.5
    
    def _calculate_source_authority(self, result: Any) -> float:
        """Calculate source authority score"""
        try:
            authority_score = 1.0
            
            # Layer-based authority
            source_type = getattr(result, 'source_type', 'unknown')
            authority_score *= self.source_weights.get(source_type, 0.8)
            
            # Section-based authority (for SEC filings)
            section = getattr(result, 'section', '').lower()
            section_weight = self.section_weights.get(section, self.section_weights['default'])
            authority_score *= section_weight
            
            # Company-based authority
            company = getattr(result, 'company', None)
            if company:
                company_weight = self.company_tiers.get(company, self.company_tiers['default'])
                authority_score *= company_weight
            
            return min(authority_score, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Authority calculation failed: {e}")
            return 0.8
    
    def _calculate_content_quality(self, result: Any) -> float:
        """Calculate content quality score"""
        try:
            content = getattr(result, 'content', '')
            if not content:
                return 0.0
            
            quality_score = 0.5  # Base score
            
            # Length factor (not too short, not too long)
            content_length = len(content)
            if 100 <= content_length <= 2000:
                quality_score += 0.2
            elif content_length > 50:
                quality_score += 0.1
            
            # Completeness (no truncation indicators)
            if not any(indicator in content.lower() for indicator in ['...', '[truncated]', '...']):
                quality_score += 0.1
            
            # Structure indicators (paragraphs, sentences)
            if content.count('.') >= 2:  # At least 2 sentences
                quality_score += 0.1
            
            if '\n' in content:  # Multiple paragraphs
                quality_score += 0.1
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Quality calculation failed: {e}")
            return 0.5
    
    def _apply_fusion_algorithm(self, results: List[Any], query_text: str) -> List[Any]:
        """Apply the main fusion algorithm"""
        # For now, use the calculated final scores
        # In more advanced implementations, you could apply:
        # - Reciprocal Rank Fusion (RRF)
        # - Borda Count
        # - Condorcet methods
        # - Machine learning-based fusion
        
        # Sort by final fusion score
        sorted_results = sorted(results, 
                              key=lambda x: x.fusion_metrics.final_score, 
                              reverse=True)
        
        # Apply minimum score threshold
        filtered_results = [
            r for r in sorted_results 
            if r.fusion_metrics.final_score >= self.fusion_config.min_score_threshold
        ]
        
        self.fusion_stats['fusion_methods_used']['score_based'] += 1
        
        return filtered_results
    
    def _final_ranking_and_selection(self, results: List[Any], max_results: int) -> List[Any]:
        """Final ranking and selection of top results"""
        # Results are already sorted by fusion score
        # Apply final selection logic
        
        selected_results = []
        companies_seen = set()
        sections_seen = set()
        
        for result in results:
            if len(selected_results) >= max_results:
                break
            
            # Diversity constraints (optional)
            company = getattr(result, 'company', None)
            section = getattr(result, 'section', None)
            
            # You could implement diversity constraints here
            # For example, limit results per company or section
            
            selected_results.append(result)
            
            if company:
                companies_seen.add(company)
            if section:
                sections_seen.add(section)
        
        return selected_results
    
    def _update_fusion_stats(self, input_count: int, output_count: int, processing_time: float):
        """Update fusion statistics"""
        self.fusion_stats['total_fusions'] += 1
        
        # Update averages
        total_fusions = self.fusion_stats['total_fusions']
        
        current_avg_input = self.fusion_stats['avg_input_results']
        self.fusion_stats['avg_input_results'] = (
            (current_avg_input * (total_fusions - 1) + input_count) / total_fusions
        )
        
        current_avg_output = self.fusion_stats['avg_output_results']
        self.fusion_stats['avg_output_results'] = (
            (current_avg_output * (total_fusions - 1) + output_count) / total_fusions
        )
    
    def reciprocal_rank_fusion(self, ranked_lists: List[List[Any]], k: int = 60) -> List[Any]:
        """
        Apply Reciprocal Rank Fusion algorithm
        
        Args:
            ranked_lists: List of ranked result lists from different sources
            k: RRF parameter (default 60)
        """
        # Calculate RRF scores
        rrf_scores = defaultdict(float)
        all_items = {}
        
        for ranked_list in ranked_lists:
            for rank, item in enumerate(ranked_list, 1):
                item_id = getattr(item, 'result_id', id(item))
                rrf_scores[item_id] += 1.0 / (k + rank)
                all_items[item_id] = item
        
        # Sort by RRF score
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return sorted results
        return [all_items[item_id] for item_id, score in sorted_items]
    
    def borda_count_fusion(self, ranked_lists: List[List[Any]]) -> List[Any]:
        """
        Apply Borda Count fusion algorithm
        """
        borda_scores = defaultdict(int)
        all_items = {}
        
        for ranked_list in ranked_lists:
            list_length = len(ranked_list)
            for rank, item in enumerate(ranked_list):
                item_id = getattr(item, 'result_id', id(item))
                borda_scores[item_id] += (list_length - rank)
                all_items[item_id] = item
        
        # Sort by Borda score
        sorted_items = sorted(borda_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [all_items[item_id] for item_id, score in sorted_items]
    
    def get_fusion_stats(self) -> Dict[str, Any]:
        """Get fusion statistics"""
        return {
            'fusion_stats': self.fusion_stats.copy(),
            'fusion_config': {
                'max_results': self.fusion_config.max_results,
                'min_score_threshold': self.fusion_config.min_score_threshold,
                'diversity_weight': self.fusion_config.diversity_weight,
                'recency_weight': self.fusion_config.recency_weight
            },
            'source_weights': self.source_weights.copy(),
            'section_weights': dict(self.section_weights)
        }
    
    def update_fusion_config(self, **kwargs):
        """Update fusion configuration"""
        for key, value in kwargs.items():
            if hasattr(self.fusion_config, key):
                setattr(self.fusion_config, key, value)
                self.logger.info(f"Updated fusion config: {key} = {value}")
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check of fusion system"""
        health = {
            'status': 'healthy',
            'issues': [],
            'components': {
                'fusion_algorithms': 'active',
                'scoring_system': 'active',
                'deduplication': 'active'
            }
        }
        
        # Check configuration
        if self.fusion_config.max_results <= 0:
            health['status'] = 'degraded'
            health['issues'].append('Invalid max_results configuration')
        
        if not self.source_weights:
            health['status'] = 'degraded'
            health['issues'].append('Source weights not configured')
        
        return health
