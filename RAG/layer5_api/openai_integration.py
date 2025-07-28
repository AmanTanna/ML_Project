"""
OpenAI Integration Module
========================

Integration with OpenAI GPT models for enhanced chat and RAG responses.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime
import logging
import asyncio
from contextlib import asynccontextmanager

from .models import OpenAIRequest, OpenAIResponse, RAGChatRequest, RAGChatResponse, QueryResult

logger = logging.getLogger(__name__)

class OpenAIManager:
    """Manages OpenAI API interactions"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Configuration
        self.default_model = os.getenv('OPENAI_DEFAULT_MODEL', 'gpt-3.5-turbo')
        self.max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', '2000'))
        self.temperature = float(os.getenv('OPENAI_TEMPERATURE', '0.7'))
        self.timeout = int(os.getenv('OPENAI_TIMEOUT', '30'))
        
        # Rate limiting and retry configuration
        self.max_retries = 3
        self.retry_delay = 1.0
        
        logger.info("OpenAI Manager initialized successfully")
    
    async def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Create a chat completion using OpenAI API"""
        start_time = time.time()
        
        try:
            # Use provided parameters or defaults
            model = model or self.default_model
            temperature = temperature if temperature is not None else self.temperature
            max_tokens = max_tokens or self.max_tokens
            
            # Validate messages
            if not messages:
                raise ValueError("Messages cannot be empty")
            
            for msg in messages:
                if 'role' not in msg or 'content' not in msg:
                    raise ValueError("Each message must have 'role' and 'content' fields")
            
            logger.info(f"Creating chat completion with model: {model}")
            
            # Make API call with retry logic - Updated for OpenAI v1.0+
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=self.api_key)
            
            response = await self._make_api_call_with_retry(
                client.chat.completions.create,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            processing_time = time.time() - start_time
            
            if stream:
                return response
            
            # Extract response content
            content = response.choices[0].message.content
            usage = {
                'total_tokens': response.usage.total_tokens,
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens
            }
            
            result = {
                'response': content,
                'model': model,
                'usage': usage,
                'processing_time': processing_time
            }
            
            logger.info(f"Chat completion successful. Processing time: {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            raise Exception(f"Chat completion failed: {str(e)}")
    
    async def create_rag_enhanced_response(
        self,
        query: str,
        context_results: List[QueryResult],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_context_length: int = 4000
    ) -> Dict[str, Any]:
        """Create a RAG-enhanced response using context from the query engine"""
        start_time = time.time()
        
        try:
            # Prepare context from query results
            context_text = self._prepare_rag_context(context_results, max_context_length)
            
            # Create system prompt with context
            system_prompt = self._create_rag_system_prompt(context_text)
            
            # Prepare messages
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            # Get chat completion
            response = await self.create_chat_completion(
                messages=messages,
                model=model,
                temperature=temperature
            )
            
            processing_time = time.time() - start_time
            
            result = {
                'response': response['response'],
                'sources': context_results,
                'rag_context': {
                    'context_length': len(context_text),
                    'num_sources': len(context_results),
                    'context_truncated': len(context_text) >= max_context_length
                },
                'processing_time': processing_time,
                'model': response['model'],
                'usage': response['usage']
            }
            
            logger.info(f"RAG-enhanced response created successfully. Processing time: {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error creating RAG-enhanced response: {str(e)}")
            raise Exception(f"RAG response failed: {str(e)}")
    
    def _prepare_rag_context(self, results: List[QueryResult], max_length: int) -> str:
        """Prepare context text from query results"""
        context_parts = []
        current_length = 0
        
        for result in results:
            # Format each result with metadata
            source_info = ""
            if result.company:
                source_info += f"Company: {result.company}"
            if result.source_document:
                source_info += f", Document: {result.source_document}"
            if result.section:
                source_info += f", Section: {result.section}"
            
            context_part = f"[Source: {source_info}]\n{result.content}\n"
            
            # Check if adding this part would exceed the limit
            if current_length + len(context_part) > max_length:
                break
            
            context_parts.append(context_part)
            current_length += len(context_part)
        
        return "\n---\n".join(context_parts)
    
    def _create_rag_system_prompt(self, context: str) -> str:
        """Create system prompt for RAG-enhanced responses"""
        return f"""You are a helpful AI assistant specializing in financial document analysis. You have access to relevant context from financial documents to answer user questions.

CONTEXT FROM FINANCIAL DOCUMENTS:
{context}

INSTRUCTIONS:
1. Use the provided context to answer the user's question accurately and comprehensively
2. If the context doesn't contain sufficient information, clearly state this limitation
3. Cite specific sources when referencing information from the context
4. Provide balanced, objective analysis of financial information
5. If asked about specific companies, focus on information from their documents
6. Highlight key insights and important details from the context
7. If the question cannot be answered with the provided context, suggest what additional information might be needed

Remember to be precise, factual, and helpful in your responses."""
    
    async def _make_api_call_with_retry(self, api_func, **kwargs):
        """Make API call with exponential backoff retry logic"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return await api_func(**kwargs)
            
            except (openai.error.RateLimitError, openai.error.ServiceUnavailableError) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)
                    logger.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries}). Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"API call failed after {self.max_retries} attempts")
                    raise
            
            except Exception as e:
                # Don't retry for other types of errors
                logger.error(f"Non-retryable error in API call: {str(e)}")
                raise
        
        raise last_exception
    
    async def create_streaming_response(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> AsyncGenerator[str, None]:
        """Create a streaming chat completion"""
        try:
            model = model or self.default_model
            temperature = temperature if temperature is not None else self.temperature
            
            logger.info(f"Creating streaming response with model: {model}")
            
            response = await self._make_api_call_with_retry(
                openai.ChatCompletion.acreate,
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True
            )
            
            async for chunk in response:
                if chunk.choices[0].delta.get('content'):
                    yield chunk.choices[0].delta.content
            
        except Exception as e:
            logger.error(f"Error in streaming response: {str(e)}")
            yield f"Error: {str(e)}"
    
    def get_available_models(self) -> List[str]:
        """Get list of available OpenAI models"""
        try:
            # For OpenAI v1.0+, return a predefined list of common models
            # In production, you might want to fetch this from OpenAI API
            chat_models = [
                'gpt-3.5-turbo',
                'gpt-3.5-turbo-16k',
                'gpt-4',
                'gpt-4-turbo-preview',
                'gpt-4-vision-preview'
            ]
            return chat_models
        except Exception as e:
            logger.error(f"Error fetching available models: {str(e)}")
            return [self.default_model]
    
    def estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count for text"""
        # Simple approximation: ~4 characters per token for English text
        return len(text) // 4
    
    def validate_request(self, request: OpenAIRequest) -> bool:
        """Validate OpenAI request parameters"""
        if not request.messages:
            return False
        
        # Check message format
        for message in request.messages:
            if not isinstance(message, dict):
                return False
            if 'role' not in message or 'content' not in message:
                return False
            if message['role'] not in ['system', 'user', 'assistant']:
                return False
        
        # Check token limits
        total_content = ' '.join([msg['content'] for msg in request.messages])
        estimated_tokens = self.estimate_tokens(total_content)
        
        if estimated_tokens > 8000:  # Conservative limit for context
            return False
        
        return True
    
    async def create_embeddings(self, texts: List[str], model: str = "text-embedding-ada-002") -> List[List[float]]:
        """Create embeddings for given texts"""
        try:
            response = await openai.Embedding.acreate(
                input=texts,
                engine=model
            )
            
            embeddings = [item['embedding'] for item in response['data']]
            logger.info(f"Created embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise Exception(f"Embedding creation failed: {str(e)}")

# Global OpenAI manager instance
openai_manager = None

def get_openai_manager() -> OpenAIManager:
    """Get or create OpenAI manager instance"""
    global openai_manager
    if openai_manager is None:
        openai_manager = OpenAIManager()
    return openai_manager

@asynccontextmanager
async def openai_client():
    """Async context manager for OpenAI client"""
    manager = get_openai_manager()
    try:
        yield manager
    except Exception as e:
        logger.error(f"Error in OpenAI client context: {str(e)}")
        raise
    finally:
        # Cleanup if needed
        pass

class OpenAIUsageTracker:
    """Track OpenAI API usage and costs"""
    
    def __init__(self):
        self.usage_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'estimated_cost': 0.0
        }
        
        # Cost per 1K tokens (approximate)
        self.model_costs = {
            'gpt-3.5-turbo': {'prompt': 0.0015, 'completion': 0.002},
            'gpt-4': {'prompt': 0.03, 'completion': 0.06},
            'gpt-4-turbo': {'prompt': 0.01, 'completion': 0.03}
        }
    
    def track_usage(self, usage: Dict[str, int], model: str, success: bool = True):
        """Track API usage"""
        self.usage_stats['total_requests'] += 1
        
        if success:
            self.usage_stats['successful_requests'] += 1
            self.usage_stats['total_tokens'] += usage.get('total_tokens', 0)
            self.usage_stats['prompt_tokens'] += usage.get('prompt_tokens', 0)
            self.usage_stats['completion_tokens'] += usage.get('completion_tokens', 0)
            
            # Estimate cost
            if model in self.model_costs:
                costs = self.model_costs[model]
                prompt_cost = (usage.get('prompt_tokens', 0) / 1000) * costs['prompt']
                completion_cost = (usage.get('completion_tokens', 0) / 1000) * costs['completion']
                self.usage_stats['estimated_cost'] += prompt_cost + completion_cost
        else:
            self.usage_stats['failed_requests'] += 1
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary"""
        success_rate = 0.0
        if self.usage_stats['total_requests'] > 0:
            success_rate = self.usage_stats['successful_requests'] / self.usage_stats['total_requests']
        
        return {
            **self.usage_stats,
            'success_rate': success_rate,
            'avg_tokens_per_request': (
                self.usage_stats['total_tokens'] / max(1, self.usage_stats['successful_requests'])
            )
        }

# Global usage tracker
usage_tracker = OpenAIUsageTracker()
