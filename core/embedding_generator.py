"""
Embedding generation functionality using OpenAI's API.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI

from config import (
    OPENAI_API_KEY, EMBEDDING_MODEL, DEFAULT_RETRY_ATTEMPTS, 
    EXPONENTIAL_BACKOFF_BASE, API_DELAY_SECONDS
)

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Handles embedding generation using OpenAI's embedding API.
    """
    
    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize the embedding generator.
        
        Args:
            api_key: OpenAI API key. If None, uses config default.
        """
        self.client = OpenAI(api_key=api_key or OPENAI_API_KEY)
        self.model = EMBEDDING_MODEL
        self.cache: Dict[str, List[float]] = {}
        
    def generate_embedding(
        self, 
        text: str, 
        max_retries: int = DEFAULT_RETRY_ATTEMPTS
    ) -> List[float]:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to embed
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of embedding values
            
        Raises:
            Exception: If all retry attempts fail
        """
        # Check cache first
        if text in self.cache:
            logger.debug(f"Cache hit for text: {text[:50]}...")
            return self.cache[text]
        
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=text,
                    encoding_format="float"
                )
                
                embedding = response.data[0].embedding
                self.cache[text] = embedding
                logger.debug(f"Generated embedding for text: {text[:50]}...")
                return embedding
                
            except Exception as e:
                logger.warning(
                    f"Embedding attempt {attempt + 1}/{max_retries} failed "
                    f"for text '{text[:50]}...': {e}"
                )
                if attempt < max_retries - 1:
                    sleep_time = EXPONENTIAL_BACKOFF_BASE ** attempt
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Failed to generate embedding for text: {text[:50]}...")
                    raise
    
    def generate_batch_embeddings(
        self, 
        texts: List[str], 
        delay_between_requests: float = API_DELAY_SECONDS
    ) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            delay_between_requests: Delay between API requests
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        total_texts = len(texts)
        
        logger.info(f"Generating embeddings for {total_texts} texts...")
        
        for i, text in enumerate(texts, 1):
            logger.info(f"Processing text {i}/{total_texts}")
            
            try:
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)
                
                # Add delay to respect rate limits
                if i < total_texts:  # Don't delay after the last request
                    time.sleep(delay_between_requests)
                    
            except Exception as e:
                logger.error(f"Failed to process text {i}: {e}")
                # Add zero vector as fallback
                embeddings.append([0.0] * 1536)
        
        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_cache_size(self) -> int:
        """Get the current cache size."""
        return len(self.cache)