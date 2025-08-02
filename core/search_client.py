"""
Search functionality using Serper API.
"""

import time
import logging
import requests
from typing import Dict, Optional, List
from requests.exceptions import RequestException

from config import (
    SERPER_API_KEY, SERPER_SEARCH_URL, DEFAULT_RETRY_ATTEMPTS,
    EXPONENTIAL_BACKOFF_BASE, SEARCH_RESULTS_LIMIT
)

logger = logging.getLogger(__name__)


class SearchClient:
    """
    Handles search operations using Serper API.
    """
    
    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize the search client.
        
        Args:
            api_key: Serper API key. If None, uses config default.
        """
        self.api_key = api_key or SERPER_API_KEY
        self.base_url = SERPER_SEARCH_URL
        self.cache: Dict[str, str] = {}
        
    def search_venue_information(
        self, 
        query: str, 
        max_results: int = SEARCH_RESULTS_LIMIT,
        max_retries: int = DEFAULT_RETRY_ATTEMPTS
    ) -> str:
        """
        Search for venue information and return concatenated snippets.
        
        Args:
            query: Search query string
            max_results: Maximum number of search results to process
            max_retries: Maximum number of retry attempts
            
        Returns:
            Concatenated snippets from search results
        """
        # Check cache first
        if query in self.cache:
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return self.cache[query]
        
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }
        payload = {"q": query}
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.base_url, 
                    headers=headers, 
                    json=payload, 
                    timeout=30
                )
                response.raise_for_status()
                
                search_data = response.json()
                snippets = self._extract_snippets(search_data, max_results)
                
                # Join snippets with separator
                combined_snippets = " | ".join(snippets) if snippets else "No search results found"
                
                # Cache the result
                self.cache[query] = combined_snippets
                logger.debug(f"Search successful for query: {query[:50]}...")
                return combined_snippets
                
            except RequestException as e:
                logger.warning(
                    f"Search attempt {attempt + 1}/{max_retries} failed "
                    f"for query '{query[:50]}...': {e}"
                )
                if attempt < max_retries - 1:
                    sleep_time = EXPONENTIAL_BACKOFF_BASE ** attempt
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Failed to search for query: {query[:50]}...")
                    return "Search failed"
                    
            except Exception as e:
                logger.warning(f"Unexpected error in search attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    sleep_time = EXPONENTIAL_BACKOFF_BASE ** attempt
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Failed to search for query: {query[:50]}...")
                    return "Search failed"
    
    def _extract_snippets(self, search_data: Dict, max_results: int) -> List[str]:
        """
        Extract snippets from search results.
        
        Args:
            search_data: Raw search response data
            max_results: Maximum number of results to extract
            
        Returns:
            List of snippet strings
        """
        snippets = []
        
        if 'organic' in search_data:
            for result in search_data['organic'][:max_results]:
                if 'snippet' in result:
                    snippets.append(result['snippet'])
        
        return snippets
    
    def clear_cache(self) -> None:
        """Clear the search cache."""
        self.cache.clear()
        logger.info("Search cache cleared")
    
    def get_cache_size(self) -> int:
        """Get the current cache size."""
        return len(self.cache)