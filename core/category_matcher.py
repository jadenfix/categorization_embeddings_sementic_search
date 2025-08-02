"""
Category matching functionality using cosine similarity.
"""

import logging
import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class CategoryMatcher:
    """
    Handles category matching using cosine similarity between embeddings.
    """
    
    def __init__(
        self, 
        category_embeddings: np.ndarray, 
        category_metadata: Dict[str, Any]
    ) -> None:
        """
        Initialize the category matcher.
        
        Args:
            category_embeddings: Precomputed category embeddings array
            category_metadata: Category metadata dictionary
        """
        self.category_embeddings = category_embeddings
        self.category_metadata = category_metadata
        
        # Validate inputs
        self._validate_inputs()
        
        logger.info(
            f"Initialized CategoryMatcher with {len(self.category_metadata['categories'])} categories"
        )
    
    def _validate_inputs(self) -> None:
        """Validate the input embeddings and metadata."""
        if self.category_embeddings is None:
            raise ValueError("Category embeddings cannot be None")
        
        if not isinstance(self.category_metadata, dict):
            raise ValueError("Category metadata must be a dictionary")
        
        if 'categories' not in self.category_metadata:
            raise ValueError("Category metadata must contain 'categories' key")
        
        expected_categories = len(self.category_metadata['categories'])
        actual_embeddings = self.category_embeddings.shape[0]
        
        if expected_categories != actual_embeddings:
            raise ValueError(
                f"Mismatch between metadata categories ({expected_categories}) "
                f"and embedding count ({actual_embeddings})"
            )
    
    def find_best_category_match(self, venue_embedding: List[float]) -> Tuple[str, str, float]:
        """
        Find the best category match for a venue embedding.
        
        Args:
            venue_embedding: The venue's embedding vector
            
        Returns:
            Tuple of (category_name, category_label, similarity_score)
        """
        # Convert venue embedding to numpy array and reshape for cosine_similarity
        venue_vector = np.array(venue_embedding).reshape(1, -1)
        
        # Calculate cosine similarities with all category embeddings
        similarities = cosine_similarity(venue_vector, self.category_embeddings)[0]
        
        # Find the index of the highest similarity
        best_match_idx = np.argmax(similarities)
        best_similarity = similarities[best_match_idx]
        
        # Get the corresponding category information
        best_category = self.category_metadata['categories'][best_match_idx]
        category_name = best_category['Category Name']
        category_label = best_category['Category Label']
        
        logger.debug(
            f"Best match: {category_name} (similarity: {best_similarity:.4f})"
        )
        
        return category_name, category_label, float(best_similarity)
    
    def find_top_k_matches(
        self, 
        venue_embedding: List[float], 
        k: int = 5
    ) -> List[Tuple[str, str, float]]:
        """
        Find the top-k category matches for a venue embedding.
        
        Args:
            venue_embedding: The venue's embedding vector
            k: Number of top matches to return
            
        Returns:
            List of tuples (category_name, category_label, similarity_score)
        """
        # Convert venue embedding to numpy array and reshape for cosine_similarity
        venue_vector = np.array(venue_embedding).reshape(1, -1)
        
        # Calculate cosine similarities with all category embeddings
        similarities = cosine_similarity(venue_vector, self.category_embeddings)[0]
        
        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            category = self.category_metadata['categories'][idx]
            category_name = category['Category Name']
            category_label = category['Category Label']
            similarity_score = float(similarities[idx])
            
            results.append((category_name, category_label, similarity_score))
        
        return results
    
    def get_category_count(self) -> int:
        """Get the total number of categories."""
        return len(self.category_metadata['categories'])
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.category_embeddings.shape[1]
    
    def get_all_category_labels(self) -> List[str]:
        """Get all available category labels."""
        return [category['Category Label'] for category in self.category_metadata['categories']]