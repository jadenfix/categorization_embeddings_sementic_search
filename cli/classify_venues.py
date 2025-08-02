#!/usr/bin/env python3
"""
CLI script for classifying venues using precomputed category embeddings.
"""

import sys
import time
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import (
    validate_config, VENUES_FILE, VENUE_MATCHES_FILE, VENUE_MATCHES_WITH_FALLBACK_FILE,
    CATEGORY_EMBEDDINGS_BASE, NUMPY_EXTENSION, METADATA_EXTENSION,
    API_DELAY_SECONDS
)
from core.embedding_generator import EmbeddingGenerator
from core.search_client import SearchClient
from core.category_matcher import CategoryMatcher
from core.intelligent_category_selector import IntelligentCategorySelector, VenueInfo, CategoryCandidate
from core.llm_fallback import apply_llm_fallback_to_results
from utils.data_loader import DataLoader
from utils.file_manager import FileManager
from utils.logger_config import setup_logging, get_logger


class VenueClassifier:
    """
    Handles the complete venue classification pipeline.
    """
    
    def __init__(self) -> None:
        """Initialize the venue classifier."""
        self.logger = get_logger(__name__)
        self.embedding_generator = EmbeddingGenerator()
        self.search_client = SearchClient()
        self.category_matcher = None
        self.intelligent_selector = IntelligentCategorySelector()
        
    def load_category_embeddings(self) -> None:
        """Load precomputed category embeddings and initialize matcher."""
        embeddings_path = f"{CATEGORY_EMBEDDINGS_BASE}{NUMPY_EXTENSION}"
        metadata_path = f"{CATEGORY_EMBEDDINGS_BASE}{METADATA_EXTENSION}"
        
        embeddings, metadata = DataLoader.load_embeddings_data(
            embeddings_path, metadata_path
        )
        
        self.category_matcher = CategoryMatcher(embeddings, metadata)
        self.logger.info(
            f"Loaded {self.category_matcher.get_category_count()} category embeddings"
        )
    
    def create_enriched_venue_text(self, name: str, address: str) -> str:
        """
        Create enriched text for a venue using search results.
        
        Args:
            name: Venue name
            address: Venue address
            
        Returns:
            Enriched text string
        """
        # Create search query
        search_query = f"{name} {address}".strip()
        
        # Get search snippets
        search_snippets = self.search_client.search_venue_information(search_query)
        
        # Create enriched text in format: {name}, {address} - {snippets}
        enriched_text = f"{name}, {address} - {search_snippets}"
        
        return enriched_text
    
    def classify_single_venue(self, venue_row: pd.Series) -> Dict[str, Any]:
        """
        Classify a single venue.
        
        Args:
            venue_row: Pandas Series containing venue data
            
        Returns:
            Dictionary with classification results
        """
        venue_name = str(venue_row['name']) if pd.notna(venue_row['name']) else ""
        venue_address = str(venue_row['address']) if pd.notna(venue_row['address']) else ""
        
        try:
            # Create enriched text
            enriched_text = self.create_enriched_venue_text(venue_name, venue_address)
            self.logger.debug(f"Enriched text length: {len(enriched_text)} characters")
            
            # Generate embedding
            venue_embedding = self.embedding_generator.generate_embedding(enriched_text)
            self.logger.debug(f"Generated embedding with {len(venue_embedding)} dimensions")
            
            # Get top 4 category candidates
            top_matches = self.category_matcher.find_top_k_matches(venue_embedding, k=4)
            
            # Create candidates for intelligent selector
            candidates = [
                CategoryCandidate(label=label, score=score)
                for _, label, score in top_matches
            ]
            
            # Use intelligent selector to choose the best category
            venue_info = VenueInfo(
                name=venue_name,
                address=venue_address,
                enriched_text=enriched_text
            )
            
            selected_category_label = self.intelligent_selector.select_best_category(
                venue_info, candidates
            )
            
            if selected_category_label:
                # Find the corresponding category name and similarity score
                for name, label, score in top_matches:
                    if label == selected_category_label:
                        category_name = name
                        category_label = label
                        similarity_score = score
                        break
                else:
                    # Fallback if not found (shouldn't happen)
                    category_name = top_matches[0][0]
                    category_label = selected_category_label
                    similarity_score = top_matches[0][2]
            else:
                # No suitable category found
                category_name = "None"
                category_label = "None"
                similarity_score = 0.0
            
            return {
                'name': venue_name,
                'address': venue_address,
                'enriched_text': enriched_text,
                'category_name': category_name,
                'category_label': category_label,
                'similarity_score': similarity_score,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to classify venue {venue_name}: {e}")
            return {
                'name': venue_name,
                'address': venue_address,
                'enriched_text': f"{venue_name}, {venue_address} - Processing failed",
                'category_name': "Unknown",
                'category_label': "Unknown",
                'similarity_score': 0.0,
                'status': 'failed'
            }
    
    def classify_venues(self, venues_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Classify all venues in the DataFrame.
        
        Args:
            venues_df: DataFrame containing venue data
            
        Returns:
            List of classification results
        """
        results = []
        total_venues = len(venues_df)
        
        self.logger.info(f"Classifying {total_venues} venues...")
        
        for idx, row in venues_df.iterrows():
            self.logger.info(f"Processing venue {idx + 1}/{total_venues}: {row['name']}")
            
            # Classify the venue
            result = self.classify_single_venue(row)
            results.append(result)
            
            # Log the result
            if result['status'] == 'success':
                self.logger.info(
                    f"  → {result['category_name']} "
                    f"(similarity: {result['similarity_score']:.4f})"
                )
            else:
                self.logger.warning(f"  → Classification failed")
            
            # Add delay to respect rate limits
            if idx < total_venues - 1:  # Don't delay after the last venue
                time.sleep(API_DELAY_SECONDS)
        
        return results


def print_classification_summary(results: List[Dict[str, Any]]) -> None:
    """
    Print a summary of classification results.
    
    Args:
        results: List of classification results
    """
    logger = get_logger(__name__)
    
    logger.info("\n" + "="*80)
    logger.info("VENUE CLASSIFICATION SUMMARY")
    logger.info("="*80)
    
    # Calculate statistics
    successful_results = [r for r in results if r['status'] == 'success']
    failed_results = [r for r in results if r['status'] == 'failed']
    
    logger.info(f"Total venues processed: {len(results)}")
    logger.info(f"Successfully classified: {len(successful_results)}")
    logger.info(f"Failed classifications: {len(failed_results)}")
    
    if successful_results:
        # Calculate average similarity
        avg_similarity = sum(r['similarity_score'] for r in successful_results) / len(successful_results)
        logger.info(f"Average similarity score: {avg_similarity:.4f}")
        
        # Count unique categories
        unique_categories = len(set(r['category_label'] for r in successful_results))
        logger.info(f"Unique categories matched: {unique_categories}")
        
        # Show top categories
        category_counts = {}
        for result in successful_results:
            cat = result['category_label']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        logger.info("Most common category matches:")
        for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"  {cat}: {count} venues")


def main() -> int:
    """
    Main function for venue classification.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    setup_logging()
    logger = get_logger(__name__)
    
    try:
        # Validate configuration
        validate_config()
        logger.info("Starting venue classification...")
        
        # Load venue data
        venues_df = DataLoader.load_csv_data(VENUES_FILE)
        DataLoader.validate_venue_data(venues_df)
        DataLoader.display_dataframe_info(venues_df, "Venues")
        
        # Initialize classifier
        classifier = VenueClassifier()
        
        # Load category embeddings
        classifier.load_category_embeddings()
        
        # Classify venues
        results = classifier.classify_venues(venues_df)
        
        # Save initial results
        FileManager.save_results_to_csv(results, VENUE_MATCHES_FILE)
        
        # Print initial summary
        print_classification_summary(results)
        
        # Apply LLM fallback for "None" classifications
        none_count = len([r for r in results if r['category_label'] == 'None'])
        if none_count > 0:
            logger.info(f"Found {none_count} venues classified as 'None'. Applying LLM fallback...")
            
            # Get all categories for fallback
            all_categories = classifier.category_matcher.get_all_category_labels()
            
            # Apply LLM fallback
            updated_results = apply_llm_fallback_to_results(results, all_categories)
            
            # Save updated results
            FileManager.save_results_to_csv(updated_results, VENUE_MATCHES_WITH_FALLBACK_FILE)
            
            # Print final summary
            print("\n" + "="*80)
            print("FINAL CLASSIFICATION SUMMARY (WITH LLM FALLBACK)")
            print("="*80)
            print_classification_summary(updated_results)
        else:
            logger.info("No venues classified as 'None' - LLM fallback not needed")
        
        logger.info("Venue classification completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error in venue classification: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())