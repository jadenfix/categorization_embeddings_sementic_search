#!/usr/bin/env python3
"""
Complete venue classification pipeline for all venues in the dataset.
Processes all venues with semantic search and LLM fallback, then saves comprehensive results.
"""

import sys
import pandas as pd
from typing import Any, Dict

from config import (
    CATEGORY_EMBEDDINGS_BASE,
    METADATA_EXTENSION,
    NUMPY_EXTENSION,
    validate_config,
)
from core.embedding_generator import EmbeddingGenerator
from core.search_client import SearchClient
from core.category_matcher import CategoryMatcher
from core.intelligent_category_selector import IntelligentCategorySelector, VenueInfo, CategoryCandidate
from core.llm_fallback import apply_llm_fallback_to_results
from utils.data_loader import DataLoader
from utils.logger_config import setup_logging, get_logger

# Set up logger
setup_logging()
logger = get_logger(__name__)

def load_foursquare_categories() -> pd.DataFrame:
    """Load Foursquare categories from CSV file."""
    return DataLoader.load_csv_data('data/foursquare_categories - foursquare_categories.csv')

def classify_venue(venue_data: Dict[str, str], components: Dict) -> Dict[str, Any]:
    """
    Classify a single venue using the full pipeline.
    
    Args:
        venue_data: Dict with 'name', 'address' keys
        components: Dict with initialized classification components
        
    Returns:
        Classification result dict
    """
    embedding_generator = components['embedding_generator']
    search_client = components['search_client']
    category_matcher = components['category_matcher']
    intelligent_selector = components['intelligent_selector']
    
    try:
        # Step 1: Search for venue information
        search_query = f"{venue_data['name']} {venue_data['address']}"
        search_results = search_client.search_venue_information(search_query)
        
        # Step 2: Create enriched text
        enriched_text = f"{venue_data['name']}, {venue_data['address']} - {search_results}"
        
        # Step 3: Generate embedding
        venue_embedding = embedding_generator.generate_embedding(enriched_text)
        
        # Step 4: Get top category candidates
        top_matches = category_matcher.find_top_k_matches(venue_embedding, k=4)
        
        # Step 5: Use intelligent category selector
        venue_info = VenueInfo(
            name=venue_data['name'],
            address=venue_data['address'],
            enriched_text=enriched_text
        )
        
        candidates = [
            CategoryCandidate(label=label, score=score)
            for _, label, score in top_matches
        ]
        
        selected_category = intelligent_selector.select_best_category(venue_info, candidates)
        
        # Find the corresponding similarity score
        similarity_score = 0.0
        category_name = "None"
        
        if selected_category:
            for name, label, score in top_matches:
                if label == selected_category:
                    similarity_score = score
                    category_name = name
                    break
        
        return {
            'name': venue_data['name'],
            'address': venue_data['address'],
            'enriched_text': enriched_text,
            'category_name': category_name,
            'category_label': selected_category or 'None',
            'similarity_score': similarity_score,
            'status': 'success',
            'top_matches': top_matches
        }
        
    except Exception as e:
        logger.error(f"Classification failed for {venue_data['name']}: {e}")
        return {
            'name': venue_data['name'],
            'address': venue_data['address'],
            'enriched_text': f"{venue_data['name']}, {venue_data['address']} - Error during search",
            'category_name': 'None',
            'category_label': 'None',
            'similarity_score': 0.0,
            'status': 'error',
            'error': str(e)
        }

def process_all_venues():
    """Process all venues from the data CSV file."""
    logger.info("Starting complete venue classification pipeline...")

    try:
        validate_config()
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        return {"error": str(exc)}
    
    # Load venue data
    venues_df = DataLoader.load_csv_data('data/foursquare_categories - data.csv')
    logger.info(f"Loaded {len(venues_df)} venues for processing")
    
    # Initialize components
    try:
        embedding_generator = EmbeddingGenerator()
        search_client = SearchClient()
        
        # Load category embeddings
        embeddings_path = f"{CATEGORY_EMBEDDINGS_BASE}{NUMPY_EXTENSION}"
        metadata_path = f"{CATEGORY_EMBEDDINGS_BASE}{METADATA_EXTENSION}"
        
        embeddings, metadata = DataLoader.load_embeddings_data(embeddings_path, metadata_path)
        category_matcher = CategoryMatcher(embeddings, metadata)
        intelligent_selector = IntelligentCategorySelector()
        
        components = {
            'embedding_generator': embedding_generator,
            'search_client': search_client,
            'category_matcher': category_matcher,
            'intelligent_selector': intelligent_selector
        }
        
        logger.info("Successfully initialized all components")
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        return {"error": f"Component initialization failed: {e}"}
    
    # Load categories for fallback
    categories_df = load_foursquare_categories()
    all_categories = categories_df['Category Label'].tolist()
    logger.info(f"Loaded {len(all_categories)} categories")
    
    # Process venues
    results = []
    for idx, row in venues_df.iterrows():
        venue_data = {
            'name': row['name'],
            'address': row['address']
        }
        
        logger.info(f"Processing venue {idx+1}/{len(venues_df)}: {venue_data['name']}")
        result = classify_venue(venue_data, components)
        results.append(result)
    
    # Apply LLM fallback to venues classified as "None"
    logger.info("Applying LLM fallback to 'None' classifications...")
    final_results = apply_llm_fallback_to_results(results, all_categories)
    
    # Report execution and categorization as separate outcomes. Successful
    # execution does not imply that the selector returned a category.
    total_venues = len(final_results)
    execution_succeeded = len(
        [r for r in final_results if str(r.get('status', '')).casefold() == 'success']
    )
    abstention_tokens = {'', 'none', 'null', 'unknown'}
    classified_venues = len(
        [
            r
            for r in final_results
            if str(r.get('status', '')).casefold() == 'success'
            and str(r.get('category_label') or '').strip().casefold()
            not in abstention_tokens
        ]
    )
    abstained_venues = execution_succeeded - classified_venues
    failed_venues = total_venues - execution_succeeded
    classification_coverage = (
        classified_venues / total_venues * 100 if total_venues else 0.0
    )
    
    fallback_reclassified = len([r for r in final_results if r.get('fallback_status') == 'llm_reclassified'])
    fallback_no_match = len([r for r in final_results if r.get('fallback_status') == 'llm_no_match'])
    
    # Save results to CSV
    results_df = pd.DataFrame(final_results)
    
    # Select columns for final CSV
    output_columns = [
        'name', 'address', 'enriched_text', 'category_name', 
        'category_label', 'similarity_score', 'status', 'fallback_status'
    ]
    
    final_df = results_df[output_columns].copy()
    output_file = 'results/complete_venue_classification_results.csv'
    final_df.to_csv(output_file, index=False)
    
    logger.info(f"Results saved to {output_file}")
    
    # Summary statistics
    logger.info("\n" + "="*60)
    logger.info("COMPLETE VENUE CLASSIFICATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total venues processed: {total_venues}")
    logger.info(f"Execution succeeded: {execution_succeeded}")
    logger.info(f"Classified: {classified_venues}")
    logger.info(f"Abstained: {abstained_venues}")
    logger.info(f"Failed: {failed_venues}")
    logger.info(f"Observed classification coverage: {classification_coverage:.1f}%")
    logger.info(f"LLM fallback reclassified: {fallback_reclassified}")
    logger.info(f"LLM fallback no match: {fallback_no_match}")
    
    return {
        'total_venues': total_venues,
        'execution_succeeded': execution_succeeded,
        'classified_venues': classified_venues,
        'abstained_venues': abstained_venues,
        'failed_venues': failed_venues,
        'classification_coverage': classification_coverage,
        'fallback_reclassified': fallback_reclassified,
        'fallback_no_match': fallback_no_match,
        'output_file': output_file,
        'results': final_results
    }

def main():
    """Run complete venue classification pipeline."""
    return process_all_venues()

if __name__ == "__main__":
    result = main()
    sys.exit(1 if result.get('error') else 0)
