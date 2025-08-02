#!/usr/bin/env python3
"""
CLI script for generating category embeddings.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import (
    validate_config, CATEGORIES_FILE, CATEGORY_EMBEDDINGS_BASE, 
    EMBEDDING_MODEL
)
from core.embedding_generator import EmbeddingGenerator
from utils.data_loader import DataLoader
from utils.file_manager import FileManager
from utils.logger_config import setup_logging, get_logger


def analyze_category_hierarchy(categories_df) -> dict:
    """
    Analyze the category hierarchy structure.
    
    Args:
        categories_df: DataFrame containing category data
        
    Returns:
        Dictionary with analysis results
    """
    analysis = {
        'total_categories': len(categories_df),
        'unique_category_names': categories_df['Category Name'].nunique(),
        'unique_category_labels': categories_df['Category Label'].nunique(),
        'top_level_categories': [],
        'category_hierarchy_depth': {}
    }
    
    # Find top-level categories (no '>' in label)
    top_level = categories_df[
        ~categories_df['Category Label'].str.contains('>')
    ]['Category Label'].tolist()
    analysis['top_level_categories'] = top_level
    
    # Analyze hierarchy depth
    for _, row in categories_df.iterrows():
        label = row['Category Label']
        depth = label.count('>') + 1
        if depth not in analysis['category_hierarchy_depth']:
            analysis['category_hierarchy_depth'][depth] = 0
        analysis['category_hierarchy_depth'][depth] += 1
    
    return analysis


def main() -> int:
    """
    Main function for category embedding generation.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    setup_logging()
    logger = get_logger(__name__)
    
    try:
        # Validate configuration
        validate_config()
        logger.info("Starting category embedding generation...")
        
        # Load category data
        categories_df = DataLoader.load_csv_data(CATEGORIES_FILE)
        DataLoader.validate_category_data(categories_df)
        DataLoader.display_dataframe_info(categories_df, "Categories")
        
        # Analyze category hierarchy
        analysis = analyze_category_hierarchy(categories_df)
        logger.info("Category Analysis:")
        logger.info(f"  Total categories: {analysis['total_categories']}")
        logger.info(f"  Top-level categories: {len(analysis['top_level_categories'])}")
        logger.info(f"  Hierarchy depth distribution: {analysis['category_hierarchy_depth']}")
        
        # Initialize embedding generator
        embedding_generator = EmbeddingGenerator()
        
        # Generate embeddings for category labels
        logger.info("Generating category embeddings...")
        category_labels = categories_df['Category Label'].tolist()
        embeddings = embedding_generator.generate_batch_embeddings(category_labels)
        
        # Add embeddings to DataFrame
        categories_with_embeddings = categories_df.copy()
        categories_with_embeddings['embedding'] = embeddings
        
        # Save results
        logger.info("Saving category embeddings...")
        FileManager.save_embeddings_data(
            categories_with_embeddings, 
            CATEGORY_EMBEDDINGS_BASE,
            EMBEDDING_MODEL
        )
        
        # Display sample results
        logger.info("Sample category embeddings (first 5 dimensions):")
        for idx, row in categories_with_embeddings.head(3).iterrows():
            embedding_sample = row['embedding'][:5]
            logger.info(f"  {row['Category Label']}: {embedding_sample}")
        
        logger.info("Category embedding generation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error in category embedding generation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())