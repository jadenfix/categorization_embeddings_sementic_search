"""
File management utilities for saving and loading various formats.
"""

import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from pathlib import Path

from config import (
    NUMPY_EXTENSION, METADATA_EXTENSION, CSV_EXTENSION, PICKLE_EXTENSION
)

logger = logging.getLogger(__name__)


class FileManager:
    """
    Handles file operations for the venue categorization system.
    """
    
    @staticmethod
    def ensure_directory_exists(directory_path: str) -> None:
        """
        Ensure that a directory exists, creating it if necessary.
        
        Args:
            directory_path: Path to the directory
        """
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory_path}")
    
    @staticmethod
    def save_embeddings_data(
        embeddings_df: pd.DataFrame, 
        output_base_path: str,
        model_name: str
    ) -> None:
        """
        Save embeddings DataFrame to multiple formats.
        
        Args:
            embeddings_df: DataFrame containing embeddings
            output_base_path: Base path for output files (without extension)
            model_name: Name of the embedding model used
        """
        try:
            # Ensure output directory exists
            output_dir = Path(output_base_path).parent
            FileManager.ensure_directory_exists(str(output_dir))
            
            # Save as CSV (embeddings as JSON strings)
            csv_df = embeddings_df.copy()
            csv_df['embedding'] = csv_df['embedding'].apply(json.dumps)
            csv_path = f"{output_base_path}{CSV_EXTENSION}"
            csv_df.to_csv(csv_path, index=False)
            logger.info(f"Saved embeddings to CSV: {csv_path}")
            
            # Save as pickle (preserves numpy arrays)
            pickle_path = f"{output_base_path}{PICKLE_EXTENSION}"
            embeddings_df.to_pickle(pickle_path)
            logger.info(f"Saved embeddings to pickle: {pickle_path}")
            
            # Save embeddings as numpy array
            embeddings_array = np.array(embeddings_df['embedding'].tolist())
            numpy_path = f"{output_base_path}{NUMPY_EXTENSION}"
            np.save(numpy_path, embeddings_array)
            logger.info(f"Saved embeddings array to: {numpy_path}")
            
            # Save metadata as JSON
            FileManager._save_embeddings_metadata(
                embeddings_df, output_base_path, model_name
            )
            
        except Exception as e:
            logger.error(f"Error saving embeddings data: {e}")
            raise
    
    @staticmethod
    def _save_embeddings_metadata(
        embeddings_df: pd.DataFrame, 
        output_base_path: str,
        model_name: str
    ) -> None:
        """
        Save embeddings metadata to JSON file.
        
        Args:
            embeddings_df: DataFrame containing embeddings
            output_base_path: Base path for output files
            model_name: Name of the embedding model used
        """
        # Determine metadata structure based on DataFrame columns
        if 'Category ID' in embeddings_df.columns:
            # Category embeddings metadata
            metadata = {
                'model': model_name,
                'total_categories': len(embeddings_df),
                'embedding_dimension': len(embeddings_df['embedding'].iloc[0]),
                'categories': embeddings_df[
                    ['Category ID', 'Category Name', 'Category Label']
                ].to_dict('records')
            }
        else:
            # Venue embeddings metadata
            metadata = {
                'model': model_name,
                'total_venues': len(embeddings_df),
                'embedding_dimension': len(embeddings_df['embedding'].iloc[0]),
                'venues': embeddings_df[
                    ['name', 'address', 'category']
                ].to_dict('records') if 'category' in embeddings_df.columns else embeddings_df[
                    ['name', 'address']
                ].to_dict('records')
            }
        
        metadata_path = f"{output_base_path}{METADATA_EXTENSION}"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved metadata to: {metadata_path}")
    
    @staticmethod
    def save_results_to_csv(
        results: List[Dict[str, Any]], 
        output_path: str
    ) -> None:
        """
        Save results list to CSV file.
        
        Args:
            results: List of result dictionaries
            output_path: Path to save the CSV file
        """
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            FileManager.ensure_directory_exists(str(output_dir))
            
            # Convert to DataFrame and save
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_path, index=False)
            logger.info(f"Saved {len(results)} results to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results to CSV: {e}")
            raise
    
    @staticmethod
    def save_json_data(data: Dict[str, Any], output_path: str) -> None:
        """
        Save dictionary data to JSON file.
        
        Args:
            data: Dictionary to save
            output_path: Path to save the JSON file
        """
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            FileManager.ensure_directory_exists(str(output_dir))
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved JSON data to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving JSON data: {e}")
            raise
    
    @staticmethod
    def get_file_size_mb(file_path: str) -> float:
        """
        Get file size in megabytes.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File size in MB
        """
        try:
            size_bytes = Path(file_path).stat().st_size
            return size_bytes / (1024 * 1024)
        except Exception:
            return 0.0