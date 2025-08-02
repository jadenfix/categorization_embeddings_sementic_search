"""
Data loading utilities for CSV files and embeddings.
"""

import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Handles loading of various data formats used in the system.
    """
    
    @staticmethod
    def load_csv_data(file_path: str) -> pd.DataFrame:
        """
        Load data from a CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            pd.errors.EmptyDataError: If the CSV is empty
        """
        try:
            logger.info(f"Loading CSV data from: {file_path}")
            
            if not Path(file_path).exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {len(df)} records from {file_path}")
            
            return df
            
        except pd.errors.EmptyDataError as e:
            logger.error(f"Empty CSV file: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading CSV data from {file_path}: {e}")
            raise
    
    @staticmethod
    def load_embeddings_data(
        embeddings_path: str, 
        metadata_path: str
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load precomputed embeddings and metadata.
        
        Args:
            embeddings_path: Path to the .npy embeddings file
            metadata_path: Path to the .json metadata file
            
        Returns:
            Tuple of (embeddings_array, metadata_dict)
            
        Raises:
            FileNotFoundError: If either file doesn't exist
        """
        try:
            # Load embeddings
            logger.info(f"Loading embeddings from: {embeddings_path}")
            
            if not Path(embeddings_path).exists():
                raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
            
            embeddings = np.load(embeddings_path)
            logger.info(f"Loaded embeddings with shape: {embeddings.shape}")
            
            # Load metadata
            logger.info(f"Loading metadata from: {metadata_path}")
            
            if not Path(metadata_path).exists():
                raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            logger.info(f"Loaded metadata for {metadata.get('total_categories', 0)} categories")
            
            return embeddings, metadata
            
        except Exception as e:
            logger.error(f"Error loading embeddings data: {e}")
            raise
    
    @staticmethod
    def display_dataframe_info(df: pd.DataFrame, name: str = "DataFrame") -> None:
        """
        Display basic information about a DataFrame.
        
        Args:
            df: DataFrame to analyze
            name: Name for logging purposes
        """
        logger.info(f"{name} Info:")
        logger.info(f"  Columns: {list(df.columns)}")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Show sample data
        if len(df) > 0:
            logger.info(f"Sample {name} records:")
            for idx, row in df.head(3).iterrows():
                # Create a summary of the row
                row_summary = " | ".join([f"{col}: {str(val)[:50]}" for col, val in row.items()])
                logger.info(f"  {idx}: {row_summary}")
    
    @staticmethod
    def validate_venue_data(df: pd.DataFrame) -> None:
        """
        Validate venue data structure.
        
        Args:
            df: Venue DataFrame to validate
            
        Raises:
            ValueError: If required columns are missing
        """
        required_columns = ['name', 'address']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in venue data: {missing_columns}")
        
        # Check for empty data
        if len(df) == 0:
            raise ValueError("Venue data is empty")
        
        logger.info("Venue data validation passed")
    
    @staticmethod
    def validate_category_data(df: pd.DataFrame) -> None:
        """
        Validate category data structure.
        
        Args:
            df: Category DataFrame to validate
            
        Raises:
            ValueError: If required columns are missing
        """
        required_columns = ['Category ID', 'Category Name', 'Category Label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in category data: {missing_columns}")
        
        # Check for empty data
        if len(df) == 0:
            raise ValueError("Category data is empty")
        
        logger.info("Category data validation passed")