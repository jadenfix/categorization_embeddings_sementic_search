#!/usr/bin/env python3
"""
CLI script for analyzing venue classification results.
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import VENUE_MATCHES_FILE, RESULTS_DIR
from utils.data_loader import DataLoader
from utils.file_manager import FileManager
from utils.logger_config import setup_logging, get_logger


class ResultsAnalyzer:
    """
    Analyzes venue classification results and generates insights.
    """
    
    def __init__(self, results_df: pd.DataFrame) -> None:
        """
        Initialize the results analyzer.
        
        Args:
            results_df: DataFrame containing classification results
        """
        self.results_df = results_df
        self.logger = get_logger(__name__)
        
        # Filter successful results for analysis
        self.successful_results = results_df[
            results_df['status'] == 'success'
        ] if 'status' in results_df.columns else results_df
        
        self.logger.info(f"Analyzing {len(self.successful_results)} successful classifications")
    
    def calculate_basic_statistics(self) -> Dict[str, Any]:
        """
        Calculate basic statistics about the classification results.
        
        Returns:
            Dictionary with basic statistics
        """
        stats = {
            'total_venues': len(self.results_df),
            'successful_classifications': len(self.successful_results),
            'failed_classifications': len(self.results_df) - len(self.successful_results),
            'success_rate': len(self.successful_results) / len(self.results_df) * 100,
        }
        
        if len(self.successful_results) > 0:
            similarity_scores = self.successful_results['similarity_score']
            stats.update({
                'avg_similarity_score': float(similarity_scores.mean()),
                'median_similarity_score': float(similarity_scores.median()),
                'min_similarity_score': float(similarity_scores.min()),
                'max_similarity_score': float(similarity_scores.max()),
                'std_similarity_score': float(similarity_scores.std()),
            })
        
        return stats
    
    def analyze_category_distribution(self) -> Dict[str, Any]:
        """
        Analyze the distribution of categories in the results.
        
        Returns:
            Dictionary with category distribution analysis
        """
        if len(self.successful_results) == 0:
            return {'error': 'No successful results to analyze'}
        
        # Count categories (filter out NaN values)
        category_labels = self.successful_results['category_label'].fillna('None').tolist()
        category_names = self.successful_results['category_name'].fillna('None').tolist()
        
        category_counts = Counter(category_labels)
        category_name_counts = Counter(category_names)
        
        # Calculate hierarchy levels
        hierarchy_levels = {}
        for category_label in category_labels:
            if category_label and category_label != 'None' and isinstance(category_label, str):
                level = category_label.count('>') + 1
                hierarchy_levels[level] = hierarchy_levels.get(level, 0) + 1
        
        return {
            'unique_categories': len(category_counts),
            'unique_category_names': len(category_name_counts),
            'most_common_categories': category_counts.most_common(10),
            'most_common_category_names': category_name_counts.most_common(10),
            'hierarchy_level_distribution': hierarchy_levels,
            'category_diversity_index': len(category_counts) / len(self.successful_results)
        }
    
    def analyze_similarity_distribution(self) -> Dict[str, Any]:
        """
        Analyze the distribution of similarity scores.
        
        Returns:
            Dictionary with similarity distribution analysis
        """
        if len(self.successful_results) == 0:
            return {'error': 'No successful results to analyze'}
        
        similarity_scores = self.successful_results['similarity_score']
        
        # Create similarity score bins
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
        
        binned_scores = pd.cut(similarity_scores, bins=bins, labels=bin_labels, include_lowest=True)
        bin_counts = binned_scores.value_counts().to_dict()
        
        # Find high and low confidence classifications
        high_confidence = self.successful_results[similarity_scores >= 0.7]
        low_confidence = self.successful_results[similarity_scores < 0.3]
        
        return {
            'similarity_distribution': bin_counts,
            'high_confidence_count': len(high_confidence),
            'low_confidence_count': len(low_confidence),
            'high_confidence_percentage': len(high_confidence) / len(self.successful_results) * 100,
            'low_confidence_percentage': len(low_confidence) / len(self.successful_results) * 100,
        }
    
    def find_potential_issues(self) -> Dict[str, Any]:
        """
        Identify potential issues in the classification results.
        
        Returns:
            Dictionary with potential issues
        """
        issues = {
            'duplicate_venues': [],
            'very_low_similarity': [],
            'potential_misclassifications': [],
            'empty_search_results': []
        }
        
        if len(self.successful_results) == 0:
            return issues
        
        # Find duplicate venues (same name and address)
        duplicates = self.successful_results.groupby(['name', 'address']).size()
        duplicate_venues = duplicates[duplicates > 1].index.tolist()
        issues['duplicate_venues'] = [f"{name} - {address}" for name, address in duplicate_venues]
        
        # Find very low similarity scores
        very_low_similarity = self.successful_results[
            self.successful_results['similarity_score'] < 0.2
        ]
        issues['very_low_similarity'] = very_low_similarity[['name', 'category_name', 'similarity_score']].to_dict('records')
        
        # Find potential search issues
        if 'enriched_text' in self.successful_results.columns:
            empty_search = self.successful_results[
                self.successful_results['enriched_text'].str.contains('No search results found|Search failed', na=False)
            ]
            issues['empty_search_results'] = empty_search[['name', 'address']].to_dict('records')
        
        return issues
    
    def generate_top_matches_by_category(self, top_n: int = 5) -> Dict[str, List[Dict]]:
        """
        Generate top matches for each category.
        
        Args:
            top_n: Number of top matches per category
            
        Returns:
            Dictionary with top matches by category
        """
        if len(self.successful_results) == 0:
            return {}
        
        top_matches = {}
        
        for category in self.successful_results['category_label'].unique():
            category_results = self.successful_results[
                self.successful_results['category_label'] == category
            ].nlargest(top_n, 'similarity_score')
            
            top_matches[category] = category_results[
                ['name', 'address', 'similarity_score']
            ].to_dict('records')
        
        return top_matches


def print_analysis_report(analyzer: ResultsAnalyzer) -> None:
    """
    Print a comprehensive analysis report.
    
    Args:
        analyzer: ResultsAnalyzer instance
    """
    logger = get_logger(__name__)
    
    logger.info("\n" + "="*80)
    logger.info("VENUE CLASSIFICATION ANALYSIS REPORT")
    logger.info("="*80)
    
    # Basic statistics
    stats = analyzer.calculate_basic_statistics()
    logger.info("\n📊 BASIC STATISTICS")
    logger.info("-" * 40)
    logger.info(f"Total venues: {stats['total_venues']}")
    logger.info(f"Successful classifications: {stats['successful_classifications']}")
    logger.info(f"Failed classifications: {stats['failed_classifications']}")
    logger.info(f"Success rate: {stats['success_rate']:.2f}%")
    
    if 'avg_similarity_score' in stats:
        logger.info(f"Average similarity score: {stats['avg_similarity_score']:.4f}")
        logger.info(f"Median similarity score: {stats['median_similarity_score']:.4f}")
        logger.info(f"Similarity score range: {stats['min_similarity_score']:.4f} - {stats['max_similarity_score']:.4f}")
        logger.info(f"Similarity score std dev: {stats['std_similarity_score']:.4f}")
    
    # Category distribution
    category_analysis = analyzer.analyze_category_distribution()
    if 'error' not in category_analysis:
        logger.info("\n🏷️  CATEGORY DISTRIBUTION")
        logger.info("-" * 40)
        logger.info(f"Unique categories: {category_analysis['unique_categories']}")
        logger.info(f"Unique category names: {category_analysis['unique_category_names']}")
        logger.info(f"Category diversity index: {category_analysis['category_diversity_index']:.4f}")
        
        logger.info("\nMost common categories:")
        for category, count in category_analysis['most_common_categories'][:5]:
            logger.info(f"  {category}: {count} venues")
        
        logger.info("\nHierarchy level distribution:")
        for level, count in sorted(category_analysis['hierarchy_level_distribution'].items()):
            logger.info(f"  Level {level}: {count} venues")
    
    # Similarity distribution
    similarity_analysis = analyzer.analyze_similarity_distribution()
    if 'error' not in similarity_analysis:
        logger.info("\n📈 SIMILARITY SCORE DISTRIBUTION")
        logger.info("-" * 40)
        logger.info(f"High confidence (≥0.7): {similarity_analysis['high_confidence_count']} venues ({similarity_analysis['high_confidence_percentage']:.1f}%)")
        logger.info(f"Low confidence (<0.3): {similarity_analysis['low_confidence_count']} venues ({similarity_analysis['low_confidence_percentage']:.1f}%)")
        
        logger.info("\nSimilarity score distribution:")
        for bin_range, count in similarity_analysis['similarity_distribution'].items():
            logger.info(f"  {bin_range}: {count} venues")
    
    # Potential issues
    issues = analyzer.find_potential_issues()
    logger.info("\n⚠️  POTENTIAL ISSUES")
    logger.info("-" * 40)
    logger.info(f"Duplicate venues: {len(issues['duplicate_venues'])}")
    logger.info(f"Very low similarity scores: {len(issues['very_low_similarity'])}")
    logger.info(f"Empty search results: {len(issues['empty_search_results'])}")
    
    if issues['very_low_similarity']:
        logger.info("\nVenues with very low similarity scores:")
        for venue in issues['very_low_similarity'][:5]:
            logger.info(f"  {venue['name']} → {venue['category_name']} ({venue['similarity_score']:.4f})")


def main() -> int:
    """
    Main function for results analysis.
    
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    setup_logging()
    logger = get_logger(__name__)
    
    try:
        logger.info("Starting venue classification results analysis...")
        
        # Load results data
        if not Path(VENUE_MATCHES_FILE).exists():
            logger.error(f"Results file not found: {VENUE_MATCHES_FILE}")
            logger.info("Please run venue classification first using: python cli/classify_venues.py")
            return 1
        
        results_df = DataLoader.load_csv_data(VENUE_MATCHES_FILE)
        DataLoader.display_dataframe_info(results_df, "Classification Results")
        
        # Initialize analyzer
        analyzer = ResultsAnalyzer(results_df)
        
        # Generate and print analysis report
        print_analysis_report(analyzer)
        
        # Save detailed analysis to JSON
        analysis_data = {
            'basic_statistics': analyzer.calculate_basic_statistics(),
            'category_distribution': analyzer.analyze_category_distribution(),
            'similarity_distribution': analyzer.analyze_similarity_distribution(),
            'potential_issues': analyzer.find_potential_issues(),
            'top_matches_by_category': analyzer.generate_top_matches_by_category(3)
        }
        
        analysis_output_path = f"{RESULTS_DIR}/classification_analysis.json"
        FileManager.save_json_data(analysis_data, analysis_output_path)
        
        logger.info("Results analysis completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error in results analysis: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())