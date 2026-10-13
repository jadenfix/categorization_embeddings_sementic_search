#!/usr/bin/env python3
"""
CLI script for analyzing venue classification results.
"""

import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
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
        
        # Execution and classification are different outcomes. A provider call can
        # complete successfully while the selector correctly abstains.
        if 'status' in results_df.columns:
            execution_succeeded = (
                results_df['status'].fillna('').astype(str).str.strip().str.casefold()
                == 'success'
            )
        else:
            # Preserve compatibility with older result exports that predate status.
            execution_succeeded = pd.Series(True, index=results_df.index)

        category_labels = results_df['category_label'] if 'category_label' in results_df.columns else pd.Series(None, index=results_df.index)
        abstention_tokens = {'', 'none', 'null', 'unknown'}
        normalized_labels = category_labels.fillna('').astype(str).str.strip().str.casefold()
        has_category = ~normalized_labels.isin(abstention_tokens)

        self.execution_succeeded_results = results_df[execution_succeeded].copy()
        self.classified_results = results_df[execution_succeeded & has_category].copy()
        self.abstained_results = results_df[execution_succeeded & ~has_category].copy()
        self.failed_results = results_df[~execution_succeeded].copy()

        self.logger.info(
            "Analyzing %d rows: %d classified, %d abstained, %d failed",
            len(results_df),
            len(self.classified_results),
            len(self.abstained_results),
            len(self.failed_results),
        )
    
    def calculate_basic_statistics(self) -> Dict[str, Any]:
        """
        Calculate basic statistics about the classification results.
        
        Returns:
            Dictionary with basic statistics
        """
        total = len(self.results_df)
        executed = len(self.execution_succeeded_results)
        classified = len(self.classified_results)
        stats = {
            'total_venues': total,
            'execution_succeeded': executed,
            'classified_venues': classified,
            'abstained_venues': len(self.abstained_results),
            'failed_venues': len(self.failed_results),
            'execution_success_rate': executed / total * 100 if total else 0.0,
            'classification_coverage': classified / total * 100 if total else 0.0,
        }
        
        if classified > 0:
            similarity_scores = pd.to_numeric(
                self.classified_results['similarity_score'], errors='coerce'
            ).dropna()
        else:
            similarity_scores = pd.Series(dtype=float)

        if len(similarity_scores) > 0:
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
        if len(self.classified_results) == 0:
            return {'error': 'No classified results to analyze'}
        
        # Count categories (filter out NaN values)
        category_labels = self.classified_results['category_label'].tolist()
        category_names = self.classified_results['category_name'].fillna('Unnamed').tolist()
        
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
            'category_diversity_index': len(category_counts) / len(self.classified_results)
        }
    
    def analyze_similarity_distribution(self) -> Dict[str, Any]:
        """
        Analyze the distribution of similarity scores.
        
        Returns:
            Dictionary with similarity distribution analysis
        """
        if len(self.classified_results) == 0:
            return {'error': 'No classified results to analyze'}
        
        similarity_scores = pd.to_numeric(
            self.classified_results['similarity_score'], errors='coerce'
        ).dropna()
        if len(similarity_scores) == 0:
            return {'error': 'No numeric similarity scores for classified results'}
        
        # Create similarity score bins
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        bin_labels = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
        
        binned_scores = pd.cut(similarity_scores, bins=bins, labels=bin_labels, include_lowest=True)
        bin_counts = binned_scores.value_counts().to_dict()
        
        # Find high and low confidence classifications
        high_confidence_count = int((similarity_scores >= 0.7).sum())
        low_confidence_count = int((similarity_scores < 0.3).sum())
        
        return {
            'similarity_distribution': bin_counts,
            'high_confidence_count': high_confidence_count,
            'low_confidence_count': low_confidence_count,
            'high_confidence_percentage': high_confidence_count / len(similarity_scores) * 100,
            'low_confidence_percentage': low_confidence_count / len(similarity_scores) * 100,
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
        
        if len(self.execution_succeeded_results) == 0:
            return issues
        
        # Find duplicate venues (same name and address)
        duplicates = self.execution_succeeded_results.groupby(['name', 'address']).size()
        duplicate_venues = duplicates[duplicates > 1].index.tolist()
        issues['duplicate_venues'] = [f"{name} - {address}" for name, address in duplicate_venues]
        
        # Find very low similarity scores
        very_low_similarity = self.classified_results[
            self.classified_results['similarity_score'] < 0.2
        ]
        issues['very_low_similarity'] = very_low_similarity[['name', 'category_name', 'similarity_score']].to_dict('records')
        
        # Find potential search issues
        if 'enriched_text' in self.execution_succeeded_results.columns:
            empty_search = self.execution_succeeded_results[
                self.execution_succeeded_results['enriched_text'].str.contains('No search results found|Search failed', na=False)
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
        if len(self.classified_results) == 0:
            return {}
        
        top_matches = {}
        
        for category in self.classified_results['category_label'].unique():
            category_results = self.classified_results[
                self.classified_results['category_label'] == category
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
    logger.info(f"Execution succeeded: {stats['execution_succeeded']}")
    logger.info(f"Classified: {stats['classified_venues']}")
    logger.info(f"Abstained: {stats['abstained_venues']}")
    logger.info(f"Failed: {stats['failed_venues']}")
    logger.info(f"Execution success rate: {stats['execution_success_rate']:.2f}%")
    logger.info(f"Observed classification coverage: {stats['classification_coverage']:.2f}%")
    
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
