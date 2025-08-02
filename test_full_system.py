#!/usr/bin/env python3
"""
Comprehensive test for the full venue classification system with LLM fallback.
Tests both semantic search and LLM fallback functionality.
"""

import logging
import pandas as pd
from typing import Dict, List, Any

from config import CATEGORY_EMBEDDINGS_BASE, NUMPY_EXTENSION, METADATA_EXTENSION
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
        venue_data: Dict with 'name', 'address', 'enriched_text' keys
        components: Dict with initialized classification components
        
    Returns:
        Classification result dict
    """
    embedding_generator = components['embedding_generator']
    category_matcher = components['category_matcher']
    intelligent_selector = components['intelligent_selector']
    
    try:
        # Generate embedding for the venue
        venue_embedding = embedding_generator.generate_embedding(venue_data['enriched_text'])
        
        # Get top category candidates
        top_matches = category_matcher.find_top_k_matches(venue_embedding, k=4)
        
        # Use intelligent category selector
        venue_info = VenueInfo(
            name=venue_data['name'],
            address=venue_data['address'],
            enriched_text=venue_data['enriched_text']
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
            'enriched_text': venue_data['enriched_text'],
            'category_name': category_name,
            'category_label': selected_category or 'None',
            'similarity_score': similarity_score,
            'status': 'success' if selected_category else 'success',
            'top_matches': top_matches
        }
        
    except Exception as e:
        logger.error(f"Classification failed for {venue_data['name']}: {e}")
        return {
            'name': venue_data['name'],
            'address': venue_data['address'],
            'enriched_text': venue_data['enriched_text'],
            'category_name': 'None',
            'category_label': 'None',
            'similarity_score': 0.0,
            'status': 'error',
            'error': str(e)
        }

def test_venues_from_csv() -> Dict[str, Any]:
    """Test venues from the existing CSV file."""
    logger.info("Testing venues from existing CSV file...")
    
    # Load test data from the results CSV that has enriched text
    csv_file = 'results/venue_category_matches_with_llm_fallback.csv'
    try:
        df = pd.read_csv(csv_file)
        logger.info(f"Loaded {len(df)} venues from CSV")
        # Take first 10 venues for testing to avoid OpenAI rate limits
        df = df.head(10)
        logger.info(f"Testing first {len(df)} venues to avoid rate limits")
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return {"error": str(e)}
    
    # Initialize components
    try:
        embedding_generator = EmbeddingGenerator()
        
        # Load category embeddings
        embeddings_path = f"{CATEGORY_EMBEDDINGS_BASE}{NUMPY_EXTENSION}"
        metadata_path = f"{CATEGORY_EMBEDDINGS_BASE}{METADATA_EXTENSION}"
        
        embeddings, metadata = DataLoader.load_embeddings_data(embeddings_path, metadata_path)
        category_matcher = CategoryMatcher(embeddings, metadata)
        intelligent_selector = IntelligentCategorySelector()
        
        components = {
            'embedding_generator': embedding_generator,
            'category_matcher': category_matcher,
            'intelligent_selector': intelligent_selector
        }
        
        logger.info("Successfully initialized all components")
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        return {"error": f"Component initialization failed: {e}"}
    
    # Load categories
    categories_df = load_foursquare_categories()
    all_categories = categories_df['Category Label'].tolist()
    logger.info(f"Loaded {len(all_categories)} categories")
    
    # Test venues
    test_results = []
    none_count = 0
    success_count = 0
    
    for idx, row in df.iterrows():
        venue_data = {
            'name': row['name'],
            'address': row['address'],
            'enriched_text': row['enriched_text']
        }
        
        # Get classification
        result = classify_venue(venue_data, components)
        
        if result['category_label'] == 'None':
            none_count += 1
        else:
            success_count += 1
            
        test_results.append(result)
        
        logger.info(f"Venue {idx+1}/{len(df)}: {result['name']} -> {result['category_label']} "
                   f"(similarity: {result['similarity_score']:.4f})")
    
    initial_success_rate = (success_count / len(test_results)) * 100
    logger.info(f"\nInitial classification results:")
    logger.info(f"  - Total venues: {len(test_results)}")
    logger.info(f"  - Successfully classified: {success_count}")
    logger.info(f"  - None classifications: {none_count}")
    logger.info(f"  - Initial success rate: {initial_success_rate:.1f}%")
    
    # Apply LLM fallback
    logger.info("\nApplying LLM fallback to 'None' classifications...")
    updated_results = apply_llm_fallback_to_results(test_results, all_categories)
    
    # Calculate final statistics
    final_none_count = sum(1 for r in updated_results if r['category_label'] == 'None')
    final_success_count = len(updated_results) - final_none_count
    final_success_rate = (final_success_count / len(updated_results)) * 100
    
    fallback_reclassified = sum(1 for r in updated_results if r.get('fallback_status') == 'llm_reclassified')
    fallback_no_match = sum(1 for r in updated_results if r.get('fallback_status') == 'llm_no_match')
    
    logger.info(f"\nFinal classification results:")
    logger.info(f"  - Total venues: {len(updated_results)}")
    logger.info(f"  - Successfully classified: {final_success_count}")
    logger.info(f"  - None classifications: {final_none_count}")
    logger.info(f"  - Final success rate: {final_success_rate:.1f}%")
    logger.info(f"  - LLM reclassified: {fallback_reclassified}")
    logger.info(f"  - LLM no match: {fallback_no_match}")
    logger.info(f"  - Improvement: +{final_success_rate - initial_success_rate:.1f}%")
    
    return {
        "total_venues": len(updated_results),
        "initial_success": success_count,
        "initial_success_rate": initial_success_rate,
        "final_success": final_success_count,
        "final_success_rate": final_success_rate,
        "fallback_reclassified": fallback_reclassified,
        "fallback_no_match": fallback_no_match,
        "improvement": final_success_rate - initial_success_rate,
        "detailed_results": updated_results
    }

def test_specific_challenging_venues() -> Dict[str, Any]:
    """Test specific venues that are challenging to classify."""
    logger.info("\nTesting specific challenging venues...")
    
    challenging_venues = [
        {
            'name': 'Dr Med Silvia Schroter Arztin Fur Psychotherapie De',
            'address': '76, Bender, 40625 Düsseldorf',
            'enriched_text': 'Dr Med Silvia Schroter Arztin Fur Psychotherapie De, 76, Bender, 40625 Düsseldorf - Adresse. Benderstraße 76 40625 Düsseldorf ; Telefon: 0211 / 9 23 30 63 ; Stadtteil: Gerresheim ; e-Mail-Adresse: schroeter@psychotherapiepraxis-duesseldorf.de.',
            'expected_category_type': 'Health and Medicine > Mental Health Service'
        },
        {
            'name': 'Hotel Puri Mira Bogor',
            'address': 'Jl. Raya Tajur No.225, RT.03/RW.03, Muarasari, Bogor Selatan, Kota Bogor, Jawa Barat, Indonesia, 16137',
            'enriched_text': 'Hotel Puri Mira Bogor. Jl. Raya Tajur No.225, RT.03/RW.03, Muarasari, Kec. Bogor Sel, East Bogor, Bogor, West Java, Indonesia, 16145. Hotel accommodation with pool.',
            'expected_category_type': 'Travel and Transportation > Lodging > Hotel'
        },
        {
            'name': 'Rumah Bersalin Juliana',
            'address': 'JL. Raya Durian, No. 70, Baranangsiang, Bogor Timur, Perumnas Bantar Kemang, Bogor, Jawa Barat, Indonesia 16143',
            'enriched_text': 'Rumah Bersalin Juliana, JL. Raya Durian, No. 70, Baranangsiang, Bogor Timur, Perumnas Bantar Kemang Bogor, Jawa Barat, Indonesia 16143. Phone: (0251) 8374637. Maternity clinic.',
            'expected_category_type': 'Health and Medicine > Medical Center'
        },
        {
            'name': 'Apotek Fokus Farma',
            'address': 'Jl. Permata No.188C, Curug, Gunung Sindur, Bogor, Jawa Barat, Indonesia, 16340',
            'enriched_text': 'Apotek Fokus Farma, Jl. Permata No.188C, Curug, Gunung Sindur, Bogor, Jawa Barat, Indonesia, 16340 - Pharmacy services available.',
            'expected_category_type': 'Retail > Pharmacy'
        },
        {
            'name': 'คอฟฟี่ ทูเดย์ Th Coffee Today En TH',
            'address': 'ลาดพร้าว, 10310 วังทองหลาง',
            'enriched_text': 'คอฟฟี่ ทูเดย์ Th Coffee Today En TH, ลาดพร้าว, 10310 วังทองหลาง - Coffee shop with variety of drinks and light meals.',
            'expected_category_type': 'Dining and Drinking > Cafe, Coffee, and Tea House'
        },
        {
            'name': 'RTC Terintegrasi Pertamina',
            'address': 'Jl. Daan Mogot KM 16, Kalideres, Kota Jakarta Barat, DKI Jakarta, Indonesia, 11840',
            'enriched_text': 'RTC Terintegrasi Pertamina, Jl. Daan Mogot KM 16, Kalideres, Kota Jakarta Barat, DKI Jakarta, Indonesia, 11840 - Research & Technology Center (RTC) Pertamina. Groundbreaking pembangunan fasilitas research center.',
            'expected_category_type': 'Business and Professional Services'
        }
    ]
    
    # Initialize components
    try:
        embedding_generator = EmbeddingGenerator()
        
        # Load category embeddings
        embeddings_path = f"{CATEGORY_EMBEDDINGS_BASE}{NUMPY_EXTENSION}"
        metadata_path = f"{CATEGORY_EMBEDDINGS_BASE}{METADATA_EXTENSION}"
        
        embeddings, metadata = DataLoader.load_embeddings_data(embeddings_path, metadata_path)
        category_matcher = CategoryMatcher(embeddings, metadata)
        intelligent_selector = IntelligentCategorySelector()
        
        components = {
            'embedding_generator': embedding_generator,
            'category_matcher': category_matcher,
            'intelligent_selector': intelligent_selector
        }
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        return {"error": f"Component initialization failed: {e}"}
    
    # Load categories
    categories_df = load_foursquare_categories()
    all_categories = categories_df['Category Label'].tolist()
    
    test_results = []
    
    for venue in challenging_venues:
        # Get initial classification
        result = classify_venue(venue, components)
        
        # Apply LLM fallback if needed
        if result['category_label'] == 'None':
            from core.llm_fallback import LLMFallback, VenueForFallback
            
            fallback = LLMFallback()
            venue_obj = VenueForFallback(
                name=venue['name'],
                address=venue['address'],
                enriched_text=venue['enriched_text'],
                similarity_score=result['similarity_score']
            )
            
            fallback_category = fallback.classify_venue_with_fallback(venue_obj, all_categories)
            if fallback_category:
                result['category_label'] = fallback_category
                result['category_name'] = fallback_category
                result['fallback_status'] = 'llm_reclassified'
            else:
                result['fallback_status'] = 'llm_no_match'
        else:
            result['fallback_status'] = 'not_applied'
        
        # Check if category matches expected type
        expected_type = venue['expected_category_type']
        is_correct = result['category_label'].startswith(expected_type) if result['category_label'] != 'None' else False
        
        test_result = {
            'venue_name': venue['name'],
            'expected_type': expected_type,
            'classified_as': result['category_label'],
            'similarity_score': result['similarity_score'],
            'fallback_status': result.get('fallback_status', 'not_applied'),
            'is_correct': is_correct
        }
        
        test_results.append(test_result)
        
        logger.info(f"Venue: {venue['name']}")
        logger.info(f"  Expected: {expected_type}")
        logger.info(f"  Classified as: {result['category_label']}")
        logger.info(f"  Similarity: {result['similarity_score']:.4f}")
        logger.info(f"  Fallback: {result.get('fallback_status', 'not_applied')}")
        logger.info(f"  Correct: {'✓' if is_correct else '✗'}")
        logger.info("")
    
    correct_count = sum(1 for r in test_results if r['is_correct'])
    accuracy = (correct_count / len(test_results)) * 100
    
    logger.info(f"Challenging venues test results:")
    logger.info(f"  - Total tested: {len(test_results)}")
    logger.info(f"  - Correctly classified: {correct_count}")
    logger.info(f"  - Accuracy: {accuracy:.1f}%")
    
    return {
        "total_tested": len(test_results),
        "correct_count": correct_count,
        "accuracy": accuracy,
        "detailed_results": test_results
    }

def main():
    """Run comprehensive system test."""
    logger.info("Starting comprehensive venue classification system test...")
    
    # Test 1: Full CSV dataset
    csv_results = test_venues_from_csv()
    
    # Test 2: Challenging specific venues
    challenging_results = test_specific_challenging_venues()
    
    # Overall summary
    logger.info("\n" + "="*60)
    logger.info("COMPREHENSIVE TEST SUMMARY")
    logger.info("="*60)
    
    if "error" not in csv_results:
        logger.info(f"CSV Dataset Results:")
        logger.info(f"  - Total venues: {csv_results['total_venues']}")
        logger.info(f"  - Initial success rate: {csv_results['initial_success_rate']:.1f}%")
        logger.info(f"  - Final success rate: {csv_results['final_success_rate']:.1f}%")
        logger.info(f"  - Improvement: +{csv_results['improvement']:.1f}%")
        logger.info(f"  - LLM fallback reclassified: {csv_results['fallback_reclassified']}")
    
    logger.info(f"\nChallenging Venues Results:")
    logger.info(f"  - Total tested: {challenging_results['total_tested']}")
    logger.info(f"  - Accuracy: {challenging_results['accuracy']:.1f}%")
    
    # Overall system assessment
    if "error" not in csv_results:
        if csv_results['final_success_rate'] >= 90:
            logger.info(f"\n✅ SYSTEM STATUS: EXCELLENT ({csv_results['final_success_rate']:.1f}% success rate)")
        elif csv_results['final_success_rate'] >= 75:
            logger.info(f"\n✅ SYSTEM STATUS: GOOD ({csv_results['final_success_rate']:.1f}% success rate)")
        else:
            logger.info(f"\n⚠️  SYSTEM STATUS: NEEDS IMPROVEMENT ({csv_results['final_success_rate']:.1f}% success rate)")
    
    logger.info("Test completed successfully!")
    
    return {
        "csv_results": csv_results,
        "challenging_results": challenging_results
    }

if __name__ == "__main__":
    main()
