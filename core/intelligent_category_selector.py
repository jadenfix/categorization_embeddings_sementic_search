"""
Intelligent category selection using OpenAI API for semantic understanding.
"""

import json
import logging
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from openai import OpenAI

from config import OPENAI_API_KEY, CHAT_MODEL

logger = logging.getLogger(__name__)


@dataclass
class CategoryCandidate:
    """Represents a category candidate with its similarity score."""
    label: str
    score: float


@dataclass
class VenueInfo:
    """Represents venue information for categorization."""
    name: str
    address: str
    enriched_text: str


class IntelligentCategorySelector:
    """
    Intelligently selects the most appropriate category using OpenAI API for semantic understanding.
    """
    
    def __init__(self) -> None:
        """Initialize the intelligent category selector with OpenAI client."""
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = CHAT_MODEL
    
    def _create_prompt(self, venue_info: VenueInfo, candidates: List[CategoryCandidate]) -> str:
        """
        Create the prompt for OpenAI API to select the best category.
        
        Args:
            venue_info: Venue information
            candidates: List of category candidates with scores
            
        Returns:
            Formatted prompt string
        """
        candidates_json = [
            {"label": candidate.label, "score": candidate.score}
            for candidate in candidates
        ]
        
        input_data = {
            "name": venue_info.name,
            "address": venue_info.address,
            "enriched_text": venue_info.enriched_text,
            "candidates": candidates_json
        }
        
        prompt = f"""You are categorizing a venue/business. Given the venue information and top similarity-matched categories, select the BEST match.

VENUE ANALYSIS GUIDELINES:
1. **Prioritize Business Type Indicators:**
   - Hotels/Lodging: "Hotel", "Guest House", "Penginapan", "Kost", "Boarding"
   - Medical: "Hospital", "Clinic", "Medical", "Rumah Sakit", "Puskesmas", "Bersalin", "Apotek", "Pharmacy"
   - Food: "Restaurant", "Cafe", "Coffee", "Warung", "Steak", "Kitchen"
   - Education: "School", "Elementary", "Middle", "University", "Sekolah"
   - Parks: "Park", "Taman", "Recreation"
   - Gas Stations: "Pertamina", "SPBU", "Gas Station", "RTC"

2. **Category Selection Rules:**
   - ALWAYS select a category unless ALL candidates are completely wrong
   - Look for business type keywords in name, address, or description
   - Prefer more specific categories over general ones
   - Use context clues from the enriched text
   - Consider regional/language variations (Indonesian terms)

3. **Only respond "None" if:**
   - ALL candidates are obviously wrong business types
   - The venue appears to be a private residence with no business indicators

VENUE DATA:
{json.dumps(input_data, indent=2)}

IMPORTANT: Select the BEST matching category from the candidates. If any candidate reasonably matches the business type, choose it. Only use "None" if ALL candidates are clearly wrong.

Response (ONLY the exact category label or "None"):"""
        
        return prompt
    
    def select_best_category(
        self, 
        venue_info: VenueInfo, 
        candidates: List[CategoryCandidate]
    ) -> Optional[str]:
        """
        Select the most appropriate category using OpenAI API.
        
        Args:
            venue_info: Information about the venue
            candidates: List of category candidates with similarity scores
            
        Returns:
            Best category label or None if no good match found
        """
        if not candidates:
            logger.warning("No category candidates provided")
            return None
        
        logger.info(f"Analyzing venue with OpenAI: {venue_info.name}")
        
        try:
            # Create prompt
            prompt = self._create_prompt(venue_info, candidates)
            logger.debug(f"Created prompt for venue: {venue_info.name}")
            
            # Call OpenAI API with retry logic
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at categorizing businesses and locations. You analyze venue information and select the most appropriate category from given candidates based on semantic understanding."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=100,   # Short response expected
                timeout=30
            )
            
            logger.debug(f"OpenAI API call successful for venue: {venue_info.name}")
            
            # Extract result
            result = response.choices[0].message.content.strip()
            
            # Validate result
            if result.lower() == "none":
                logger.info("OpenAI determined no suitable category match")
                return None
            
            # Verify the result is one of the candidate labels
            candidate_labels = [candidate.label for candidate in candidates]
            if result not in candidate_labels:
                logger.warning(f"OpenAI returned unexpected category: {result}")
                # Try to find a partial match
                for label in candidate_labels:
                    if result in label or label in result:
                        logger.info(f"Using partial match: {label}")
                        return label
                return None
            
            logger.info(f"OpenAI selected category: {result}")
            return result
            
        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"OpenAI API error ({error_type}): {e}")
            
            # Categorize error types for better handling
            if "rate limit" in str(e).lower():
                logger.warning("Rate limit exceeded - using fallback")
            elif "timeout" in str(e).lower():
                logger.warning("Request timeout - using fallback")
            elif "api key" in str(e).lower() or "authentication" in str(e).lower():
                logger.warning("Authentication error - using fallback")
            else:
                logger.warning(f"Unexpected error ({error_type}) - using fallback")
            
            # Fallback to highest similarity score if OpenAI fails or is too restrictive
            if candidates:
                best_candidate = max(candidates, key=lambda x: x.score)
                # Use a lower threshold for fallback - accept reasonable similarity scores
                if best_candidate.score >= 0.15:  # Lower threshold for fallback
                    logger.info(f"Using best similarity fallback: {best_candidate.label} (score: {best_candidate.score:.4f})")
                    return best_candidate.label
                else:
                    logger.info(f"Best candidate score too low: {best_candidate.score:.4f}")
            
            logger.warning("No candidates available for fallback")
            return None


def select_category_for_venue(
    name: str,
    address: str,
    enriched_text: str,
    category_candidates: List[Tuple[str, float]]
) -> Optional[str]:
    """
    Convenience function to select the best category for a venue using OpenAI.
    
    Args:
        name: Venue name
        address: Venue address
        enriched_text: Enriched text description from search results
        category_candidates: List of (category_label, similarity_score) tuples
        
    Returns:
        Best category label or None if no good match
        
    Example:
        >>> candidates = [
        ...     ("Dining and Drinking > Restaurant > German Restaurant > Schnitzel Restaurant", 0.3758),
        ...     ("Health and Medicine > Medical Center", 0.3185),
        ...     ("Community and Government > Organization > Charity", 0.2964),
        ...     ("Business and Professional Services > Office > Corporate Coffee Shop", 0.2881)
        ... ]
        >>> select_category_for_venue(
        ...     "Dr Med Silvia Schroter Arztin Fur Psychotherapie De",
        ...     "76, Bender, 40625 Düsseldorf",
        ...     "Dr Med Silvia Schroter ... Psychotherapie ... Praxis ... Ärztin ... Düsseldorf",
        ...     candidates
        ... )
        'Health and Medicine > Medical Center'
    """
    venue_info = VenueInfo(name=name, address=address, enriched_text=enriched_text)
    
    candidates = [
        CategoryCandidate(label=label, score=score)
        for label, score in category_candidates
    ]
    
    selector = IntelligentCategorySelector()
    return selector.select_best_category(venue_info, candidates)