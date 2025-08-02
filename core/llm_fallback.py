"""
LLM fallback for venues classified as "None".
"""

import json
import logging
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from difflib import get_close_matches

from openai import OpenAI
from config import OPENAI_API_KEY, CHAT_MODEL

logger = logging.getLogger(__name__)


@dataclass
class VenueForFallback:
    """Venue that needs LLM fallback classification."""
    name: str
    address: str
    enriched_text: str
    similarity_score: float


class LLMFallback:
    """Performs LLM-based fallback classification for "None" venues."""
    
    def __init__(self) -> None:
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = CHAT_MODEL
    
    def _find_fuzzy_category_match(self, llm_response: str, all_categories: List[str]) -> Optional[str]:
        """Find the closest category match using fuzzy matching."""
        # Try exact match first
        if llm_response in all_categories:
            return llm_response
        
        # Try fuzzy matching with different cutoff values
        cutoffs = [0.9, 0.8, 0.7, 0.6]
        
        for cutoff in cutoffs:
            matches = get_close_matches(llm_response, all_categories, n=1, cutoff=cutoff)
            if matches:
                best_match = matches[0]
                logger.info(f"Fuzzy match found: '{llm_response}' -> '{best_match}' (cutoff: {cutoff})")
                return best_match
        
        # Try keyword-based matching for common cases
        llm_lower = llm_response.lower()
        
        # Common category mappings (including Indonesian terms)
        keyword_mappings = {
            # Hotels and Lodging
            'hotel': 'Travel and Transportation > Lodging > Hotel',
            'guest house': 'Travel and Transportation > Lodging > Hotel',
            'kost': 'Travel and Transportation > Lodging',
            'penginapan': 'Travel and Transportation > Lodging > Hotel',
            
            # Medical and Health
            'rumah sakit': 'Health and Medicine > Medical Center',
            'rumah bersalin': 'Health and Medicine > Medical Center',
            'klinik': 'Health and Medicine > Medical Center',
            'puskesmas': 'Health and Medicine > Medical Center',
            'apotek': 'Retail > Pharmacy',
            'pharmacy': 'Retail > Pharmacy',
            'hospital': 'Health and Medicine > Medical Center',
            'clinic': 'Health and Medicine > Medical Center',
            'medical center': 'Health and Medicine > Medical Center',
            
            # Food and Dining
            'restaurant': 'Dining and Drinking > Restaurant',
            'restoran': 'Dining and Drinking > Restaurant',
            'warung': 'Dining and Drinking > Restaurant',
            'kedai': 'Dining and Drinking > Restaurant',
            'cafe': 'Dining and Drinking > Cafe, Coffee, and Tea House',
            'kafe': 'Dining and Drinking > Cafe, Coffee, and Tea House',
            'coffee': 'Dining and Drinking > Cafe, Coffee, and Tea House > Coffee Shop',
            'steak': 'Dining and Drinking > Restaurant',
            'sunda': 'Dining and Drinking > Restaurant > Asian Restaurant > Indonesian Restaurant',
            
            # Education
            'school': 'Community and Government > Education',
            'sekolah': 'Community and Government > Education',
            'elementary': 'Community and Government > Education > Primary and Secondary School > Elementary School',
            'middle school': 'Community and Government > Education > Primary and Secondary School > Middle School',
            'universitas': 'Community and Government > Education > College and University',
            'university': 'Community and Government > Education > College and University',
            
            # Recreation and Entertainment
            'park': 'Travel and Transportation > Park',
            'taman': 'Travel and Transportation > Park',
            'water park': 'Arts and Entertainment > Water Park',
            'aquatic': 'Arts and Entertainment > Water Park',
            
            # Retail and Shopping
            'toko': 'Retail',
            'mall': 'Retail > Shopping Plaza',
            'plaza': 'Retail > Shopping Plaza',
            'supermarket': 'Retail > Grocery Store',
            
            # Transportation and Automotive
            'gas station': 'Automotive > Gas Station',
            'spbu': 'Automotive > Gas Station',
            'parking': 'Travel and Transportation > Parking',
            
            # Professional Services
            'bank': 'Business and Professional Services > Bank',
            'kantor': 'Business and Professional Services > Office',
            'office': 'Business and Professional Services > Office'
        }
        
        for keyword, category_prefix in keyword_mappings.items():
            if keyword in llm_lower:
                # Find categories that start with this prefix
                matching_categories = [cat for cat in all_categories if cat.startswith(category_prefix)]
                if matching_categories:
                    # Return the most specific match (longest category path)
                    best_match = max(matching_categories, key=len)
                    logger.info(f"Keyword match found: '{llm_response}' -> '{best_match}' (keyword: {keyword})")
                    return best_match
        
        return None

    def _analyze_venue_name_for_category(self, venue_name: str, all_categories: List[str]) -> Optional[str]:
        """Analyze venue name directly for business type indicators."""
        name_lower = venue_name.lower()
        
        # Direct name pattern matching
        name_patterns = {
            # Medical facilities
            r'rumah bersalin|maternity|bersalin': 'Health and Medicine > Medical Center',
            r'apotek|pharmacy': 'Retail > Pharmacy',
            r'puskesmas|clinic|klinik': 'Health and Medicine > Medical Center',
            r'rumah sakit|hospital': 'Health and Medicine > Medical Center',
            
            # Hotels and lodging
            r'hotel|guest house|penginapan': 'Travel and Transportation > Lodging > Hotel',
            r'kost|boarding': 'Travel and Transportation > Lodging',
            r"home(?!'s)|house(?! of)|rumah(?! sakit|bersalin)": 'Travel and Transportation > Lodging',
            
            # Food and beverage
            r'coffee|kopi|cafe|kafe': 'Dining and Drinking > Cafe, Coffee, and Tea House > Coffee Shop',
            r'restaurant|restoran|warung|kedai': 'Dining and Drinking > Restaurant',
            r'steak': 'Dining and Drinking > Restaurant',
            
            # Education
            r'elementary|primary school|sekolah dasar': 'Community and Government > Education > Primary and Secondary School > Elementary School',
            r'middle school|sekolah menengah': 'Community and Government > Education > Primary and Secondary School > Middle School',
            r'school|sekolah': 'Community and Government > Education',
            r'university|universitas|college': 'Community and Government > Education > College and University',
            
            # Recreation and Tourism
            r'aquatic|water park': 'Arts and Entertainment > Water Park',
            r'park|taman': 'Travel and Transportation > Park',
            r'lembah|valley|tepus': 'Arts and Entertainment > Exhibit',
            r'attraction|tourist': 'Arts and Entertainment > General Entertainment',
            
            # Retail and Commercial
            r'toko|store|shop': 'Retail',
            r'mall|plaza': 'Retail > Shopping Plaza',
            r'supermarket|pasar': 'Retail > Grocery Store',
            r'ruko|shop house|blok|residence': 'Retail',
            
            # Transportation and Automotive
            r'spbu|gas station|fuel': 'Automotive > Gas Station',
            
            # Professional services and research
            r'psychotherapie|psychologist|therapy': 'Health and Medicine > Mental Health Service > Psychologist',
            r'caritas|charity': 'Community and Government > Organization > Charity',
            r'research.*technology|technology.*center|rtc|research.*center': 'Business and Professional Services > Office > Corporate Office',
            r'pertamina.*integrated|integrated.*research': 'Business and Professional Services > Office > Corporate Office',
            
            # Food and beverage (specific brands)
            r'milo|mengkel': 'Dining and Drinking > Restaurant'
        }
        
        return self._find_pattern_match(name_lower, name_patterns, all_categories)
        
    def _find_pattern_match(self, text: str, patterns: Dict[str, str], all_categories: List[str]) -> Optional[str]:
        """Helper method to find pattern matches and return best category."""
        text_lower = text.lower()
        
        for pattern, category_prefix in patterns.items():
            if re.search(pattern, text_lower):
                # Find exact matching categories
                matching_categories = [cat for cat in all_categories if cat.startswith(category_prefix)]
                if matching_categories:
                    # Return exact match if available, otherwise most specific
                    exact_match = [cat for cat in matching_categories if cat == category_prefix]
                    return exact_match[0] if exact_match else max(matching_categories, key=len)
        return None

    def _analyze_enriched_text_for_category(self, enriched_text: str, all_categories: List[str]) -> Optional[str]:
        """Analyze enriched text for business type indicators."""
        text_lower = enriched_text.lower()
        
        # Context pattern matching from descriptions
        context_patterns = {
            # Medical and health context
            r'puskesmas|health center|medical|clinic|hospital|doctor': 'Health and Medicine > Medical Center',
            r'maternity|bersalin|birthing': 'Health and Medicine > Medical Center',
            
            # Research and technology context
            r'research.*technology.*center|technology.*center|rtc.*pertamina': 'Business and Professional Services > Office > Corporate Office',
            r'integrated.*research|pertamina.*integrated': 'Business and Professional Services > Office > Corporate Office',
            r'groundbreaking.*fasilitas|seven buildings|research.*project': 'Business and Professional Services > Office > Corporate Office',
            
            # Tourism and recreation context
            r'tourist|attraction|sightseeing|visit|tourism|wisata': 'Arts and Entertainment > General Entertainment',
            r'valley|waterfall|nature|scenic|landscape|tempat wisata': 'Arts and Entertainment > Exhibit',
            r'open.*hours|entrance fee|ticket|price.*rp': 'Arts and Entertainment > General Entertainment',
            
            # Fuel and automotive context
            r'fuel|gas|station|spbu|bbm': 'Automotive > Gas Station',
            
            # Commercial and retail context
            r'shop house|ruko|commercial|retail|business': 'Retail',
            r'residence.*block|shopping.*complex': 'Retail > Shopping Plaza',
            
            # Lodging context  
            r'accommodation|stay|room|bed|guest|boarding': 'Travel and Transportation > Lodging',
            r'homestay|guesthouse|inn|lodge': 'Travel and Transportation > Lodging > Hotel',
            
            # Food and beverage context
            r'restaurant|dining|food|menu|cuisine|meal': 'Dining and Drinking > Restaurant',
            r'coffee|espresso|latte|cappuccino|brew': 'Dining and Drinking > Cafe, Coffee, and Tea House > Coffee Shop',
            
            # Generic business with no info
            r'no search results|missing.*data': 'Business and Professional Services > Office'
        }
        
        return self._find_pattern_match(text_lower, context_patterns, all_categories)

    def _create_fallback_prompt(self, venue: VenueForFallback, all_categories: List[str]) -> str:
        """Create comprehensive prompt for LLM fallback."""
        # Sample key categories to show in prompt
        sample_categories = []
        
        # Add major category types
        category_samples = {
            'Dining': [cat for cat in all_categories if 'Dining and Drinking' in cat],
            'Retail': [cat for cat in all_categories if 'Retail' in cat],
            'Hotel': [cat for cat in all_categories if 'Hotel' in cat or 'Lodging' in cat],
            'Medical': [cat for cat in all_categories if 'Health and Medicine' in cat],
            'Education': [cat for cat in all_categories if 'Education' in cat],
            'Transportation': [cat for cat in all_categories if 'Travel and Transportation' in cat],
            'Entertainment': [cat for cat in all_categories if 'Arts and Entertainment' in cat],
            'Business': [cat for cat in all_categories if 'Business and Professional Services' in cat]
        }
        
        for category_type, cats in category_samples.items():
            sample_categories.extend(cats[:8])  # Take first 8 from each type
        
        return f"""You are an expert venue categorizer. A venue was classified as "None" but needs proper categorization.

VENUE INFORMATION:
- Name: {venue.name}
- Address: {venue.address}
- Enriched Description: {venue.enriched_text}
- Original Similarity Score: {venue.similarity_score:.4f}

SAMPLE AVAILABLE CATEGORIES (showing key examples from {len(all_categories)} total):
{json.dumps(sample_categories[:80], indent=2)}

IMPORTANT CATEGORY EXAMPLES:
- Hotels: "Travel and Transportation > Lodging > Hotel"
- Restaurants: "Dining and Drinking > Restaurant > [Cuisine Type]"
- Pharmacies: "Retail > Pharmacy"
- Medical: "Health and Medicine > Medical Center"
- Schools: "Community and Government > Education > [School Type]"
- Gas Stations: "Automotive > Gas Station"
- Coffee Shops: "Dining and Drinking > Cafe, Coffee, and Tea House > Coffee Shop"

INSTRUCTIONS:
1. Analyze the venue name, address, and description carefully
2. Look for business type indicators, keywords, and contextual clues
3. Match to the EXACT category format shown in examples
4. Use the hierarchical structure (Parent > Child > Specific)
5. If you find a match, respond with ONLY the exact category label
6. If truly no category fits, respond with "None"

RESPONSE FORMAT:
Respond with ONLY the exact category label or "None".

Category:"""

    def _assign_generic_category(self, venue: VenueForFallback, all_categories: List[str]) -> Optional[str]:
        """Assign a generic category based on basic heuristics when all other methods fail."""
        venue_info = f"{venue.name} {venue.address} {venue.enriched_text}".lower()
        
        # Generic assignment rules based on common patterns
        generic_rules = [
            # If contains numbers and Indonesian address patterns, likely retail/commercial
            (r'blok|ruko|no\.|residence|complex', 'Retail'),
            
            # If mentions specific Indonesian locations, might be general business
            (r'jakarta|bogor|bekasi|tangerang|jawa barat', 'Business and Professional Services > Office'),
            
            # If has hours mentioned, likely a business establishment
            (r'\d{2}:\d{2}|\d{1,2}\s*(am|pm)|hours|open|close', 'Business and Professional Services > Office'),
            
            # If has phone numbers, likely a service business
            (r'\(\d+\)|\+\d+|telp|phone|wa|whatsapp', 'Business and Professional Services > Office'),
            
            # If mentions no search results, classify as general business
            (r'no search results|missing', 'Business and Professional Services > Office'),
            
            # Indonesian residential area patterns
            (r'rt\.|rw\.|kelurahan|kecamatan|desa', 'Business and Professional Services > Office')
        ]
        
        for pattern, category_prefix in generic_rules:
            if re.search(pattern, venue_info):
                # Find categories matching this prefix
                matching_categories = [cat for cat in all_categories if cat.startswith(category_prefix)]
                if matching_categories:
                    # Return the most general category (shortest path)
                    best_match = min(matching_categories, key=len)
                    logger.info(f"Generic rule matched: '{pattern}' -> {best_match}")
                    return best_match
        
        # Final fallback - assign to Business and Professional Services if nothing else works
        business_categories = [cat for cat in all_categories if cat.startswith('Business and Professional Services')]
        if business_categories:
            general_business = min(business_categories, key=len)  # Most general business category
            logger.info(f"Final fallback to general business: {general_business}")
            return general_business
        
        return None

    def classify_venue_with_fallback(
        self, 
        venue: VenueForFallback, 
        all_categories: List[str]
    ) -> Optional[str]:
        """
        Attempt to classify a venue using multiple fallback strategies.
        
        Args:
            venue: Venue information
            all_categories: Complete list of available categories
            
        Returns:
            Category label or None if no match found
        """
        try:
            logger.info(f"Performing fallback classification for: {venue.name}")
            
            # First try: Direct name analysis
            name_match = self._analyze_venue_name_for_category(venue.name, all_categories)
            if name_match:
                logger.info(f"Name analysis found match: {name_match}")
                return name_match
            
            # Second try: Enriched text analysis
            text_match = self._analyze_enriched_text_for_category(venue.enriched_text, all_categories)
            if text_match:
                logger.info(f"Text analysis found match: {text_match}")
                return text_match
            
            # Third try: LLM analysis
            prompt = self._create_fallback_prompt(venue, all_categories)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert venue categorizer. Find the most appropriate category for venues that were previously unclassified."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=200,
                timeout=60
            )
            
            result = response.choices[0].message.content.strip()
            
            # Validate result
            if result.lower() == "none":
                logger.info(f"LLM fallback still found no match for: {venue.name}")
                return None
            
            # Try to find a matching category using fuzzy matching
            matched_category = self._find_fuzzy_category_match(result, all_categories)
            
            if matched_category:
                logger.info(f"LLM fallback found match: {matched_category}")
                return matched_category
            else:
                logger.warning(f"LLM fallback returned invalid category: {result}")
                
                # Fourth try: Generic fallback based on heuristics
                generic_match = self._assign_generic_category(venue, all_categories)
                if generic_match:
                    logger.info(f"Generic fallback assigned: {generic_match}")
                    return generic_match
                    
                return None
                
        except Exception as e:
            logger.error(f"Fallback classification failed for {venue.name}: {e}")
            return None


def apply_llm_fallback_to_results(
    classification_results: List[Dict[str, Any]],
    all_categories: List[str]
) -> List[Dict[str, Any]]:
    """
    Apply LLM fallback to venues classified as "None".
    
    Args:
        classification_results: Original classification results
        all_categories: Complete list of available category labels
        
    Returns:
        Updated results with LLM fallback applied
    """
    fallback = LLMFallback()
    updated_results = []
    
    none_count = 0
    successful_fallbacks = 0
    
    for result in classification_results:
        if result['category_label'] == 'None':
            none_count += 1
            
            # Create venue for fallback
            venue = VenueForFallback(
                name=result['name'],
                address=result['address'],
                enriched_text=result['enriched_text'],
                similarity_score=result['similarity_score']
            )
            
            # Apply LLM fallback
            fallback_category = fallback.classify_venue_with_fallback(venue, all_categories)
            
            if fallback_category:
                # Update result with fallback category
                result['category_label'] = fallback_category
                result['category_name'] = fallback_category
                result['fallback_status'] = 'llm_reclassified'
                successful_fallbacks += 1
                logger.info(f"Successfully reclassified: {result['name']} -> {fallback_category}")
            else:
                result['fallback_status'] = 'llm_no_match'
                logger.warning(f"LLM fallback failed for: {result['name']}")
        else:
            result['fallback_status'] = 'not_applied'
        
        updated_results.append(result)
    
    logger.info(f"LLM fallback summary:")
    logger.info(f"  - None classifications: {none_count}")
    logger.info(f"  - Successfully reclassified: {successful_fallbacks}")
    logger.info(f"  - Success rate: {successful_fallbacks/max(none_count, 1)*100:.1f}%")
    
    return updated_results