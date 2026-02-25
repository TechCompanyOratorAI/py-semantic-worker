"""
Semantic Analysis Service

Performs semantic analysis comparing:
1. Transcript segments vs Topic (content relevance)
2. Transcript segments vs Slides (semantic similarity) 
3. Transcript timing vs Slide sequence (alignment)
"""

import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

from config.settings import settings
from utils.logger import get_logger
from utils.exceptions import SemanticAnalysisError, EmbeddingError, SimilarityError
from services.database_service import PresentationData

logger = get_logger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

@dataclass
class SegmentAnalysis:
    """Analysis result for a single transcript segment"""
    segment_id: int
    segment_number: int
    segment_text: str
    
    # Content relevance (vs topic)
    relevance_score: float
    topic_keywords_found: List[str]
    off_topic_indicators: List[str]
    
    # Semantic similarity (vs slides)
    semantic_score: float
    best_matching_slide: Optional[int]
    slide_similarities: List[Tuple[int, float]]
    
    # Alignment (timing vs expected slide)
    alignment_score: float
    expected_slide_number: Optional[int]
    timing_deviation: Optional[float]
    
    # Issues and suggestions
    issues: List[str]
    suggestions: List[str]

@dataclass
class OverallScores:
    """Overall presentation scores"""
    content_relevance: float
    semantic_similarity: float
    slide_alignment: float
    overall_score: float

class SemanticAnalysisService:
    """Service for performing semantic analysis"""
    
    def __init__(self):
        self.model = None
        self.vietnamese_stopwords = set()
        self._initialize()
    
    def _initialize(self):
        """Initialize the service"""
        try:
            logger.info(f"🧠 Loading embedding model: {settings.EMBEDDING_MODEL}")
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info("✅ Embedding model loaded successfully")
            
            # Load Vietnamese stopwords
            self._load_vietnamese_stopwords()
            
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize semantic analysis service: {e}")
    
    def _load_vietnamese_stopwords(self):
        """Load Vietnamese stopwords"""
        try:
            # English stopwords from NLTK
            english_stopwords = set(stopwords.words('english'))
            
            # Common Vietnamese stopwords
            vietnamese_stopwords = {
                'là', 'của', 'và', 'có', 'trong', 'với', 'để', 'cho', 'về', 'từ', 'theo',
                'như', 'khi', 'được', 'sẽ', 'đã', 'này', 'đó', 'những', 'các', 'một',
                'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín', 'mười',
                'tôi', 'bạn', 'anh', 'chị', 'em', 'chúng ta', 'họ', 'nó',
                'thì', 'mà', 'nhưng', 'hoặc', 'nếu', 'vì', 'do', 'bởi',
                'rất', 'khá', 'khôn', 'quá', 'cũng', 'đều', 'chỉ', 'còn'
            }
            
            self.vietnamese_stopwords = english_stopwords.union(vietnamese_stopwords)
            logger.info(f"✅ Loaded {len(self.vietnamese_stopwords)} stopwords")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to load stopwords: {e}")
            self.vietnamese_stopwords = set()
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep Vietnamese characters
        text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_keywords(self, text: str, top_k: int = 10) -> List[str]:
        """Extract keywords from text"""
        try:
            processed_text = self._preprocess_text(text)
            
            # Tokenize
            words = word_tokenize(processed_text)
            
            # Remove stopwords and short words
            keywords = [
                word for word in words 
                if word not in self.vietnamese_stopwords 
                and len(word) > 2
                and word.isalpha()
            ]
            
            # Count frequency and return top keywords
            from collections import Counter
            word_freq = Counter(keywords)
            
            return [word for word, count in word_freq.most_common(top_k)]
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract keywords: {e}")
            return []
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for list of texts"""
        try:
            if not texts:
                return np.array([])
            
            # Filter out empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            if not valid_texts:
                return np.array([])
            
            embeddings = self.model.encode(valid_texts, batch_size=settings.BATCH_SIZE)
            return embeddings
            
        except Exception as e:
            raise EmbeddingError(f"Failed to generate embeddings: {e}")
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            if embedding1.size == 0 or embedding2.size == 0:
                return 0.0
            
            # Reshape if necessary
            if embedding1.ndim == 1:
                embedding1 = embedding1.reshape(1, -1)
            if embedding2.ndim == 1:
                embedding2 = embedding2.reshape(1, -1)
            
            similarity = cosine_similarity(embedding1, embedding2)[0][0]
            return float(similarity)
            
        except Exception as e:
            raise SimilarityError(f"Failed to calculate similarity: {e}")
    
    def _analyze_content_relevance(
        self, 
        segment_text: str, 
        topic_name: str, 
        topic_description: str
    ) -> Tuple[float, List[str], List[str]]:
        """
        Analyze content relevance of segment vs topic
        
        Returns:
            (relevance_score, topic_keywords_found, off_topic_indicators)
        """
        try:
            # Combine topic information
            topic_text = f"{topic_name}. {topic_description or ''}"
            
            # Generate embeddings
            segment_embedding = self._generate_embeddings([segment_text])
            topic_embedding = self._generate_embeddings([topic_text])
            
            # Calculate similarity
            relevance_score = self._calculate_similarity(segment_embedding, topic_embedding)
            
            # Extract keywords from topic and segment
            topic_keywords = self._extract_keywords(topic_text, top_k=15)
            segment_keywords = self._extract_keywords(segment_text, top_k=10)
            
            # Find topic keywords mentioned in segment
            topic_keywords_found = [
                keyword for keyword in topic_keywords
                if keyword in segment_text.lower()
            ]
            
            # Simple heuristic for off-topic indicators
            off_topic_indicators = []
            if relevance_score < settings.SIMILARITY_THRESHOLD * 0.5:
                off_topic_indicators.append("Low semantic similarity to topic")
            
            if not topic_keywords_found:
                off_topic_indicators.append("No topic keywords mentioned")
            
            return relevance_score, topic_keywords_found, off_topic_indicators
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to analyze content relevance: {e}")
            return 0.0, [], ["Analysis failed"]
    
    def _analyze_semantic_similarity(
        self, 
        segment_text: str, 
        slides: List[Dict[str, Any]]
    ) -> Tuple[float, Optional[int], List[Tuple[int, float]]]:
        """
        Analyze semantic similarity of segment vs slides
        
        Returns:
            (best_similarity_score, best_slide_number, all_similarities)
        """
        try:
            if not slides:
                return 0.0, None, []
            
            # Extract slide texts
            slide_texts = []
            slide_numbers = []
            
            for slide in slides:
                slide_text = slide.get('extractedText', '') or ''
                if slide_text.strip():
                    slide_texts.append(slide_text)
                    slide_numbers.append(slide.get('slideNumber', 0))
            
            if not slide_texts:
                return 0.0, None, []
            
            # Generate embeddings
            segment_embedding = self._generate_embeddings([segment_text])
            slide_embeddings = self._generate_embeddings(slide_texts)
            
            # Calculate similarities
            similarities = []
            for i, slide_embedding in enumerate(slide_embeddings):
                similarity = self._calculate_similarity(segment_embedding, slide_embedding)
                similarities.append((slide_numbers[i], similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Get best match
            best_similarity = similarities[0][1] if similarities else 0.0
            best_slide_number = similarities[0][0] if similarities else None
            
            return best_similarity, best_slide_number, similarities
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to analyze semantic similarity: {e}")
            return 0.0, None, []
    
    def _analyze_alignment(
        self, 
        segment_number: int, 
        segment_start_time: float,
        segment_end_time: float,
        total_segments: int,
        total_slides: int,
        presentation_duration: Optional[float] = None
    ) -> Tuple[float, Optional[int], Optional[float]]:
        """
        Analyze timing alignment of segment vs expected slide
        
        Returns:
            (alignment_score, expected_slide_number, timing_deviation)
        """
        try:
            if total_slides == 0:
                return 0.0, None, None
            
            # Simple heuristic: assume slides should be evenly distributed over time
            segment_progress = segment_number / total_segments
            expected_slide_number = max(1, min(total_slides, int(segment_progress * total_slides) + 1))
            
            # Calculate timing deviation
            if presentation_duration and presentation_duration > 0:
                expected_time_progress = expected_slide_number / total_slides
                actual_time_progress = (segment_start_time + segment_end_time) / 2 / presentation_duration
                timing_deviation = abs(expected_time_progress - actual_time_progress)
                
                # Convert to alignment score (1.0 = perfect alignment, 0.0 = completely misaligned)
                alignment_score = max(0.0, 1.0 - timing_deviation * 2)
            else:
                # Fallback: use segment position alignment
                expected_segment_for_slide = (expected_slide_number - 1) * total_segments / total_slides
                position_deviation = abs(segment_number - expected_segment_for_slide) / total_segments
                alignment_score = max(0.0, 1.0 - position_deviation * 2)
                timing_deviation = position_deviation
            
            return alignment_score, expected_slide_number, timing_deviation
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to analyze alignment: {e}")
            return 0.0, None, None
    
    def analyze_presentation(self, presentation_data: PresentationData) -> Tuple[List[SegmentAnalysis], OverallScores]:
        """
        Perform comprehensive semantic analysis of presentation
        
        Args:
            presentation_data: Complete presentation data
            
        Returns:
            (segment_analyses, overall_scores)
        """
        logger.info("🔍 Starting semantic analysis...")
        
        segment_analyses = []
        total_segments = len(presentation_data.transcript_segments)
        total_slides = len(presentation_data.slides)
        
        # Estimate presentation duration from last segment
        presentation_duration = None
        if presentation_data.transcript_segments:
            last_segment = max(presentation_data.transcript_segments, key=lambda s: s.get('endtimestamp', 0))
            presentation_duration = last_segment.get('endtimestamp', 0)
        
        logger.info(f"   - Analyzing {total_segments} segments against {total_slides} slides")
        logger.info(f"   - Topic: {presentation_data.topic_name}")
        logger.info(f"   - Duration: {presentation_duration:.1f}s" if presentation_duration else "   - Duration: Unknown")
        
        # Analyze each segment
        for i, segment in enumerate(presentation_data.transcript_segments):
            segment_id = segment.get('segmentId', 0)
            segment_number = segment.get('segmentNumber', i + 1)
            segment_text = segment.get('segmentText', '')
            start_time = segment.get('startTimestamp', 0)
            end_time = segment.get('endTimestamp', 0)
            
            if not segment_text or not segment_text.strip():
                logger.warning(f"⚠️ Segment {segment_number} has no text, skipping analysis")
                continue
            
            logger.debug(f"Analyzing segment {segment_number}: {segment_text[:50]}...")
            
            # Content relevance analysis
            relevance_score, topic_keywords_found, off_topic_indicators = self._analyze_content_relevance(
                segment_text, 
                presentation_data.topic_name, 
                presentation_data.topic_description or ''
            )
            
            # Semantic similarity analysis
            semantic_score, best_matching_slide, slide_similarities = self._analyze_semantic_similarity(
                segment_text, 
                presentation_data.slides
            )
            
            # Alignment analysis
            alignment_score, expected_slide_number, timing_deviation = self._analyze_alignment(
                segment_number, start_time, end_time, total_segments, total_slides, presentation_duration
            )
            
            # Generate issues and suggestions
            issues = []
            suggestions = []
            
            if relevance_score < settings.SIMILARITY_THRESHOLD:
                issues.append(f"Low topic relevance ({relevance_score:.2f})")
                suggestions.append("Focus more on the main topic")
            
            if semantic_score < settings.SIMILARITY_THRESHOLD:
                issues.append(f"Low slide alignment ({semantic_score:.2f})")
                suggestions.append("Ensure speech content matches slide content")
            
            if alignment_score < 0.5:
                issues.append(f"Poor timing alignment ({alignment_score:.2f})")
                suggestions.append("Adjust pacing to match slide progression")
            
            if off_topic_indicators:
                issues.extend(off_topic_indicators)
            
            # Create segment analysis
            analysis = SegmentAnalysis(
                segment_id=segment_id,
                segment_number=segment_number,
                segment_text=segment_text,
                relevance_score=relevance_score,
                topic_keywords_found=topic_keywords_found,
                off_topic_indicators=off_topic_indicators,
                semantic_score=semantic_score,
                best_matching_slide=best_matching_slide,
                slide_similarities=slide_similarities[:5],  # Top 5 matches
                alignment_score=alignment_score,
                expected_slide_number=expected_slide_number,
                timing_deviation=timing_deviation,
                issues=issues,
                suggestions=suggestions
            )
            
            segment_analyses.append(analysis)
        
        # Calculate overall scores
        if segment_analyses:
            avg_relevance = np.mean([s.relevance_score for s in segment_analyses])
            avg_semantic = np.mean([s.semantic_score for s in segment_analyses])
            avg_alignment = np.mean([s.alignment_score for s in segment_analyses])
            overall = (avg_relevance + avg_semantic + avg_alignment) / 3
        else:
            avg_relevance = avg_semantic = avg_alignment = overall = 0.0
        
        overall_scores = OverallScores(
            content_relevance=float(avg_relevance),
            semantic_similarity=float(avg_semantic),
            slide_alignment=float(avg_alignment),
            overall_score=float(overall)
        )
        
        logger.info("✅ Semantic analysis completed:")
        logger.info(f"   - Content relevance: {overall_scores.content_relevance:.3f}")
        logger.info(f"   - Semantic similarity: {overall_scores.semantic_similarity:.3f}")
        logger.info(f"   - Slide alignment: {overall_scores.slide_alignment:.3f}")
        logger.info(f"   - Overall score: {overall_scores.overall_score:.3f}")
        
        return segment_analyses, overall_scores

# Singleton instance
_semantic_service = None

def get_semantic_service() -> SemanticAnalysisService:
    """Get semantic analysis service singleton"""
    global _semantic_service
    if _semantic_service is None:
        _semantic_service = SemanticAnalysisService()
    return _semantic_service