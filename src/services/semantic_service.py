"""
Semantic Analysis Service

Performs comprehensive analysis including:
1. Transcript segments vs Topic (content relevance)
2. Transcript segments vs Slides (semantic similarity) 
3. Transcript timing vs Slide sequence (alignment)
4. Speech quality analysis (hesitation patterns, fluency)
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
    
    # Speech quality analysis (optional)
    speech_quality: Optional[Dict[str, Any]] = None
    # Speaker label from diarization (e.g. SPEAKER_00, SPEAKER_01)
    speaker_label: Optional[str] = None

@dataclass
class OverallScores:
    """Overall presentation scores"""
    content_relevance: float
    semantic_similarity: float
    slide_alignment: float
    overall_score: float
    
    # Speech quality scores (optional)
    speech_fluency: Optional[float] = None
    speech_clarity: Optional[float] = None
    speech_confidence: Optional[float] = None
    speech_overall: Optional[float] = None

class SemanticAnalysisService:
    """Service for performing semantic analysis"""
    
    def __init__(self):
        self.model = None
        self.vietnamese_stopwords = set()
        self._initialize()
    
    def _initialize(self):
        """Initialize the service with intfloat/multilingual-e5-small"""
        try:
            logger.info(f"🧠 Loading embedding model: {settings.EMBEDDING_MODEL}")
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info("✅ Embedding model loaded — using E5 query/passage prefixes")
            
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
    
    def _generate_embeddings(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """Generate embeddings using intfloat/multilingual-e5-small.
        
        E5 requires specific prefixes to work correctly:
          - is_query=True  → prefix 'query: '   (transcript segments searching for matches)
          - is_query=False → prefix 'passage: ' (topic text / slide content being searched)
        """
        try:
            if not texts:
                return np.array([])
            
            # Filter out empty texts
            valid_texts = [text for text in texts if text and text.strip()]
            if not valid_texts:
                return np.array([])
            
            # E5 requires query/passage prefix — mandatory for accuracy
            prefix = 'query: ' if is_query else 'passage: '
            prefixed_texts = [prefix + t for t in valid_texts]
            
            embeddings = self.model.encode(prefixed_texts, batch_size=settings.BATCH_SIZE)
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
            # segment_text is what we search FOR (query), topic is the document (passage)
            segment_embedding = self._generate_embeddings([segment_text], is_query=True)
            topic_embedding = self._generate_embeddings([topic_text], is_query=False)
            
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
        slides: List[Dict[str, Any]],
        segment_embedding: Optional[np.ndarray] = None,
    ) -> Tuple[float, Optional[int], List[Tuple[int, float]]]:
        """
        Analyze semantic similarity of segment vs slides.

        Applies contrastive normalization to correct multilingual-E5 baseline
        inflation: same-language pairs score 0.55–0.75 even when unrelated.
        Calibrated score = (best_raw - mean_raw) / (1 - mean_raw)
        → ≈ 0 when all slides uniformly unrelated
        → high only when there is a clear best-matching slide

        Returns:
            (calibrated_score, best_slide_number, raw_similarities)
        """
        try:
            if not slides:
                return 0.0, None, []

            # Extract slide texts
            slide_texts, slide_numbers = [], []
            for slide in slides:
                slide_text = slide.get('extractedText', '') or ''
                if slide_text.strip():
                    slide_texts.append(slide_text)
                    slide_numbers.append(slide.get('slideNumber', 0))

            if not slide_texts:
                return 0.0, None, []

            # Use pre-computed embedding if available to avoid double inference
            if segment_embedding is None or (
                hasattr(segment_embedding, 'size') and segment_embedding.size == 0
            ):
                segment_embedding = self._generate_embeddings([segment_text], is_query=True)

            slide_embeddings = self._generate_embeddings(slide_texts, is_query=False)

            # Compute raw cosine similarities
            similarities = []
            for i, slide_emb in enumerate(slide_embeddings):
                raw_sim = self._calculate_similarity(segment_embedding, slide_emb)
                similarities.append((slide_numbers[i], raw_sim))

            similarities.sort(key=lambda x: x[1], reverse=True)

            best_raw    = similarities[0][1] if similarities else 0.0
            best_slide  = similarities[0][0] if similarities else None

            # --- Contrastive normalization ---
            # calibrated = (best - mean) / (1 - mean)
            # If ALL slides score similarly (all unrelated), calibrated ≈ 0.
            # If one slide clearly dominates, calibrated is high.
            if len(similarities) > 1:
                all_raw   = [s[1] for s in similarities]
                mean_raw  = float(np.mean(all_raw))
                denom     = 1.0 - mean_raw
                calibrated = (best_raw - mean_raw) / denom if denom > 1e-6 else 0.0
                calibrated = max(0.0, min(1.0, calibrated))
            else:
                # Only 1 slide — shift by ~0.5 baseline and normalise
                calibrated = max(0.0, min(1.0, (best_raw - 0.5) * 2.0))

            return calibrated, best_slide, similarities

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
        presentation_duration: Optional[float] = None,
        segment_embedding: Optional[np.ndarray] = None,
        slides: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[float, Optional[int], Optional[float]]:
        """
        Analyze timing alignment of segment vs expected slide.

        Strict scoring — 35 % temporal position + 65 % content match:
        - temporal_score  : how close is timing to the expected slide
        - content_match   : cosine(segment, EXPECTED slide text), baseline-corrected

        If the speaker talks about an unrelated topic the content-match term
        drives the score to ~0 regardless of temporal ordering.

        Returns:
            (alignment_score, expected_slide_number, timing_deviation)
        """
        TEMPORAL_WEIGHT = 0.35
        CONTENT_WEIGHT  = 0.65
        # Typical E5 cosine for unrelated same-language pairs ≈ 0.55
        CONTENT_BASELINE = 0.55

        try:
            if total_slides == 0:
                return 0.0, None, None

            # 1. Expected slide from temporal progress
            segment_progress     = segment_number / total_segments
            expected_slide_number = max(1, min(total_slides, int(segment_progress * total_slides) + 1))

            # 2. Temporal score
            if presentation_duration and presentation_duration > 0:
                expected_t = expected_slide_number / total_slides
                actual_t   = (
                    (segment_start_time + segment_end_time) / 2 / presentation_duration
                )
                timing_deviation = abs(expected_t - actual_t)
                temporal_score   = max(0.0, 1.0 - timing_deviation * 2)
            else:
                expected_seg_pos = (expected_slide_number - 1) * total_segments / total_slides
                position_dev     = abs(segment_number - expected_seg_pos) / total_segments
                temporal_score   = max(0.0, 1.0 - position_dev * 2)
                timing_deviation = position_dev

            # 3. Content match: segment vs the EXPECTED slide page
            content_match_score = 0.0
            if segment_embedding is not None and slides:
                idx = expected_slide_number - 1  # 0-indexed
                if 0 <= idx < len(slides):
                    expected_text = slides[idx].get('extractedText', '') or ''
                    if expected_text.strip():
                        try:
                            expected_emb   = self._generate_embeddings([expected_text], is_query=False)
                            raw_content    = self._calculate_similarity(segment_embedding, expected_emb)
                            # Baseline-correct: shift out the ≈0.55 floor
                            denom = 1.0 - CONTENT_BASELINE
                            content_match_score = max(
                                0.0,
                                min(1.0, (raw_content - CONTENT_BASELINE) / denom)
                            )
                        except Exception:
                            content_match_score = 0.0

            # 4. Blend (strict: content dominates)
            if segment_embedding is not None and slides:
                alignment_score = (
                    TEMPORAL_WEIGHT * temporal_score
                    + CONTENT_WEIGHT  * content_match_score
                )
            else:
                # Fallback when no embeddings available
                alignment_score = temporal_score

            return alignment_score, expected_slide_number, timing_deviation

        except Exception as e:
            logger.warning(f"⚠️ Failed to analyze alignment: {e}")
            return 0.0, None, None

    # ------------------------------------------------------------------
    # Helper: expand slides with multi-page extractedText into virtual
    # per-page entries so alignment analysis maps segments to individual
    # pages rather than treating 31 pages as a single slide.
    # Detects "[Trang N]" page markers written by the Node API webhook.
    # ------------------------------------------------------------------
    def _expand_slides_by_pages(self, slides: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Expand multi-page slides into virtual per-page slide entries."""
        expanded = []
        virtual_number = 0

        for slide in slides:
            extracted_text = slide.get('extractedText', '') or ''
            # Split on [Trang N] / [Slide N] markers (case-insensitive)
            parts = re.split(r'\[(?:Trang|Slide)\s+(\d+)\]\s*\n?', extracted_text, flags=re.IGNORECASE)
            # parts = ['pre', '1', 'content1', '2', 'content2', ...]
            # If no markers found, parts == [extracted_text] (length 1)
            pages = []
            if len(parts) >= 3:
                for idx in range(1, len(parts) - 1, 2):
                    page_num = int(parts[idx])
                    page_text = parts[idx + 1].strip() if idx + 1 < len(parts) else ''
                    pages.append((page_num, page_text))

            if len(pages) > 1:
                logger.info(
                    f"   - Slide {slide.get('slideNumber', '?')}: expanding "
                    f"{len(pages)} pages into virtual slides for alignment"
                )
                for page_num, page_text in pages:
                    virtual_number += 1
                    virtual = dict(slide)
                    virtual['slideNumber'] = virtual_number
                    virtual['extractedText'] = page_text
                    virtual['_virtualPage'] = page_num
                    virtual['_originalSlideId'] = slide.get('slideId')
                    expanded.append(virtual)
            else:
                # Single-page or no markers — use as-is
                virtual_number += 1
                entry = dict(slide)
                entry['slideNumber'] = virtual_number
                expanded.append(entry)

        return expanded

    def analyze_presentation(
        self, 
        presentation_data: PresentationData,
        audio_file_path: Optional[str] = None
    ) -> Tuple[List[SegmentAnalysis], OverallScores, Any]:
        """
        Perform comprehensive semantic analysis of presentation
        
        Args:
            presentation_data: Complete presentation data
            audio_file_path: Optional path to audio file for speech quality analysis
            
        Returns:
            (segment_analyses, overall_scores, speech_quality_metrics)
            speech_quality_metrics is None if audio analysis was not performed
        """
        logger.info("🔍 Starting comprehensive semantic analysis...")
        
        segment_analyses = []
        total_segments = len(presentation_data.transcript_segments)

        # Expand multi-page slides into virtual per-page entries.
        # A PDF with 31 pages stored in 1 Slide row becomes 31 virtual
        # slides so that alignment scores reflect actual page progress.
        expanded_slides = self._expand_slides_by_pages(presentation_data.slides)
        total_slides = len(expanded_slides)
        if total_slides != len(presentation_data.slides):
            logger.info(
                f"   - Expanded {len(presentation_data.slides)} DB slide(s) "
                f"→ {total_slides} virtual pages for alignment analysis"
            )
        
        # Initialize speech quality analysis if enabled and audio file provided
        speech_quality_metrics = None
        if settings.SPEECH_ANALYSIS_ENABLED and audio_file_path:
            try:
                from pathlib import Path
                from services.speech_quality_service import get_speech_quality_service
                
                logger.info("🎤 Performing speech quality analysis...")
                speech_service = get_speech_quality_service()
                speech_quality_metrics = speech_service.analyze_speech_quality(Path(audio_file_path))
                logger.info("✅ Speech quality analysis completed")
                
            except Exception as e:
                logger.warning(f"⚠️ Speech quality analysis failed: {e}")
                speech_quality_metrics = None
        
        # Estimate presentation duration from last segment
        presentation_duration = None
        if presentation_data.transcript_segments:
            last_segment = max(presentation_data.transcript_segments,
                               key=lambda s: s.get('endTimestamp') or s.get('endtimestamp') or 0)
            raw_dur = last_segment.get('endTimestamp') or last_segment.get('endtimestamp') or 0
            # Normalize to seconds (same heuristic as _to_seconds)
            presentation_duration = raw_dur / 1000.0 if raw_dur > 1000 else (raw_dur or None)

        
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
            # Speaker label from diarization (may be None if not diarized)
            speaker_label = (
                segment.get('aiSpeakerLabel')
                or segment.get('speakerName')
                or None
            )

            # Normalize timestamps to seconds.
            # ASR worker stores timestamps in milliseconds (e.g. 5200),
            # while librosa produces times in seconds (e.g. 5.2).
            # Heuristic: if value > 1000 it is almost certainly milliseconds.
            def _to_seconds(ts):
                return ts / 1000.0 if ts is not None and ts > 1000 else (ts or 0.0)

            start_time_s = _to_seconds(start_time)
            end_time_s   = _to_seconds(end_time)
            
            if not segment_text or not segment_text.strip():
                logger.warning(f"⚠️ Segment {segment_number} has no text, skipping analysis")
                continue
            
            logger.debug(f"Analyzing segment {segment_number}: {segment_text[:50]}...")

            # Generate segment embedding ONCE — reused by semantic similarity AND
            # alignment (content-match term). Avoids redundant model inference.
            try:
                segment_embedding = self._generate_embeddings([segment_text], is_query=True)
                if not hasattr(segment_embedding, 'size') or segment_embedding.size == 0:
                    segment_embedding = None
            except Exception as emb_err:
                logger.warning(f"⚠️ Embedding failed for segment {segment_number}: {emb_err}")
                segment_embedding = None

            # Content relevance analysis (topic match)
            relevance_score, topic_keywords_found, off_topic_indicators = self._analyze_content_relevance(
                segment_text,
                presentation_data.topic_name,
                presentation_data.topic_description or ''
            )

            # Semantic similarity — contrastive-normalised against all virtual pages
            semantic_score, best_matching_slide, slide_similarities = self._analyze_semantic_similarity(
                segment_text,
                expanded_slides,
                segment_embedding=segment_embedding,
            )

            # Alignment — strict blend: 35 % temporal + 65 % content match with expected slide
            alignment_score, expected_slide_number, timing_deviation = self._analyze_alignment(
                segment_number, start_time_s, end_time_s, total_segments, total_slides,
                presentation_duration,
                segment_embedding=segment_embedding,
                slides=expanded_slides,
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
            
            # Add speech quality information for this segment if available
            segment_speech_quality = None
            if speech_quality_metrics:
                # Find hesitation patterns that overlap with this segment
                segment_hesitations = []
                for pattern in speech_quality_metrics.hesitation_patterns:
                    # Compare in the same unit: seconds (librosa) vs seconds (normalized)
                    if (pattern.start_time <= end_time_s and pattern.end_time >= start_time_s):
                        segment_hesitations.append({
                            'startTime': pattern.start_time,
                            'endTime': pattern.end_time,
                            'duration': pattern.duration,
                            'pattern_type': pattern.pattern_type,
                            'confidence': pattern.confidence,
                            'description': pattern.description
                        })
                
                total_hesitation_time = sum(p['duration'] for p in segment_hesitations)
                segment_duration = end_time - start_time
                
                # Always set speech_quality for segments when audio analysis is available
                # (even if no hesitations found — so SegmentSpeechQuality row is always created)
                segment_speech_quality = {
                    'hesitationPatterns': segment_hesitations,
                    'hesitationCount': len(segment_hesitations),
                    'totalHesitationTime': total_hesitation_time
                }
                
                if segment_hesitations:
                    # Add speech-related issues and suggestions
                    if len(segment_hesitations) > 1:
                        issues.append(f"Multiple hesitations detected ({len(segment_hesitations)})")
                        suggestions.append("Practice this section to reduce hesitations")
                    
                    if segment_duration > 0 and total_hesitation_time / segment_duration > 0.2:
                        issues.append("High hesitation ratio in this segment")
                        suggestions.append("Focus on smoother delivery for this part")

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
                suggestions=suggestions,
                speech_quality=segment_speech_quality,
                speaker_label=speaker_label
            )
            
            segment_analyses.append(analysis)
        
        # Calculate overall scores (structural dimensions only)
        # NOTE: contentRelevance is stored per-segment for reference but is NOT
        # included in overallScore. Embedding similarity cannot reliably judge
        # whether the speaker actually covered the topic — that requires LLM
        # understanding. overallScore = structural quality only (semantic + alignment).
        if segment_analyses:
            avg_relevance = np.mean([s.relevance_score for s in segment_analyses])
            avg_semantic  = np.mean([s.semantic_score  for s in segment_analyses])
            avg_alignment = np.mean([s.alignment_score for s in segment_analyses])
            # 50% slide content match + 50% slide ordering alignment
            overall = avg_semantic * 0.5 + avg_alignment * 0.5
        else:
            avg_relevance = avg_semantic = avg_alignment = overall = 0.0
        
        # Include speech quality scores if available
        speech_fluency = speech_clarity = speech_confidence = speech_overall = None
        if speech_quality_metrics:
            speech_fluency = float(speech_quality_metrics.fluency_score)
            speech_clarity = float(speech_quality_metrics.clarity_score)
            speech_confidence = float(speech_quality_metrics.confidence_score)
            speech_overall = float(speech_quality_metrics.overall_quality)
            
            # Adjust overall score to include speech quality (weighted)
            overall = (overall * 0.7 + speech_overall * 0.3)  # 70% semantic, 30% speech
        
        overall_scores = OverallScores(
            content_relevance=float(avg_relevance),
            semantic_similarity=float(avg_semantic),
            slide_alignment=float(avg_alignment),
            overall_score=float(overall),
            speech_fluency=speech_fluency,
            speech_clarity=speech_clarity,
            speech_confidence=speech_confidence,
            speech_overall=speech_overall
        )
        
        logger.info("✅ Comprehensive analysis completed:")
        logger.info(f"   - Content relevance: {overall_scores.content_relevance:.3f}")
        logger.info(f"   - Semantic similarity: {overall_scores.semantic_similarity:.3f}")
        logger.info(f"   - Slide alignment: {overall_scores.slide_alignment:.3f}")
        if speech_quality_metrics:
            logger.info(f"   - Speech fluency: {overall_scores.speech_fluency:.3f}")
            logger.info(f"   - Speech clarity: {overall_scores.speech_clarity:.3f}")
            logger.info(f"   - Speech confidence: {overall_scores.speech_confidence:.3f}")
            logger.info(f"   - Speech overall: {overall_scores.speech_overall:.3f}")
        logger.info(f"   - Overall score: {overall_scores.overall_score:.3f}")
        
        return segment_analyses, overall_scores, speech_quality_metrics

# Singleton instance
_semantic_service = None

def get_semantic_service() -> SemanticAnalysisService:
    """Get semantic analysis service singleton"""
    global _semantic_service
    if _semantic_service is None:
        _semantic_service = SemanticAnalysisService()
    return _semantic_service