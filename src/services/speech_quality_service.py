"""
Speech Quality Analysis Service

Analyzes speech quality using openSMILE to detect:
1. Hesitation patterns (ấp úng)
2. Speech fluency
3. Voice quality metrics
4. Prosodic features
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import opensmile
from pydub import AudioSegment
import tempfile

from config.settings import settings
from utils.logger import get_logger
from utils.exceptions import SemanticWorkerError

logger = get_logger(__name__)

class SpeechAnalysisError(SemanticWorkerError):
    """Speech analysis specific error"""
    pass

@dataclass
class HesitationPattern:
    """Represents a detected hesitation pattern"""
    start_time: float
    end_time: float
    duration: float
    pattern_type: str  # 'silence', 'filler', 'repetition'
    confidence: float
    description: str

@dataclass
class SpeechQualityMetrics:
    """Speech quality analysis results"""
    # Overall scores (0-1, higher is better)
    fluency_score: float
    clarity_score: float
    confidence_score: float
    overall_quality: float
    
    # Hesitation analysis
    hesitation_patterns: List[HesitationPattern]
    hesitation_count: int
    hesitation_rate: float  # hesitations per minute
    total_hesitation_time: float
    
    # Voice quality metrics
    pitch_mean: float
    pitch_std: float
    energy_mean: float
    energy_std: float
    speaking_rate: float  # words per minute
    
    # Prosodic features
    pitch_variation: float
    volume_variation: float
    speech_rhythm_score: float
    
    # Detailed analysis
    silence_ratio: float
    voiced_ratio: float
    spectral_centroid_mean: float
    mfcc_features: List[float]
    
    # Issues and suggestions
    issues: List[str]
    suggestions: List[str]

class SpeechQualityService:
    """Service for analyzing speech quality"""
    
    def __init__(self):
        self.smile = None
        self.feature_set = settings.OPENSMILE_CONFIG
        self._initialize()
    
    def _initialize(self):
        """Initialize openSMILE and other components"""
        try:
            logger.info(f"🎤 Initializing speech quality service...")
            logger.info(f"   - Feature set: {self.feature_set}")
            logger.info(f"   - Sample rate: {settings.SPEECH_SAMPLE_RATE}Hz")
            
            # Initialize openSMILE
            self.smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
            
            logger.info("✅ Speech quality service initialized successfully")
            
        except Exception as e:
            raise SpeechAnalysisError(f"Failed to initialize speech quality service: {e}")
    
    def _preprocess_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """
        Preprocess audio file for analysis
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            (audio_data, sample_rate)
        """
        temp_wav_path = None
        try:
            logger.info(f"🔊 Preprocessing audio file: {audio_path}")

            load_path = str(audio_path)

            # soundfile cannot decode MP4/MP3/etc. – convert to WAV first
            # to avoid "PySoundFile failed" warning and deprecated audioread fallback.
            suffix = Path(audio_path).suffix.lower()
            if suffix not in ('.wav', '.flac', '.ogg', '.aiff', '.aif'):
                logger.info(f"   - Converting {suffix} → WAV for librosa compatibility")
                audio_segment = AudioSegment.from_file(str(audio_path))
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    temp_wav_path = tmp.name
                audio_segment.export(temp_wav_path, format='wav')
                load_path = temp_wav_path

            # Load audio file (soundfile reads WAV natively – no audioread fallback)
            audio_data, sr = librosa.load(
                load_path,
                sr=settings.SPEECH_SAMPLE_RATE,
                mono=True
            )
            
            # Normalize audio
            audio_data = librosa.util.normalize(audio_data)
            
            # Remove silence from beginning and end
            audio_data, _ = librosa.effects.trim(audio_data, top_db=20)
            
            duration = len(audio_data) / sr
            logger.info(f"   - Duration: {duration:.2f}s")
            logger.info(f"   - Sample rate: {sr}Hz")
            logger.info(f"   - Samples: {len(audio_data):,}")
            
            return audio_data, sr
            
        except Exception as e:
            raise SpeechAnalysisError(f"Failed to preprocess audio: {e}")
        finally:
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.unlink(temp_wav_path)
                except OSError:
                    pass
    
    def _detect_voice_activity(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """
        Detect voice activity in audio
        
        Args:
            audio_data: Audio signal
            sr: Sample rate
            
        Returns:
            Boolean array indicating voice activity
        """
        try:
            # Use energy-based voice activity detection
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.010 * sr)    # 10ms hop
            
            # Calculate short-time energy
            energy = librosa.feature.rms(
                y=audio_data,
                frame_length=frame_length,
                hop_length=hop_length
            )[0]
            
            # Dynamic threshold based on energy distribution
            energy_threshold = np.percentile(energy, 30)  # Bottom 30% is likely silence
            
            # Voice activity detection
            voice_activity = energy > energy_threshold
            
            return voice_activity
            
        except Exception as e:
            logger.warning(f"⚠️ Voice activity detection failed: {e}")
            # Fallback: assume all frames are voiced
            return np.ones(len(audio_data) // int(0.010 * sr), dtype=bool)
    
    def _detect_hesitation_patterns(
        self, 
        audio_data: np.ndarray, 
        sr: int,
        voice_activity: np.ndarray
    ) -> List[HesitationPattern]:
        """
        Detect hesitation patterns in speech
        
        Args:
            audio_data: Audio signal
            sr: Sample rate
            voice_activity: Voice activity detection results
            
        Returns:
            List of detected hesitation patterns
        """
        patterns = []
        
        try:
            hop_length = int(0.010 * sr)  # 10ms hop
            frame_times = librosa.frames_to_time(
                np.arange(len(voice_activity)), 
                sr=sr, 
                hop_length=hop_length
            )
            
            # Detect silence gaps (potential hesitations)
            silence_frames = ~voice_activity
            silence_segments = self._find_segments(silence_frames, frame_times)
            
            for start_time, end_time in silence_segments:
                duration = end_time - start_time
                
                # Consider silences longer than 200ms as potential hesitations
                if duration > 0.2:
                    confidence = min(1.0, duration / 2.0)  # Longer silences = higher confidence
                    
                    pattern = HesitationPattern(
                        start_time=start_time,
                        end_time=end_time,
                        duration=duration,
                        pattern_type='silence',
                        confidence=confidence,
                        description=f"Silence gap ({duration:.2f}s)"
                    )
                    patterns.append(pattern)
            
            # Detect repetitive patterns in voiced segments
            voiced_segments = self._find_segments(voice_activity, frame_times)
            
            for start_time, end_time in voiced_segments:
                duration = end_time - start_time
                
                # Analyze short voiced segments (potential fillers like "ừm", "à")
                if 0.1 < duration < 1.0:
                    # Extract audio segment
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    segment = audio_data[start_sample:end_sample]
                    
                    # Simple heuristic: low spectral complexity might indicate fillers
                    # Cap n_fft to avoid 'n_fft too large' warning for very short segments
                    seg_len = len(segment)
                    if seg_len < 2:
                        continue  # Too short to analyze
                    safe_n_fft = 2 ** int(np.log2(seg_len)) if seg_len < 2048 else 2048
                    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr, n_fft=safe_n_fft))
                    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr, n_fft=safe_n_fft))
                    
                    # Low complexity + short duration = potential filler
                    complexity_score = (spectral_centroid / 1000) * (spectral_bandwidth / 1000)
                    
                    if complexity_score < 0.5:  # Threshold for low complexity
                        confidence = 1.0 - complexity_score
                        
                        pattern = HesitationPattern(
                            start_time=start_time,
                            end_time=end_time,
                            duration=duration,
                            pattern_type='filler',
                            confidence=confidence,
                            description=f"Potential filler word ({duration:.2f}s)"
                        )
                        patterns.append(pattern)
            
            logger.info(f"   - Detected {len(patterns)} hesitation patterns")
            
            return patterns
            
        except Exception as e:
            logger.warning(f"⚠️ Hesitation pattern detection failed: {e}")
            return []
    
    def _find_segments(self, activity: np.ndarray, times: np.ndarray) -> List[Tuple[float, float]]:
        """Find continuous segments where activity is True"""
        segments = []
        
        if len(activity) == 0:
            return segments
        
        # Find start and end points of segments
        diff = np.diff(np.concatenate(([False], activity, [False])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for start_idx, end_idx in zip(starts, ends):
            if start_idx < len(times) and end_idx <= len(times):
                start_time = times[start_idx] if start_idx < len(times) else times[-1]
                end_time = times[end_idx - 1] if end_idx > 0 else times[0]
                segments.append((start_time, end_time))
        
        return segments
    
    def _extract_opensmile_features(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract features using openSMILE
        
        Args:
            audio_data: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary of extracted features
        """
        temp_path = None
        try:
            # Create temp file – close it BEFORE openSMILE opens it.
            # On Windows, keeping the file open (via 'with' context) causes
            # [WinError 32] when openSMILE tries to read it.
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Write audio to the (now-closed) temporary file
            sf.write(temp_path, audio_data, sr)
            
            # Extract features using openSMILE
            features = self.smile.process_file(temp_path)
            
            # Convert to dictionary
            feature_dict = {}
            if not features.empty:
                for col in features.columns:
                    feature_dict[col] = float(features[col].iloc[0])
            
            logger.info(f"   - Extracted {len(feature_dict)} openSMILE features")
            return feature_dict
                
        except Exception as e:
            logger.warning(f"⚠️ openSMILE feature extraction failed: {e}")
            return {}
        finally:
            # Always clean up temp file, even if openSMILE fails
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
    
    def _calculate_speech_metrics(
        self, 
        audio_data: np.ndarray, 
        sr: int,
        voice_activity: np.ndarray,
        opensmile_features: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate various speech quality metrics
        
        Args:
            audio_data: Audio signal
            sr: Sample rate
            voice_activity: Voice activity detection results
            opensmile_features: Features from openSMILE
            
        Returns:
            Dictionary of calculated metrics
        """
        metrics = {}
        
        try:
            # Basic audio metrics
            duration = len(audio_data) / sr
            voiced_frames = np.sum(voice_activity)
            total_frames = len(voice_activity)
            
            metrics['duration'] = duration
            metrics['voiced_ratio'] = voiced_frames / total_frames if total_frames > 0 else 0
            metrics['silence_ratio'] = 1.0 - metrics['voiced_ratio']
            
            # Pitch analysis
            f0 = librosa.yin(audio_data, fmin=50, fmax=400, sr=sr)
            f0_voiced = f0[f0 > 0]  # Remove unvoiced frames
            
            if len(f0_voiced) > 0:
                metrics['pitch_mean'] = float(np.mean(f0_voiced))
                metrics['pitch_std'] = float(np.std(f0_voiced))
                metrics['pitch_variation'] = metrics['pitch_std'] / metrics['pitch_mean'] if metrics['pitch_mean'] > 0 else 0
            else:
                metrics['pitch_mean'] = 0
                metrics['pitch_std'] = 0
                metrics['pitch_variation'] = 0
            
            # Energy analysis
            energy = librosa.feature.rms(y=audio_data)[0]
            metrics['energy_mean'] = float(np.mean(energy))
            metrics['energy_std'] = float(np.std(energy))
            metrics['volume_variation'] = metrics['energy_std'] / metrics['energy_mean'] if metrics['energy_mean'] > 0 else 0
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
            metrics['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            
            # MFCC features (first 13 coefficients)
            mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
            metrics['mfcc_features'] = [float(np.mean(mfcc)) for mfcc in mfccs]
            
            # Speaking rate estimation (rough approximation)
            # Estimate based on voiced segments and typical syllable rate
            voiced_duration = metrics['voiced_ratio'] * duration
            estimated_syllables = voiced_duration * 3  # ~3 syllables per second average
            metrics['speaking_rate'] = (estimated_syllables * 60) / duration if duration > 0 else 0  # syllables per minute
            
            # Speech rhythm (based on energy variation in voiced segments)
            if voiced_frames > 0:
                voiced_energy = energy[voice_activity[:len(energy)]]
                if len(voiced_energy) > 1:
                    rhythm_variation = np.std(voiced_energy) / np.mean(voiced_energy)
                    metrics['speech_rhythm_score'] = max(0, 1.0 - rhythm_variation)  # Higher = more rhythmic
                else:
                    metrics['speech_rhythm_score'] = 0.5
            else:
                metrics['speech_rhythm_score'] = 0
            
            # Integrate openSMILE features
            metrics.update(opensmile_features)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"⚠️ Speech metrics calculation failed: {e}")
            return metrics
    
    def _calculate_quality_scores(
        self, 
        metrics: Dict[str, float],
        hesitation_patterns: List[HesitationPattern]
    ) -> Tuple[float, float, float, float]:
        """
        Calculate overall quality scores
        
        Args:
            metrics: Calculated speech metrics
            hesitation_patterns: Detected hesitation patterns
            
        Returns:
            (fluency_score, clarity_score, confidence_score, overall_quality)
        """
        try:
            # Fluency score (based on hesitations and speech rhythm)
            hesitation_penalty = min(1.0, len(hesitation_patterns) / 10.0)  # Penalty for too many hesitations
            rhythm_score = metrics.get('speech_rhythm_score', 0.5)
            silence_penalty = min(1.0, metrics.get('silence_ratio', 0.5) / 0.5)  # Penalty for too much silence
            
            fluency_score = max(0.0, 1.0 - hesitation_penalty - silence_penalty * 0.3) * rhythm_score
            
            # Clarity score (based on spectral features and energy)
            spectral_clarity = min(1.0, metrics.get('spectral_centroid_mean', 1000) / 2000)  # Higher frequency = clearer
            energy_consistency = 1.0 - min(1.0, metrics.get('volume_variation', 1.0))  # Less variation = more consistent
            voiced_ratio = metrics.get('voiced_ratio', 0.5)
            
            clarity_score = (spectral_clarity * 0.4 + energy_consistency * 0.3 + voiced_ratio * 0.3)
            
            # Confidence score (based on pitch variation and energy)
            pitch_confidence = 1.0 - min(1.0, metrics.get('pitch_variation', 1.0))  # Less variation = more confident
            energy_level = min(1.0, metrics.get('energy_mean', 0.1) / 0.2)  # Higher energy = more confident
            speaking_rate_score = min(1.0, metrics.get('speaking_rate', 120) / 180)  # Optimal around 150-180 spm
            
            confidence_score = (pitch_confidence * 0.4 + energy_level * 0.3 + speaking_rate_score * 0.3)
            
            # Overall quality (weighted average)
            overall_quality = (fluency_score * 0.4 + clarity_score * 0.3 + confidence_score * 0.3)
            
            # Ensure scores are in [0, 1] range
            fluency_score = max(0.0, min(1.0, fluency_score))
            clarity_score = max(0.0, min(1.0, clarity_score))
            confidence_score = max(0.0, min(1.0, confidence_score))
            overall_quality = max(0.0, min(1.0, overall_quality))
            
            return fluency_score, clarity_score, confidence_score, overall_quality
            
        except Exception as e:
            logger.warning(f"⚠️ Quality score calculation failed: {e}")
            return 0.5, 0.5, 0.5, 0.5
    
    def _generate_issues_and_suggestions(
        self, 
        metrics: Dict[str, float],
        hesitation_patterns: List[HesitationPattern],
        quality_scores: Tuple[float, float, float, float]
    ) -> Tuple[List[str], List[str]]:
        """
        Generate issues and suggestions based on analysis
        
        Args:
            metrics: Calculated speech metrics
            hesitation_patterns: Detected hesitation patterns
            quality_scores: Calculated quality scores
            
        Returns:
            (issues, suggestions)
        """
        issues = []
        suggestions = []
        
        fluency_score, clarity_score, confidence_score, overall_quality = quality_scores
        
        # Fluency issues
        if fluency_score < 0.6:
            issues.append(f"Low speech fluency ({fluency_score:.2f})")
            suggestions.append("Practice speaking more smoothly, reduce hesitations")
        
        if len(hesitation_patterns) > 5:
            issues.append(f"Frequent hesitations ({len(hesitation_patterns)} detected)")
            suggestions.append("Prepare content better to reduce 'ừm', 'à' and long pauses")
        
        # Clarity issues
        if clarity_score < 0.6:
            issues.append(f"Low speech clarity ({clarity_score:.2f})")
            suggestions.append("Speak more clearly, improve articulation")
        
        if metrics.get('voiced_ratio', 1.0) < 0.7:
            issues.append("Too much silence in speech")
            suggestions.append("Reduce long pauses, maintain consistent speaking pace")
        
        # Confidence issues
        if confidence_score < 0.6:
            issues.append(f"Low confidence indicators ({confidence_score:.2f})")
            suggestions.append("Speak with more energy and consistent tone")
        
        speaking_rate = metrics.get('speaking_rate', 150)
        if speaking_rate < 100:
            issues.append("Speaking too slowly")
            suggestions.append("Increase speaking pace for better engagement")
        elif speaking_rate > 200:
            issues.append("Speaking too quickly")
            suggestions.append("Slow down for better comprehension")
        
        # Pitch variation
        if metrics.get('pitch_variation', 0.5) > 0.8:
            issues.append("Inconsistent pitch variation")
            suggestions.append("Practice maintaining steady vocal tone")
        
        return issues, suggestions
    
    def analyze_speech_quality(self, audio_path: Path) -> SpeechQualityMetrics:
        """
        Perform comprehensive speech quality analysis
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            SpeechQualityMetrics with analysis results
        """
        logger.info("🎤 Starting speech quality analysis...")
        logger.info(f"   - Audio file: {audio_path}")
        
        try:
            # Step 1: Preprocess audio
            audio_data, sr = self._preprocess_audio(audio_path)
            
            # Step 2: Voice activity detection
            voice_activity = self._detect_voice_activity(audio_data, sr)
            
            # Step 3: Extract openSMILE features
            opensmile_features = self._extract_opensmile_features(audio_data, sr)
            
            # Step 4: Calculate speech metrics
            metrics = self._calculate_speech_metrics(audio_data, sr, voice_activity, opensmile_features)
            
            # Step 5: Detect hesitation patterns
            hesitation_patterns = self._detect_hesitation_patterns(audio_data, sr, voice_activity)
            
            # Step 6: Calculate quality scores
            quality_scores = self._calculate_quality_scores(metrics, hesitation_patterns)
            fluency_score, clarity_score, confidence_score, overall_quality = quality_scores
            
            # Step 7: Generate issues and suggestions
            issues, suggestions = self._generate_issues_and_suggestions(metrics, hesitation_patterns, quality_scores)
            
            # Calculate hesitation statistics
            total_hesitation_time = sum(p.duration for p in hesitation_patterns)
            hesitation_rate = (len(hesitation_patterns) / metrics['duration']) * 60 if metrics['duration'] > 0 else 0
            
            # Create result object
            result = SpeechQualityMetrics(
                fluency_score=fluency_score,
                clarity_score=clarity_score,
                confidence_score=confidence_score,
                overall_quality=overall_quality,
                hesitation_patterns=hesitation_patterns,
                hesitation_count=len(hesitation_patterns),
                hesitation_rate=hesitation_rate,
                total_hesitation_time=total_hesitation_time,
                pitch_mean=metrics.get('pitch_mean', 0),
                pitch_std=metrics.get('pitch_std', 0),
                energy_mean=metrics.get('energy_mean', 0),
                energy_std=metrics.get('energy_std', 0),
                speaking_rate=metrics.get('speaking_rate', 0),
                pitch_variation=metrics.get('pitch_variation', 0),
                volume_variation=metrics.get('volume_variation', 0),
                speech_rhythm_score=metrics.get('speech_rhythm_score', 0),
                silence_ratio=metrics.get('silence_ratio', 0),
                voiced_ratio=metrics.get('voiced_ratio', 0),
                spectral_centroid_mean=metrics.get('spectral_centroid_mean', 0),
                mfcc_features=metrics.get('mfcc_features', []),
                issues=issues,
                suggestions=suggestions
            )
            
            logger.info("✅ Speech quality analysis completed:")
            logger.info(f"   - Overall quality: {overall_quality:.3f}")
            logger.info(f"   - Fluency: {fluency_score:.3f}")
            logger.info(f"   - Clarity: {clarity_score:.3f}")
            logger.info(f"   - Confidence: {confidence_score:.3f}")
            logger.info(f"   - Hesitations: {len(hesitation_patterns)}")
            logger.info(f"   - Issues found: {len(issues)}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Speech quality analysis failed: {e}", exc_info=True)
            raise SpeechAnalysisError(f"Speech quality analysis failed: {e}")

# Singleton instance
_speech_quality_service = None

def get_speech_quality_service() -> SpeechQualityService:
    """Get speech quality service singleton"""
    global _speech_quality_service
    if _speech_quality_service is None:
        _speech_quality_service = SpeechQualityService()
    return _speech_quality_service