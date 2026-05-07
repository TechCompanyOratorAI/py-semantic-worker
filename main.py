"""
Semantic Worker - Main Entry Point

This worker polls messages from AWS SQS analysis queue, performs semantic analysis
comparing transcript segments with slide content and topic information,
then sends results back to Node API via webhook.
"""

import sys
import signal
import time
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, 'src')

from config.settings import settings
from utils.logger import get_logger
from utils.exceptions import (
    SemanticWorkerError,
    DatabaseError,
    SQSError,
    WebhookError,
    SemanticAnalysisError
)

# Import services
from services.sqs_service import get_sqs_service, SQSMessage
from services.database_service import get_database_service
from services.webhook_service import get_webhook_service
from services.semantic_service import get_semantic_service

logger = get_logger(__name__)

class SemanticWorker:
    """Main Semantic Worker class"""
    
    def __init__(self):
        self.running = False
        self.worker_name = f"SemanticWorker-{settings.WORKER_ID}"
        
        # Initialize services
        self.sqs_service = None
        self.database_service = None
        self.webhook_service = None
        self.semantic_service = None
        
        # Statistics
        self.jobs_processed = 0
        self.jobs_succeeded = 0
        self.jobs_failed = 0
    
    def start(self):
        """Start the worker"""
        logger.info("=" * 80)
        logger.info(f"🚀 Starting {self.worker_name}")
        logger.info("=" * 80)
        logger.info(f"📊 Configuration:")
        logger.info(f"   - Embedding Model: {settings.EMBEDDING_MODEL}")
        logger.info(f"   - Enable GPU: {settings.ENABLE_GPU}")
        logger.info(f"   - Embedding Device: {settings.EMBEDDING_DEVICE}")
        logger.info(f"   - Similarity Threshold: {settings.SIMILARITY_THRESHOLD}")
        logger.info(f"   - Poll Interval: {settings.POLL_INTERVAL}s")
        logger.info(f"   - Max Messages: {settings.MAX_MESSAGES}")
        logger.info(f"   - Batch Size: {settings.BATCH_SIZE}")
        logger.info("=" * 80)
        
        # Validate configuration
        try:
            settings.validate()
            logger.info("✅ Configuration validated")
        except ValueError as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            sys.exit(1)
        
        # Initialize services
        self._initialize_services()
        
        self.running = True
        
        try:
            self._run_loop()
        except KeyboardInterrupt:
            logger.info("⚠️ Received keyboard interrupt")
            self.stop()
        except Exception as e:
            logger.error(f"❌ Fatal error: {e}", exc_info=True)
            self.stop()
            sys.exit(1)
    
    def _initialize_services(self):
        """Initialize all services"""
        try:
            logger.info("🔧 Initializing services...")
            
            # SQS Service
            self.sqs_service = get_sqs_service()
            logger.info("   ✅ SQS Service ready")
            
            # Database Service
            self.database_service = get_database_service()
            logger.info("   ✅ Database Service ready")
            
            # Webhook Service
            self.webhook_service = get_webhook_service()
            
            # Test webhook connectivity
            if self.webhook_service.test_connection():
                logger.info("   ✅ Webhook Service ready")
            else:
                logger.warning("   ⚠️ Webhook Service initialized but endpoint not reachable")
            
            # Semantic Analysis Service (lazy load model)
            self.semantic_service = get_semantic_service()
            logger.info("   ✅ Semantic Analysis Service ready")
            
            logger.info("✅ All services initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize services: {e}", exc_info=True)
            raise
    
    def _run_loop(self):
        """Main worker loop - poll and process messages"""
        logger.info("🔄 Worker started, polling for messages...")
        logger.info("")
        
        while self.running:
            try:
                # Poll SQS for messages
                messages = self.sqs_service.poll_messages(
                    max_messages=settings.MAX_MESSAGES,
                    wait_time_seconds=settings.WAIT_TIME_SECONDS
                )
                
                if not messages:
                    # No messages - continue polling
                    continue
                
                # Process each message
                for message in messages:
                    self._process_message(message)
                
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(f"❌ Error in worker loop: {e}", exc_info=True)
                time.sleep(settings.POLL_INTERVAL)
    
    def _process_message(self, message: SQSMessage):
        """
        Process a single SQS message
        
        Args:
            message: SQSMessage object
        """
        job_id = message.job_id
        presentation_id = message.presentation_id
        
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"🎯 Processing Analysis Job {job_id}")
        logger.info("=" * 80)
        logger.info(f"   - Presentation ID: {presentation_id}")
        logger.info(f"   - Message ID: {message.message_id}")
        
        # Track processing time
        start_time = time.time()
        
        try:
            # Process the analysis job
            result = self._process_analysis_job(
                job_id=job_id,
                presentation_id=presentation_id,
                metadata=message.metadata
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            result['metadata']['processingTime'] = round(processing_time, 2)
            
            # Send success webhook
            logger.info(f"📤 Sending success webhook...")
            self.webhook_service.send_analysis_complete(
                job_id=job_id,
                presentation_id=presentation_id,
                segment_analyses=result['segmentAnalyses'],
                overall_scores=result['overallScores'],
                metadata=result['metadata']
            )
            
            # Delete message from queue (success)
            logger.info(f"🗑️ Deleting message from queue...")
            self.sqs_service.delete_message(message)
            
            # Update statistics
            self.jobs_processed += 1
            self.jobs_succeeded += 1
            
            logger.info("=" * 80)
            logger.info(f"✅ Job {job_id} completed successfully in {processing_time:.2f}s")
            logger.info(f"📊 Stats: {self.jobs_succeeded} succeeded, {self.jobs_failed} failed, {self.jobs_processed} total")
            logger.info("=" * 80)
            logger.info("")
            
        except Exception as e:
            # Calculate processing time
            processing_time = time.time() - start_time
            
            logger.error("=" * 80)
            logger.error(f"❌ Job {job_id} failed after {processing_time:.2f}s")
            logger.error(f"❌ Error: {e}", exc_info=True)
            logger.error("=" * 80)
            
            # Send failure webhook
            try:
                logger.warning(f"📤 Sending failure webhook...")
                self.webhook_service.send_analysis_failed(
                    job_id=job_id,
                    presentation_id=presentation_id,
                    error_message=str(e),
                    error_details={
                        'error_type': type(e).__name__,
                        'processing_time': round(processing_time, 2)
                    }
                )
            except Exception as webhook_error:
                logger.error(f"❌ Failed to send failure webhook: {webhook_error}")
            
            # Update statistics
            self.jobs_processed += 1
            self.jobs_failed += 1
            
            # Delete message if max retries reached, otherwise allow retry
            max_retries = 3
            if message.receive_count >= max_retries:
                logger.error(f"🚫 Job {job_id} exceeded max retries ({max_retries}). Discarding message.")
                self.sqs_service.delete_message(message)
            else:
                logger.warning(f"⚠️ Message NOT deleted - will retry (attempt {message.receive_count}/{max_retries})")
            logger.info(f"📊 Stats: {self.jobs_succeeded} succeeded, {self.jobs_failed} failed, {self.jobs_processed} total")
            logger.info("")
    
    def _process_analysis_job(
        self,
        job_id: int,
        presentation_id: int,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process semantic analysis job
        
        Args:
            job_id: Job ID
            presentation_id: Presentation ID
            metadata: Job metadata
            
        Returns:
            Dictionary with analysis results
        """
        # Step 1: Get presentation data
        logger.info(f"📥 Step 1/5: Loading presentation data...")
        presentation_data = self.database_service.get_presentation_data(presentation_id)
        
        if not presentation_data:
            raise DatabaseError(f"Presentation {presentation_id} not found or incomplete")
        
        logger.info(f"   - Title: {presentation_data.title}")
        logger.info(f"   - Topic: {presentation_data.topic_name}")
        logger.info(f"   - Transcript segments: {len(presentation_data.transcript_segments)}")
        logger.info(f"   - Slides: {len(presentation_data.slides)}")
        
        if not presentation_data.transcript_segments:
            raise SemanticAnalysisError("No transcript segments found for analysis")
        
        # Step 2: Download audio file for speech analysis (if enabled)
        audio_file_path = None
        if settings.SPEECH_ANALYSIS_ENABLED:
            try:
                logger.info(f"📥 Step 2/5: Downloading audio file for speech analysis...")
                
                # Import S3 service
                from services.s3_service import get_s3_service
                s3_service = get_s3_service()
                
                # Resolve audio filename with security constraints
                audio_filename = self._resolve_audio_filename_secure(presentation_id, metadata, s3_service)
                
                if audio_filename:
                    # Download audio file with validation
                    audio_file_path = s3_service.download_audio_file(
                        presentation_id=presentation_id,
                        audio_filename=audio_filename
                    )
                    
                    # Validate downloaded file
                    if audio_file_path and audio_file_path.exists():
                        file_size = audio_file_path.stat().st_size
                        if file_size > 100 * 1024 * 1024:  # 100MB limit
                            logger.warning(f"⚠️ Audio file too large: {file_size / 1024 / 1024:.1f}MB")
                            s3_service.cleanup_local_file(audio_file_path)
                            audio_file_path = None
                        else:
                            logger.info(f"✅ Audio file downloaded and validated: {audio_file_path}")
                    else:
                        logger.warning("⚠️ Audio file download failed validation")
                        audio_file_path = None
                else:
                    logger.warning("⚠️ No valid audio file found for this presentation")
                    logger.warning("   Speech quality analysis will be skipped")
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to download audio file: {e}")
                logger.warning("   Speech quality analysis will be skipped")
                audio_file_path = None
        
        # Step 3: Perform comprehensive analysis (semantic + speech quality)
        logger.info(f"🔍 Step 3/5: Performing comprehensive analysis...")
        
        # analyze_presentation returns (List[SegmentAnalysis dataclasses], OverallScores dataclass, SpeechQualityMetrics)
        # We pass the dataclass objects directly to webhook_service which handles conversion
        segment_analyses_obj, overall_scores_obj, speech_quality_metrics = self.semantic_service.analyze_presentation(
            presentation_data, 
            audio_file_path=str(audio_file_path) if audio_file_path else None
        )
        
        # Step 4: Format results
        logger.info(f"📦 Step 4/5: Formatting results...")
        
        result_metadata = {
            'embeddingModel': settings.EMBEDDING_MODEL,
            'similarityThreshold': settings.SIMILARITY_THRESHOLD,
            'totalSegments': len(segment_analyses_obj),
            'totalSlides': len(presentation_data.slides),
            'topicName': presentation_data.topic_name,
            'topicDescription': presentation_data.topic_description,
            'speechAnalysisEnabled': settings.SPEECH_ANALYSIS_ENABLED,
            'speechAnalysisPerformed': audio_file_path is not None,
            'opensmileConfig': settings.OPENSMILE_CONFIG,
            'sampleRate': settings.SPEECH_SAMPLE_RATE,
        }
        
        # Include rich speech metrics so node-api can populate SpeechQualityAnalyses fully
        if speech_quality_metrics:
            result_metadata.update({
                'speakingRate': speech_quality_metrics.speaking_rate,
                'pitchMean': speech_quality_metrics.pitch_mean,
                'pitchStd': speech_quality_metrics.pitch_std,
                'energyMean': speech_quality_metrics.energy_mean,
                'energyStd': speech_quality_metrics.energy_std,
                'pitchVariation': speech_quality_metrics.pitch_variation,
                'volumeVariation': speech_quality_metrics.volume_variation,
                'speechRhythmScore': speech_quality_metrics.speech_rhythm_score,
                'silenceRatio': speech_quality_metrics.silence_ratio,
                'voicedRatio': speech_quality_metrics.voiced_ratio,
                'spectralCentroidMean': speech_quality_metrics.spectral_centroid_mean,
                'mfccFeatures': speech_quality_metrics.mfcc_features if speech_quality_metrics.mfcc_features else None,
            })
        
        logger.info(f"✅ Analysis complete:")
        logger.info(f"   - Analyzed segments: {len(segment_analyses_obj)}")
        logger.info(f"   - Content relevance: {overall_scores_obj.content_relevance:.2f}")
        logger.info(f"   - Semantic similarity: {overall_scores_obj.semantic_similarity:.2f}")
        logger.info(f"   - Slide alignment: {overall_scores_obj.slide_alignment:.2f}")
        if overall_scores_obj.speech_overall is not None:
            logger.info(f"   - Speech quality: {overall_scores_obj.speech_overall:.2f}")
        
        # Step 5: Cleanup audio file
        if audio_file_path and settings.CLEANUP_TEMP_FILES:
            try:
                logger.info(f"🗑️ Step 5/5: Cleaning up audio file...")
                from services.s3_service import get_s3_service
                s3_service = get_s3_service()
                s3_service.cleanup_local_file(audio_file_path)
            except Exception as e:
                logger.warning(f"⚠️ Failed to cleanup audio file: {e}")
        
        # Pass dataclass objects directly; webhook_service handles camelCase conversion
        return {
            'segmentAnalyses': segment_analyses_obj,
            'overallScores': overall_scores_obj,
            'metadata': result_metadata
        }
    
    def _resolve_audio_filename_secure(self, presentation_id: int, metadata: Dict[str, Any], s3_service) -> Optional[str]:
        """
        Securely resolve audio filename with validation constraints
        
        Args:
            presentation_id: Presentation ID
            metadata: Job metadata from SQS
            s3_service: S3 service instance
            
        Returns:
            Validated audio filename or None
        """
        logger.info(f"🔍 Resolving audio file for presentation {presentation_id}")
        logger.info(f"   - Metadata audioFilename: {metadata.get('audioFilename', 'Not provided')}")
        logger.info(f"   - User ID: {metadata.get('userId', 'Not provided')}")
        
        # Priority 1: Explicit filename from metadata (with validation)
        if metadata.get('audioFilename'):
            filename = metadata['audioFilename']
            
            # Security validation
            if self._validate_audio_filename_security(filename, presentation_id):
                if s3_service.check_audio_file_exists(filename):
                    logger.info(f"✅ Using explicit audio filename: {filename}")
                    return filename
                else:
                    logger.warning(f"⚠️ Explicit audio file not found: {filename}")
            else:
                logger.warning(f"⚠️ Audio filename failed security validation: {filename}")
        
        # Priority 2: Standard naming conventions (most secure)
        standard_patterns = [
            f"presentation_{presentation_id}.wav",
            f"presentation_{presentation_id}.mp3",
            f"presentation_{presentation_id}.mp4",
            f"pres_{presentation_id}.wav",
            f"pres_{presentation_id}.mp3",
            f"pres_{presentation_id}.mp4"
        ]
        
        for pattern in standard_patterns:
            if s3_service.check_audio_file_exists(pattern):
                logger.info(f"✅ Using standard pattern: {pattern}")
                return pattern
        
        # Priority 3: User-specific patterns (if user info available)
        if metadata.get('userId'):
            user_id = metadata['userId']
            user_patterns = [
                f"user_{user_id}_presentation_{presentation_id}.wav",
                f"user_{user_id}_presentation_{presentation_id}.mp3",
                f"user_{user_id}_presentation_{presentation_id}.mp4",
                f"u{user_id}_p{presentation_id}.wav",
                f"u{user_id}_p{presentation_id}.mp3",
                f"u{user_id}_p{presentation_id}.mp4"
            ]
            
            for pattern in user_patterns:
                if s3_service.check_audio_file_exists(pattern):
                    logger.info(f"✅ Using user-specific pattern: {pattern}")
                    return pattern
        
        logger.warning(f"❌ No valid audio file found for presentation {presentation_id}")
        return None
    
    def _validate_audio_filename_security(self, filename: str, presentation_id: int) -> bool:
        """
        Validate audio filename for security constraints
        
        Args:
            filename: Audio filename to validate
            presentation_id: Expected presentation ID
            
        Returns:
            True if filename passes security validation
        """
        # Check for path traversal attempts (only block '..', allow '/' since S3 keys can be paths)
        if '..' in filename:
            logger.warning(f"⚠️ Path traversal attempt detected: {filename}")
            return False
        
        # Must contain presentation ID for security
        if str(presentation_id) not in filename:
            logger.warning(f"⚠️ Filename doesn't contain presentation ID {presentation_id}: {filename}")
            return False
        
        # Must have valid audio extension
        valid_extensions = ['.wav', '.mp3', '.mp4', '.m4a', '.flac', '.aac']
        if not any(filename.lower().endswith(ext) for ext in valid_extensions):
            logger.warning(f"⚠️ Invalid audio file extension: {filename}")
            return False
        
        # Check filename length (prevent extremely long filenames)
        if len(filename) > 255:
            logger.warning(f"⚠️ Filename too long: {len(filename)} characters")
            return False
        
        # Check for suspicious characters
        suspicious_chars = ['<', '>', '|', ':', '*', '?', '"', '\\']
        if any(char in filename for char in suspicious_chars):
            logger.warning(f"⚠️ Suspicious characters in filename: {filename}")
            return False
        
        return True
    
    def stop(self):
        """Stop the worker gracefully"""
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"🛑 Stopping {self.worker_name}...")
        logger.info(f"📊 Final Stats:")
        logger.info(f"   - Jobs succeeded: {self.jobs_succeeded}")
        logger.info(f"   - Jobs failed: {self.jobs_failed}")
        logger.info(f"   - Jobs total: {self.jobs_processed}")
        logger.info("=" * 80)
        self.running = False
        
        # Close database connection
        if self.database_service:
            self.database_service.close()

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"⚠️ Received signal {signum}")
    sys.exit(0)

def main():
    """Main entry point"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and start worker
    worker = SemanticWorker()
    worker.start()

if __name__ == "__main__":
    main()
