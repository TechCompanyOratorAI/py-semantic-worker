"""
Semantic Worker - Main Entry Point

This worker polls messages from AWS SQS analysis queue, performs semantic analysis
comparing transcript segments with slide content and topic information,
then sends results back to Node API via webhook.
"""

import sys
import signal
import time
from typing import Dict, Any

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
            
            # DO NOT delete message - allow retry after visibility timeout
            logger.warning(f"⚠️ Message NOT deleted - will retry after visibility timeout")
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
        logger.info(f"📥 Step 1/4: Loading presentation data...")
        presentation_data = self.database_service.get_presentation_data(presentation_id)
        
        if not presentation_data:
            raise DatabaseError(f"Presentation {presentation_id} not found or incomplete")
        
        logger.info(f"   - Title: {presentation_data.title}")
        logger.info(f"   - Topic: {presentation_data.topic_name}")
        logger.info(f"   - Transcript segments: {len(presentation_data.transcript_segments)}")
        logger.info(f"   - Slides: {len(presentation_data.slides)}")
        
        if not presentation_data.transcript_segments:
            raise SemanticAnalysisError("No transcript segments found for analysis")
        
        # Step 2: Perform semantic analysis
        logger.info(f"🔍 Step 2/4: Performing semantic analysis...")
        
        segment_analyses_obj, overall_scores_obj = self.semantic_service.analyze_presentation(presentation_data)
        
        # Convert to API format
        segment_analyses = []
        for analysis in segment_analyses_obj:
            segment_analyses.append({
                'segmentId': analysis.segment_id,
                'relevanceScore': analysis.relevance_score,
                'semanticScore': analysis.semantic_score,
                'alignmentScore': analysis.alignment_score,
                'issues': analysis.issues,
                'suggestions': analysis.suggestions,
                'topicKeywordsFound': analysis.topic_keywords_found,
                'bestMatchingSlide': analysis.best_matching_slide,
                'expectedSlideNumber': analysis.expected_slide_number,
                'timingDeviation': analysis.timing_deviation
            })
        
        overall_scores = {
            'contentRelevance': overall_scores_obj.content_relevance,
            'semanticSimilarity': overall_scores_obj.semantic_similarity,
            'slideAlignment': overall_scores_obj.slide_alignment,
            'overallScore': overall_scores_obj.overall_score
        }
        
        # Step 3: Format results
        logger.info(f"📦 Step 3/4: Formatting results...")
        
        result_metadata = {
            'embeddingModel': settings.EMBEDDING_MODEL,
            'similarityThreshold': settings.SIMILARITY_THRESHOLD,
            'totalSegments': len(segment_analyses),
            'totalSlides': len(presentation_data.slides),
            'topicName': presentation_data.topic_name,
            'topicDescription': presentation_data.topic_description
        }
        
        logger.info(f"✅ Analysis complete:")
        logger.info(f"   - Analyzed segments: {len(segment_analyses)}")
        logger.info(f"   - Content relevance: {overall_scores['contentRelevance']:.2f}")
        logger.info(f"   - Semantic similarity: {overall_scores['semanticSimilarity']:.2f}")
        logger.info(f"   - Slide alignment: {overall_scores['slideAlignment']:.2f}")
        
        return {
            'segmentAnalyses': segment_analyses,
            'overallScores': overall_scores,
            'metadata': result_metadata
        }
    
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