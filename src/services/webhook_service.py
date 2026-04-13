"""
Webhook service for sending analysis results back to Node API
"""

import json
import math
import requests
import dataclasses
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin

from config.settings import settings
from utils.logger import get_logger
from utils.exceptions import WebhookError

logger = get_logger(__name__)


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert numpy / non-serializable types to plain Python types
    so json.dumps / requests.post(json=...) never fail with
    'Object of type float32 is not JSON serializable'.
    """
    # Try numpy first (optional dependency – graceful fallback if not installed)
    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.ndarray):
            return sanitize_for_json(obj.tolist())
        if isinstance(obj, np.bool_):
            return bool(obj)
    except ImportError:
        pass

    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj

class WebhookService:
    """Service for sending webhook notifications"""
    
    def __init__(self):
        self.base_url = settings.WEBHOOK_BASE_URL
        self.secret = settings.WEBHOOK_SECRET
        self.timeout = 300  # Increased to 300s for speech quality analysis with large datasets
        self.max_retries = 3
        self.retry_delay = 5  # Base delay in seconds
    
    def _get_headers(self, idempotency_key: str = None) -> Dict[str, str]:
        """Get headers for webhook requests"""
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': f'OratorAI-SemanticWorker/{settings.WORKER_ID}'
        }
        
        if self.secret:
            headers['Authorization'] = f'Bearer {self.secret}'

        if idempotency_key:
            headers['Idempotency-Key'] = idempotency_key
        
        return headers
    
    def _send_webhook(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send webhook request
        
        Args:
            endpoint: Webhook endpoint path
            payload: Request payload
            
        Returns:
            Response data
        """
        # Ensure proper URL joining - remove leading slash from endpoint if base_url has path
        if self.base_url.endswith('/'):
            url = self.base_url + endpoint.lstrip('/')
        else:
            url = self.base_url + endpoint
        idempotency_key = payload.get('_idempotency_key')
        # Remove internal key from payload before sending
        send_payload = {k: v for k, v in payload.items() if k != '_idempotency_key'}
        headers = self._get_headers(idempotency_key=idempotency_key)
        
        try:
            logger.info(f"📤 Sending webhook to {url}")
            # Sanitize payload – converts numpy float32/int64/ndarray → native Python
            send_payload = sanitize_for_json(send_payload)
            logger.debug(f"Payload: {json.dumps(send_payload, indent=2)}")
            
            response = requests.post(
                url,
                json=send_payload,
                headers=headers,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            response_data = response.json()
            logger.info(f"✅ Webhook sent successfully: {response.status_code}")
            
            return response_data
            
        except requests.exceptions.Timeout:
            raise WebhookError(f"Webhook request timed out after {self.timeout}s")
        except requests.exceptions.ConnectionError:
            raise WebhookError(f"Failed to connect to webhook endpoint: {url}")
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            try:
                error_data = e.response.json()
                error_message = error_data.get('message', 'Unknown error')
            except:
                error_message = e.response.text
            raise WebhookError(f"Webhook failed with status {status_code}: {error_message}")
        except requests.exceptions.RequestException as e:
            raise WebhookError(f"Webhook request failed: {e}")
        except json.JSONDecodeError:
            raise WebhookError("Invalid JSON response from webhook endpoint")
        except Exception as e:
            raise WebhookError(f"Unexpected webhook error: {e}")
    
    def send_analysis_complete(
        self,
        job_id: int,
        presentation_id: int,
        segment_analyses: List[Dict[str, Any]],
        overall_scores: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Send analysis complete webhook
        
        Args:
            job_id: Job ID
            presentation_id: Presentation ID
            segment_analyses: List of segment-level analysis results
            overall_scores: Overall presentation scores
            metadata: Additional metadata
        """
        # Convert segment analyses dataclasses to dicts with correct field names
        segment_analyses_dict = []
        for analysis in segment_analyses:
            if dataclasses.is_dataclass(analysis):
                analysis_dict = dataclasses.asdict(analysis)
                # Convert snake_case to camelCase for compatibility with Node API
                converted_dict = {
                    'segmentId': analysis_dict.get('segment_id', 0),
                    'segmentNumber': analysis_dict.get('segment_number', 1),
                    'segmentText': analysis_dict.get('segment_text', ''),
                    'relevanceScore': analysis_dict.get('relevance_score', 0.0),
                    'topicKeywordsFound': analysis_dict.get('topic_keywords_found', []),
                    # BUG FIX: combine issues + off_topic_indicators so node-api gets both
                    'issues': (analysis_dict.get('issues', []) or []) + (analysis_dict.get('off_topic_indicators', []) or []),
                    'semanticScore': analysis_dict.get('semantic_score', 0.0),
                    'bestMatchingSlide': analysis_dict.get('best_matching_slide', 1),
                    'slideSimilarities': analysis_dict.get('slide_similarities', []),
                    'alignmentScore': analysis_dict.get('alignment_score', 0.0),
                    'expectedSlideNumber': analysis_dict.get('expected_slide_number', 1),
                    'timingDeviation': analysis_dict.get('timing_deviation', 0.0),
                    'suggestions': analysis_dict.get('suggestions', []),
                    # Speaker label from diarization (e.g. SPEAKER_00)
                    'speakerLabel': analysis_dict.get('speaker_label') or None,
                }
                
                # Add speech quality data with camelCase keys
                speech_quality_data = analysis_dict.get('speech_quality')
                if speech_quality_data:
                    processed_sq = {
                        'hesitationCount': speech_quality_data.get('hesitation_count', 0),
                        'totalHesitationTime': speech_quality_data.get('total_hesitation_time', 0.0),
                        'hesitationPatterns': []
                    }
                    if 'hesitation_patterns' in speech_quality_data:
                        for pattern in speech_quality_data['hesitation_patterns']:
                            processed_sq['hesitationPatterns'].append({
                                'startTime': pattern.get('startTime'),       # already camelCase from semantic_service
                                'endTime': pattern.get('endTime'),
                                'duration': pattern.get('duration'),
                                'patternType': pattern.get('pattern_type'),  # semantic_service uses snake_case key
                                'confidence': pattern.get('confidence'),
                                'description': pattern.get('description')
                            })
                    converted_dict['speechQuality'] = processed_sq
                else:
                    converted_dict['speechQuality'] = None
                    
                segment_analyses_dict.append(converted_dict)
            else:
                segment_analyses_dict.append(analysis)
        
        # Process overall_scores to handle dataclass conversion
        if dataclasses.is_dataclass(overall_scores):
            overall_scores_dict = dataclasses.asdict(overall_scores)
            # Convert snake_case to camelCase for speech quality fields
            processed_overall_scores = {
                'contentRelevance': overall_scores_dict.get('content_relevance', 0.0),
                'semanticSimilarity': overall_scores_dict.get('semantic_similarity', 0.0),
                'slideAlignment': overall_scores_dict.get('slide_alignment', 0.0),
                'overallScore': overall_scores_dict.get('overall_score', 0.0)
            }
            
            # Add speech quality scores if available
            if overall_scores_dict.get('speech_fluency') is not None:
                processed_overall_scores['speechFluency'] = overall_scores_dict['speech_fluency']
            if overall_scores_dict.get('speech_clarity') is not None:
                processed_overall_scores['speechClarity'] = overall_scores_dict['speech_clarity']
            if overall_scores_dict.get('speech_confidence') is not None:
                processed_overall_scores['speechConfidence'] = overall_scores_dict['speech_confidence']
            if overall_scores_dict.get('speech_overall') is not None:
                processed_overall_scores['speechOverall'] = overall_scores_dict['speech_overall']
        else:
            processed_overall_scores = overall_scores

        payload = {
            'jobId': job_id,
            'presentationId': presentation_id,
            'status': 'success',
            'analysis': {
                'segmentAnalyses': segment_analyses_dict,
                'overallScores': processed_overall_scores,
                'metadata': metadata or {}
            },
            # Internal key used to set Idempotency-Key header (stripped before sending)
            '_idempotency_key': f"{job_id}-{presentation_id}"
        }
        
        return self._send_webhook('/webhooks/analysis-complete', payload)
    
    def send_analysis_failed(
        self,
        job_id: int,
        presentation_id: int,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None
    ):
        """
        Send analysis failed webhook
        
        Args:
            job_id: Job ID
            presentation_id: Presentation ID
            error_message: Error message
            error_details: Additional error details
        """
        payload = {
            'jobId': job_id,
            'presentationId': presentation_id,
            'status': 'failed',
            'error': error_message,
            'errorDetails': error_details or {}
        }
        
        return self._send_webhook('/webhooks/analysis-complete', payload)
    
    def test_connection(self) -> bool:
        """
        Test webhook endpoint connectivity
        
        Returns:
            True if connection successful
        """
        try:
            # Ensure proper URL joining for health check
            if self.base_url.endswith('/'):
                url = self.base_url + 'webhooks/health'
            else:
                url = self.base_url + '/webhooks/health'
            headers = self._get_headers()
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            logger.info("✅ Webhook endpoint is reachable")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Webhook endpoint test failed: {e}")
            return False

# Singleton instance
_webhook_service = None

def get_webhook_service() -> WebhookService:
    """Get webhook service singleton"""
    global _webhook_service
    if _webhook_service is None:
        _webhook_service = WebhookService()
    return _webhook_service