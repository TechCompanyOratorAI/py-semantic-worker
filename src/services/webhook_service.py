"""
Webhook service for sending analysis results back to Node API
"""

import json
import requests
import dataclasses
from typing import Dict, List, Any, Optional
from urllib.parse import urljoin

from config.settings import settings
from utils.logger import get_logger
from utils.exceptions import WebhookError

logger = get_logger(__name__)

class WebhookService:
    """Service for sending webhook notifications"""
    
    def __init__(self):
        self.base_url = settings.WEBHOOK_BASE_URL
        self.secret = settings.WEBHOOK_SECRET
        self.timeout = 30
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for webhook requests"""
        headers = {
            'Content-Type': 'application/json',
            'User-Agent': f'OratorAI-SemanticWorker/{settings.WORKER_ID}'
        }
        
        if self.secret:
            headers['Authorization'] = f'Bearer {self.secret}'
        
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
        headers = self._get_headers()
        
        try:
            logger.info(f"📤 Sending webhook to {url}")
            logger.debug(f"Payload: {json.dumps(payload, indent=2)}")
            
            response = requests.post(
                url,
                json=payload,
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
                    'issues': analysis_dict.get('off_topic_indicators', []),  # Map off_topic_indicators to issues
                    'semanticScore': analysis_dict.get('semantic_score', 0.0),
                    'bestMatchingSlide': analysis_dict.get('best_matching_slide', 1),
                    'slideSimilarities': analysis_dict.get('slide_similarities', []),
                    'alignmentScore': analysis_dict.get('alignment_score', 0.0),
                    'expectedSlideNumber': analysis_dict.get('expected_slide_number', 1),
                    'timingDeviation': analysis_dict.get('timing_deviation', 0.0),
                    'suggestions': analysis_dict.get('suggestions', [])
                }
                segment_analyses_dict.append(converted_dict)
            else:
                segment_analyses_dict.append(analysis)
        
        payload = {
            'jobId': job_id,
            'presentationId': presentation_id,
            'status': 'success',
            'analysis': {
                'segmentAnalyses': segment_analyses_dict,
                'overallScores': overall_scores,
                'metadata': metadata or {}
            }
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