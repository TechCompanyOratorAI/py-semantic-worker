"""
SQS Service for polling analysis queue
"""

import json
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import boto3
from botocore.exceptions import ClientError, BotoCoreError

from config.settings import settings
from utils.logger import get_logger
from utils.exceptions import SQSError

logger = get_logger(__name__)

@dataclass
class SQSMessage:
    """SQS Message data structure"""
    message_id: str
    receipt_handle: str
    job_id: int
    presentation_id: int
    metadata: Dict[str, Any]
    receive_count: int = 1

    @classmethod
    def from_sqs_message(cls, sqs_message: Dict[str, Any]) -> 'SQSMessage':
        """Create SQSMessage from AWS SQS message"""
        try:
            body = json.loads(sqs_message['Body'])
            attributes = sqs_message.get('Attributes', {})
            receive_count = int(attributes.get('ApproximateReceiveCount', '1'))

            return cls(
                message_id=sqs_message['MessageId'],
                receipt_handle=sqs_message['ReceiptHandle'],
                job_id=body['jobId'],
                presentation_id=body['presentationId'],
                metadata=body.get('metadata', {}),
                receive_count=receive_count
            )
        except (json.JSONDecodeError, KeyError) as e:
            raise SQSError(f"Invalid SQS message format: {e}")

class SQSService:
    """Service for handling SQS operations"""
    
    def __init__(self):
        self.client = None
        self.queue_url = settings.AWS_SQS_SEMANTIC_QUEUE_URL
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize SQS client"""
        try:
            self.client = boto3.client(
                'sqs',
                region_name=settings.AWS_REGION,
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
            )
            logger.info(f"✅ SQS client initialized for region: {settings.AWS_REGION}")
        except Exception as e:
            raise SQSError(f"Failed to initialize SQS client: {e}")
    
    def poll_messages(self, max_messages: int = 1, wait_time_seconds: int = 20) -> List[SQSMessage]:
        """
        Poll messages from SQS queue
        
        Args:
            max_messages: Maximum number of messages to retrieve (1-10)
            wait_time_seconds: Long polling wait time (0-20)
            
        Returns:
            List of SQSMessage objects
        """
        if not self.client:
            raise SQSError("SQS client not initialized")
        
        try:
            # Validate parameters
            max_messages = max(1, min(max_messages, 10))
            wait_time_seconds = max(0, min(wait_time_seconds, 20))
            
            response = self.client.receive_message(
                QueueUrl=self.queue_url,
                MaxNumberOfMessages=max_messages,
                WaitTimeSeconds=wait_time_seconds,
                MessageAttributeNames=['All'],
                AttributeNames=['All']
            )
            
            messages = response.get('Messages', [])
            
            if messages:
                logger.info(f"📥 Received {len(messages)} message(s) from analysis queue")
                return [SQSMessage.from_sqs_message(msg) for msg in messages]
            else:
                logger.debug("No messages received from analysis queue")
                return []
                
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            raise SQSError(f"AWS SQS error [{error_code}]: {error_message}")
        except BotoCoreError as e:
            raise SQSError(f"AWS connection error: {e}")
        except Exception as e:
            raise SQSError(f"Unexpected error polling messages: {e}")
    
    def delete_message(self, message: SQSMessage):
        """
        Delete message from queue after successful processing
        
        Args:
            message: SQSMessage to delete
        """
        if not self.client:
            raise SQSError("SQS client not initialized")
        
        try:
            self.client.delete_message(
                QueueUrl=self.queue_url,
                ReceiptHandle=message.receipt_handle
            )
            logger.info(f"🗑️ Deleted message {message.message_id} from queue")
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            raise SQSError(f"Failed to delete message [{error_code}]: {error_message}")
        except Exception as e:
            raise SQSError(f"Unexpected error deleting message: {e}")
    
    def get_queue_attributes(self) -> Dict[str, Any]:
        """Get queue attributes for monitoring"""
        if not self.client:
            raise SQSError("SQS client not initialized")
        
        try:
            response = self.client.get_queue_attributes(
                QueueUrl=self.queue_url,
                AttributeNames=['All']
            )
            
            attributes = response.get('Attributes', {})
            
            return {
                'approximate_number_of_messages': int(attributes.get('ApproximateNumberOfMessages', 0)),
                'approximate_number_of_messages_not_visible': int(attributes.get('ApproximateNumberOfMessagesNotVisible', 0)),
                'approximate_number_of_messages_delayed': int(attributes.get('ApproximateNumberOfMessagesDelayed', 0)),
                'created_timestamp': attributes.get('CreatedTimestamp'),
                'last_modified_timestamp': attributes.get('LastModifiedTimestamp'),
                'visibility_timeout_seconds': int(attributes.get('VisibilityTimeout', 30)),
                'message_retention_period': int(attributes.get('MessageRetentionPeriod', 1209600))
            }
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            raise SQSError(f"Failed to get queue attributes [{error_code}]: {error_message}")
        except Exception as e:
            raise SQSError(f"Unexpected error getting queue attributes: {e}")

# Singleton instance
_sqs_service = None

def get_sqs_service() -> SQSService:
    """Get SQS service singleton"""
    global _sqs_service
    if _sqs_service is None:
        _sqs_service = SQSService()
    return _sqs_service