"""
S3 Service for downloading audio files

Handles downloading audio files from AWS S3 bucket for speech quality analysis
"""

import os
import boto3
from pathlib import Path
from typing import Optional, Dict, Any
from botocore.exceptions import ClientError, NoCredentialsError

from config.settings import settings
from utils.logger import get_logger
from utils.exceptions import SemanticWorkerError

logger = get_logger(__name__)

class S3Error(SemanticWorkerError):
    """S3 service specific error"""
    pass

class S3Service:
    """Service for handling S3 operations"""
    
    def __init__(self):
        self.client = None
        self._initialize()
    
    def _initialize(self):
        """Initialize S3 client"""
        try:
            logger.info("🔧 Initializing S3 service...")
            
            self.client = boto3.client(
                's3',
                aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
                region_name=settings.AWS_REGION
            )
            
            # Test connection by checking if bucket exists
            if settings.AWS_S3_AUDIO_BUCKET:
                self._test_bucket_access()
            
            logger.info("✅ S3 service initialized successfully")
            
        except NoCredentialsError:
            raise S3Error("AWS credentials not found")
        except ClientError as e:
            raise S3Error(f"Failed to initialize S3 client: {e}")
        except Exception as e:
            raise S3Error(f"Unexpected error initializing S3 service: {e}")
    
    def _test_bucket_access(self):
        """Test if we can access the configured S3 bucket"""
        try:
            self.client.head_bucket(Bucket=settings.AWS_S3_AUDIO_BUCKET)
            logger.info(f"✅ S3 bucket '{settings.AWS_S3_AUDIO_BUCKET}' is accessible")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise S3Error(f"S3 bucket '{settings.AWS_S3_AUDIO_BUCKET}' not found")
            elif error_code == '403':
                raise S3Error(f"Access denied to S3 bucket '{settings.AWS_S3_AUDIO_BUCKET}'")
            else:
                raise S3Error(f"Error accessing S3 bucket: {e}")
    
    def download_audio_file(
        self, 
        presentation_id: int, 
        audio_filename: str,
        local_path: Optional[Path] = None
    ) -> Path:
        """
        Download audio file from S3 bucket
        
        Args:
            presentation_id: Presentation ID for organizing files
            audio_filename: Name of audio file in S3
            local_path: Optional local path, if not provided will use temp directory
            
        Returns:
            Path to downloaded audio file
            
        Raises:
            S3Error: If download fails
        """
        try:
            # Construct S3 key
            s3_key = f"{settings.S3_AUDIO_PREFIX}{audio_filename}"
            
            # Determine local path
            if local_path is None:
                local_path = settings.TEMP_DIR / f"presentation_{presentation_id}" / audio_filename
            
            # Create directory if it doesn't exist
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"📥 Downloading audio file from S3...")
            logger.info(f"   - Bucket: {settings.AWS_S3_AUDIO_BUCKET}")
            logger.info(f"   - Key: {s3_key}")
            logger.info(f"   - Local path: {local_path}")
            
            # Download file
            self.client.download_file(
                Bucket=settings.AWS_S3_AUDIO_BUCKET,
                Key=s3_key,
                Filename=str(local_path)
            )
            
            # Verify file was downloaded
            if not local_path.exists():
                raise S3Error(f"File was not downloaded successfully: {local_path}")
            
            file_size = local_path.stat().st_size
            logger.info(f"✅ Audio file downloaded successfully ({file_size:,} bytes)")
            
            return local_path
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                raise S3Error(f"Audio file not found in S3: {s3_key}")
            elif error_code == 'NoSuchBucket':
                raise S3Error(f"S3 bucket not found: {settings.AWS_S3_AUDIO_BUCKET}")
            else:
                raise S3Error(f"Failed to download audio file: {e}")
        except Exception as e:
            raise S3Error(f"Unexpected error downloading audio file: {e}")
    
    def check_audio_file_exists(self, audio_filename: str) -> bool:
        """
        Check if audio file exists in S3 bucket
        
        Args:
            audio_filename: Name of audio file in S3
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            s3_key = f"{settings.S3_AUDIO_PREFIX}{audio_filename}"
            
            self.client.head_object(
                Bucket=settings.AWS_S3_AUDIO_BUCKET,
                Key=s3_key
            )
            
            return True
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                return False
            else:
                logger.warning(f"⚠️ Error checking if audio file exists: {e}")
                return False
        except Exception as e:
            logger.warning(f"⚠️ Unexpected error checking audio file: {e}")
            return False
    
    def get_audio_file_info(self, audio_filename: str) -> Optional[Dict[str, Any]]:
        """
        Get information about audio file in S3
        
        Args:
            audio_filename: Name of audio file in S3
            
        Returns:
            Dictionary with file info or None if file doesn't exist
        """
        try:
            s3_key = f"{settings.S3_AUDIO_PREFIX}{audio_filename}"
            
            response = self.client.head_object(
                Bucket=settings.AWS_S3_AUDIO_BUCKET,
                Key=s3_key
            )
            
            return {
                'size': response['ContentLength'],
                'last_modified': response['LastModified'],
                'content_type': response.get('ContentType', 'unknown'),
                'etag': response['ETag'].strip('"')
            }
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                return None
            else:
                logger.warning(f"⚠️ Error getting audio file info: {e}")
                return None
        except Exception as e:
            logger.warning(f"⚠️ Unexpected error getting audio file info: {e}")
            return None
    
    def cleanup_local_file(self, file_path: Path):
        """
        Clean up local audio file after processing
        
        Args:
            file_path: Path to local file to delete
        """
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"🗑️ Cleaned up local audio file: {file_path}")
                
                # Also try to remove parent directory if empty
                try:
                    file_path.parent.rmdir()
                    logger.debug(f"🗑️ Removed empty directory: {file_path.parent}")
                except OSError:
                    # Directory not empty, that's fine
                    pass
                    
        except Exception as e:
            logger.warning(f"⚠️ Failed to cleanup local audio file: {e}")

# Singleton instance
_s3_service = None

def get_s3_service() -> S3Service:
    """Get S3 service singleton"""
    global _s3_service
    if _s3_service is None:
        _s3_service = S3Service()
    return _s3_service