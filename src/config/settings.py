"""
Semantic Worker Configuration Settings

Loads configuration from environment variables with defaults
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def _env_bool(name, default=False):
    """Read a boolean environment variable."""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}

class Settings:
    """Configuration settings for semantic worker"""
    
    # Worker Configuration
    WORKER_ID = os.getenv('WORKER_ID', 'semantic-worker-1')
    POLL_INTERVAL = int(os.getenv('POLL_INTERVAL', '5'))
    MAX_MESSAGES = int(os.getenv('MAX_MESSAGES', '1'))
    WAIT_TIME_SECONDS = int(os.getenv('WAIT_TIME_SECONDS', '20'))
    
    # AWS Configuration
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    AWS_REGION = os.getenv('AWS_REGION', 'ap-southeast-1')
    AWS_SQS_SEMANTIC_QUEUE_URL = os.getenv('AWS_SQS_SEMANTIC_QUEUE_URL')
    
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL')
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = int(os.getenv('DB_PORT', '5432'))
    DB_NAME = os.getenv('DB_NAME', 'oratorai')
    DB_USER = os.getenv('DB_USER')
    DB_PASSWORD = os.getenv('DB_PASSWORD')
    
    # Webhook Configuration
    WEBHOOK_BASE_URL = os.getenv('WEBHOOK_BASE_URL', 'http://localhost:3000')
    WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET')
    
    # Semantic Analysis Configuration
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    ENABLE_GPU = _env_bool('ENABLE_GPU', False)
    _EMBEDDING_DEVICE = os.getenv('EMBEDDING_DEVICE')
    EMBEDDING_DEVICE = (
        _EMBEDDING_DEVICE.strip().lower()
        if _EMBEDDING_DEVICE and _EMBEDDING_DEVICE.strip()
        else ('cuda' if ENABLE_GPU else 'cpu')
    )
    SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))
    
    # Speech Quality Analysis Configuration
    SPEECH_ANALYSIS_ENABLED = True  # Forced to always be enabled
    OPENSMILE_CONFIG = os.getenv('OPENSMILE_CONFIG', 'eGeMAPSv02')  # Default feature set
    SPEECH_SAMPLE_RATE = int(os.getenv('SPEECH_SAMPLE_RATE', '16000'))
    SPEECH_CHUNK_DURATION = float(os.getenv('SPEECH_CHUNK_DURATION', '3.0'))  # seconds
    HESITATION_THRESHOLD = float(os.getenv('HESITATION_THRESHOLD', '0.3'))  # silence ratio threshold
    
    # S3 Configuration for Audio Files
    AWS_S3_AUDIO_BUCKET = os.getenv('AWS_S3_AUDIO_BUCKET')
    S3_AUDIO_PREFIX = os.getenv('S3_AUDIO_PREFIX', 'audio/')
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = os.getenv('LOG_FORMAT', 'colored')
    
    # File System
    TEMP_DIR = Path(os.getenv('TEMP_DIR', '/tmp/semantic-worker'))
    CLEANUP_TEMP_FILES = os.getenv('CLEANUP_TEMP_FILES', 'true').lower() == 'true'
    
    @classmethod
    def validate(cls):
        """Validate required settings"""
        required_settings = [
            'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY', 
            'AWS_SQS_SEMANTIC_QUEUE_URL',
            'DB_USER',
            'DB_PASSWORD'
        ]
        
        # Add S3 bucket requirement if speech analysis is enabled
        if cls.SPEECH_ANALYSIS_ENABLED:
            required_settings.append('AWS_S3_AUDIO_BUCKET')
        
        missing = []
        for setting in required_settings:
            if not getattr(cls, setting):
                missing.append(setting)
        
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        valid_embedding_devices = {'auto', 'cpu', 'cuda', 'gpu'}
        if cls.EMBEDDING_DEVICE not in valid_embedding_devices:
            raise ValueError(
                "EMBEDDING_DEVICE must be one of: auto, cpu, cuda, gpu"
            )
        
        # Create temp directory if it doesn't exist
        cls.TEMP_DIR.mkdir(parents=True, exist_ok=True)
        
        return True

# Create settings instance
settings = Settings()
