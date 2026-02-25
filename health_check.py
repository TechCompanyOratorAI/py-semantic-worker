#!/usr/bin/env python3
"""
Health check script for semantic worker
"""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

try:
    from services.database_service import get_database_service
    from services.sqs_service import get_sqs_service
    from config.settings import settings
    
    # Test database connection
    db_service = get_database_service()
    db_service.check_presentation_exists(1)
    
    # Test SQS connection
    sqs_service = get_sqs_service()
    sqs_service.get_queue_attributes()
    
    print("✅ Health check passed")
    sys.exit(0)
    
except Exception as e:
    print(f"❌ Health check failed: {e}")
    sys.exit(1)