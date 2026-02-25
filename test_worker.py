#!/usr/bin/env python3
"""
Test script for semantic worker components
"""

import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from config.settings import settings
from utils.logger import get_logger
from services.database_service import get_database_service
from services.webhook_service import get_webhook_service
from services.semantic_service import get_semantic_service

logger = get_logger(__name__)

def test_configuration():
    """Test configuration loading"""
    logger.info("🔧 Testing configuration...")
    
    try:
        settings.validate()
        logger.info("✅ Configuration is valid")
        
        logger.info(f"   - Worker ID: {settings.WORKER_ID}")
        logger.info(f"   - Embedding Model: {settings.EMBEDDING_MODEL}")
        logger.info(f"   - Database: {settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}")
        logger.info(f"   - SQS Queue: {settings.AWS_SQS_ANALYSIS_QUEUE_URL}")
        logger.info(f"   - Webhook: {settings.WEBHOOK_BASE_URL}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Configuration error: {e}")
        return False

def test_database():
    """Test database connectivity"""
    logger.info("🗄️ Testing database connection...")
    
    try:
        db_service = get_database_service()
        
        # Test basic connectivity
        exists = db_service.check_presentation_exists(1)
        logger.info(f"✅ Database connection successful")
        logger.info(f"   - Test query result: {exists}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
        return False

def test_webhook():
    """Test webhook connectivity"""
    logger.info("📡 Testing webhook connection...")
    
    try:
        webhook_service = get_webhook_service()
        
        # Test connectivity
        is_reachable = webhook_service.test_connection()
        
        if is_reachable:
            logger.info("✅ Webhook endpoint is reachable")
        else:
            logger.warning("⚠️ Webhook endpoint is not reachable")
        
        return True
    except Exception as e:
        logger.error(f"❌ Webhook error: {e}")
        return False

def test_semantic_service():
    """Test semantic analysis service"""
    logger.info("🧠 Testing semantic analysis service...")
    
    try:
        semantic_service = get_semantic_service()
        logger.info("✅ Semantic service initialized successfully")
        
        # Test text preprocessing
        test_text = "Đây là một bài thuyết trình về AI và Machine Learning."
        processed = semantic_service._preprocess_text(test_text)
        logger.info(f"   - Text preprocessing: '{test_text}' → '{processed}'")
        
        # Test keyword extraction
        keywords = semantic_service._extract_keywords(test_text)
        logger.info(f"   - Keywords extracted: {keywords}")
        
        # Test embedding generation
        embeddings = semantic_service._generate_embeddings([test_text])
        logger.info(f"   - Embedding shape: {embeddings.shape}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Semantic service error: {e}")
        return False

def test_full_analysis():
    """Test full analysis with sample data"""
    logger.info("🔍 Testing full analysis with sample presentation...")
    
    try:
        db_service = get_database_service()
        semantic_service = get_semantic_service()
        
        # Try to get a real presentation for testing
        presentation_data = db_service.get_presentation_data(1)  # Adjust ID as needed
        
        if not presentation_data:
            logger.warning("⚠️ No presentation data found for testing")
            return True
        
        logger.info(f"   - Testing with presentation: {presentation_data.title}")
        logger.info(f"   - Topic: {presentation_data.topic_name}")
        logger.info(f"   - Segments: {len(presentation_data.transcript_segments)}")
        logger.info(f"   - Slides: {len(presentation_data.slides)}")
        
        # Perform analysis
        segment_analyses, overall_scores = semantic_service.analyze_presentation(presentation_data)
        
        logger.info("✅ Full analysis completed successfully")
        logger.info(f"   - Analyzed segments: {len(segment_analyses)}")
        logger.info(f"   - Overall scores: {overall_scores}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Full analysis error: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("🧪 OratorAI Semantic Worker Test Suite")
    logger.info("=" * 60)
    
    tests = [
        ("Configuration", test_configuration),
        ("Database", test_database),
        ("Webhook", test_webhook),
        ("Semantic Service", test_semantic_service),
        ("Full Analysis", test_full_analysis)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info("")
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("📊 Test Results Summary")
    logger.info("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    logger.info("")
    logger.info(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Worker is ready to run.")
        return 0
    else:
        logger.error("💥 Some tests failed. Please check configuration and dependencies.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)