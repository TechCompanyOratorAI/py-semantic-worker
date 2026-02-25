"""
Custom exceptions for semantic worker
"""

class SemanticWorkerError(Exception):
    """Base exception for semantic worker"""
    pass

class DatabaseError(SemanticWorkerError):
    """Database connection or query error"""
    pass

class SQSError(SemanticWorkerError):
    """SQS polling or message processing error"""
    pass

class WebhookError(SemanticWorkerError):
    """Webhook sending error"""
    pass

class SemanticAnalysisError(SemanticWorkerError):
    """Semantic analysis processing error"""
    pass

class EmbeddingError(SemanticAnalysisError):
    """Embedding generation error"""
    pass

class SimilarityError(SemanticAnalysisError):
    """Similarity calculation error"""
    pass