"""
Custom Exceptions
=================
Standardized exceptions for the LLM abuse detection system.
"""


class DetectionError(Exception):
    """Base exception for detection errors"""
    pass


class InvalidInputError(DetectionError):
    """Raised when input validation fails"""
    pass


class PatternNotFoundError(DetectionError):
    """Raised when a pattern is not found in database"""
    pass


class DetectorUnavailableError(DetectionError):
    """Raised when a detector service is unavailable"""
    pass


class ConfigurationError(Exception):
    """Raised when configuration is invalid"""
    pass


class DatabaseError(Exception):
    """Base exception for database errors"""
    pass


class PatternAlreadyExistsError(DatabaseError):
    """Raised when trying to add a pattern that already exists"""
    pass


class RateLimitExceededError(DetectionError):
    """Raised when rate limit is exceeded"""
    pass
