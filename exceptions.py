"""
Custom exceptions for the Blackcoffer Text Analysis System.
Provides specific error types for better error handling and debugging.
"""

from typing import Optional


class BlackcofferAnalysisError(Exception):
    """Base exception class for all Blackcoffer analysis errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class DataExtractionError(BlackcofferAnalysisError):
    """Exception raised when data extraction fails."""
    pass


class URLExtractionError(DataExtractionError):
    """Exception raised when URL content extraction fails."""
    
    def __init__(self, url: str, message: str, status_code: Optional[int] = None):
        self.url = url
        self.status_code = status_code
        super().__init__(f"Failed to extract from {url}: {message}")


class ContentParsingError(DataExtractionError):
    """Exception raised when content parsing fails."""
    pass


class TextAnalysisError(BlackcofferAnalysisError):
    """Exception raised when text analysis fails."""
    pass


class FileOperationError(BlackcofferAnalysisError):
    """Exception raised when file operations fail."""
    
    def __init__(self, filepath: str, operation: str, message: str):
        self.filepath = filepath
        self.operation = operation
        super().__init__(f"Failed to {operation} file {filepath}: {message}")


class ConfigurationError(BlackcofferAnalysisError):
    """Exception raised when configuration is invalid."""
    pass


class ValidationError(BlackcofferAnalysisError):
    """Exception raised when data validation fails."""
    pass


class NetworkError(BlackcofferAnalysisError):
    """Exception raised when network operations fail."""
    
    def __init__(self, message: str, url: Optional[str] = None, status_code: Optional[int] = None):
        self.url = url
        self.status_code = status_code
        super().__init__(message)


class RateLimitError(NetworkError):
    """Exception raised when rate limit is exceeded."""
    
    def __init__(self, url: str, retry_after: Optional[int] = None):
        self.retry_after = retry_after
        message = f"Rate limit exceeded for {url}"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        super().__init__(message, url)