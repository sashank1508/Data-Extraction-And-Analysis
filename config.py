"""
Configuration module for the Blackcoffer Text Analysis System.
Contains all configuration parameters, file paths, and constants.
"""

import os
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

from pydantic import Field


class Settings(BaseSettings):
    """Application settings using Pydantic for validation."""
    
    # File paths
    INPUT_FILE: str = "Input.xlsx"
    OUTPUT_EXCEL: str = "output_results.xlsx"
    OUTPUT_CSV: str = "output_results.csv"
    EXTRACTED_ARTICLES_DIR: str = "extracted_articles"
    LOGS_DIR: str = "logs"
    
    # Processing settings
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT: int = 30
    REQUEST_DELAY: float = 1.0
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 2.0
    
    # Text processing
    MIN_WORD_LENGTH: int = 2
    MAX_ARTICLE_LENGTH: int = 1000000  # 1MB
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@dataclass
class FilePaths:
    """Centralized file path management."""
    
    # Base directories
    BASE_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = BASE_DIR / "data"
    MASTER_DICT_DIR: Path = BASE_DIR / "MasterDictionary"
    STOP_WORDS_DIR: Path = BASE_DIR / "StopWords"
    
    # Dictionary files
    POSITIVE_WORDS: Path = MASTER_DICT_DIR / "positive-words.txt"
    NEGATIVE_WORDS: Path = MASTER_DICT_DIR / "negative-words.txt"
    
    # Stop words files
    STOP_WORDS_FILES: List[Path] = None
    
    def __post_init__(self):
        """Initialize stop words file paths."""
        self.STOP_WORDS_FILES = [
            self.STOP_WORDS_DIR / "StopWords_Auditor.txt",
            self.STOP_WORDS_DIR / "StopWords_Currencies.txt",
            self.STOP_WORDS_DIR / "StopWords_DatesandNumbers.txt",
            self.STOP_WORDS_DIR / "StopWords_GenericLong.txt",
            self.STOP_WORDS_DIR / "StopWords_Generic.txt",
            self.STOP_WORDS_DIR / "StopWords_Geographic.txt",
            self.STOP_WORDS_DIR / "StopWords_Names.txt",
        ]
    
    def create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.DATA_DIR,
            Path(settings.EXTRACTED_ARTICLES_DIR),
            Path(settings.LOGS_DIR)
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class TextAnalysisConstants:
    """Constants for text analysis calculations."""
    
    # Sentiment analysis
    POLARITY_EPSILON: float = 0.000001
    SUBJECTIVITY_EPSILON: float = 0.000001
    
    # Readability
    FOG_INDEX_MULTIPLIER: float = 0.4
    COMPLEX_WORD_SYLLABLE_THRESHOLD: int = 2
    
    # Personal pronouns
    PERSONAL_PRONOUNS: List[str] = None
    
    # Vowels for syllable counting
    VOWELS: str = "aeiouy"
    
    def __post_init__(self):
        """Initialize personal pronouns list."""
        self.PERSONAL_PRONOUNS = [
            'i', 'we', 'my', 'ours', 'us'
        ]


@dataclass
class HTTPConfig:
    """HTTP request configuration."""
    
    HEADERS: Dict[str, str] = None
    TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    BACKOFF_FACTOR: float = 1.0
    
    def __post_init__(self):
        """Initialize HTTP headers."""
        self.HEADERS = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/91.0.4472.124 Safari/537.36'
            ),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }


@dataclass
class ContentSelectors:
    """CSS selectors for extracting article content."""
    
    # Primary article selectors (ordered by priority)
    ARTICLE_SELECTORS: List[str] = None
    
    # Title selectors
    TITLE_SELECTORS: List[str] = None
    
    # Elements to remove
    REMOVE_SELECTORS: List[str] = None
    
    def __post_init__(self):
        """Initialize CSS selectors."""
        self.ARTICLE_SELECTORS = [
            'article',
            '[role="main"]',
            '.post-content',
            '.entry-content',
            '.article-content',
            '.content',
            '.post-body',
            '.article-body',
            '.story-body',
            '.article-text',
            'main',
            '.main-content',
            '.content-body',
            '#content',
            '.page-content'
        ]
        
        self.TITLE_SELECTORS = [
            'h1.entry-title',
            'h1.post-title',
            'h1.article-title',
            'h1',
            '.entry-title',
            '.post-title',
            '.article-title',
            'title'
        ]
        
        self.REMOVE_SELECTORS = [
            'script', 'style', 'nav', 'header', 'footer', 'aside', 
            'form', '.advertisement', '.ads', '.social-share',
            '.comments', '.comment', '.sidebar', '.related-posts',
            '.newsletter', '.popup', '.modal', '.cookie-notice'
        ]


# Initialize global configuration instances
settings = Settings()
file_paths = FilePaths()
text_constants = TextAnalysisConstants()
http_config = HTTPConfig()
content_selectors = ContentSelectors()

# Ensure directories exist
file_paths.create_directories()