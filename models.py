"""
Data models for the Blackcoffer Text Analysis System.
Uses Pydantic for data validation and serialization.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator, HttpUrl
from datetime import datetime


class URLInput(BaseModel):
    """Model for input URL data."""
    
    url_id: str = Field(..., description="Unique identifier for the URL")
    url: HttpUrl = Field(..., description="URL to extract content from")
    
    @validator('url_id')
    def validate_url_id(cls, v):
        if not v or not v.strip():
            raise ValueError('URL_ID cannot be empty')
        return v.strip()


class ArticleContent(BaseModel):
    """Model for extracted article content."""
    
    url_id: str
    url: str
    title: Optional[str] = None
    content: str
    extracted_at: datetime = Field(default_factory=datetime.now)
    word_count: int = Field(ge=0, description="Raw word count")
    character_count: int = Field(ge=0, description="Character count")
    
    @validator('content')
    def validate_content(cls, v):
        if len(v.strip()) == 0:
            raise ValueError('Content cannot be empty')
        return v


class TextAnalysisResult(BaseModel):
    """Model for text analysis results."""
    
    # Input fields
    url_id: str
    url: str
    
    # Sentiment scores
    positive_score: int = Field(ge=0, description="Count of positive words")
    negative_score: int = Field(ge=0, description="Count of negative words")
    polarity_score: float = Field(ge=-1, le=1, description="Polarity score between -1 and 1")
    subjectivity_score: float = Field(ge=0, le=1, description="Subjectivity score between 0 and 1")
    
    # Readability metrics
    avg_sentence_length: float = Field(ge=0, description="Average words per sentence")
    percentage_complex_words: float = Field(ge=0, le=100, description="Percentage of complex words")
    fog_index: float = Field(ge=0, description="Fog readability index")
    avg_words_per_sentence: float = Field(ge=0, description="Average words per sentence")
    
    # Word analysis
    complex_word_count: int = Field(ge=0, description="Count of complex words")
    word_count: int = Field(ge=0, description="Total word count after cleaning")
    syllable_per_word: float = Field(ge=0, description="Average syllables per word")
    personal_pronouns: int = Field(ge=0, description="Count of personal pronouns")
    avg_word_length: float = Field(ge=0, description="Average character length of words")
    
    # Metadata
    analysis_timestamp: datetime = Field(default_factory=datetime.now)
    processing_time_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with proper column names for output."""
        return {
            'URL_ID': self.url_id,
            'URL': self.url,
            'POSITIVE SCORE': self.positive_score,
            'NEGATIVE SCORE': self.negative_score,
            'POLARITY SCORE': self.polarity_score,
            'SUBJECTIVITY SCORE': self.subjectivity_score,
            'AVG SENTENCE LENGTH': self.avg_sentence_length,
            'PERCENTAGE OF COMPLEX WORDS': self.percentage_complex_words,
            'FOG INDEX': self.fog_index,
            'AVG NUMBER OF WORDS PER SENTENCE': self.avg_words_per_sentence,
            'COMPLEX WORD COUNT': self.complex_word_count,
            'WORD COUNT': self.word_count,
            'SYLLABLE PER WORD': self.syllable_per_word,
            'PERSONAL PRONOUNS': self.personal_pronouns,
            'AVG WORD LENGTH': self.avg_word_length
        }
    
    @classmethod
    def create_default(cls, url_id: str, url: str):
        """Create default result for failed analysis."""
        return cls(
            url_id=url_id,
            url=url,
            positive_score=0,
            negative_score=0,
            polarity_score=0.0,
            subjectivity_score=0.0,
            avg_sentence_length=0.0,
            percentage_complex_words=0.0,
            fog_index=0.0,
            avg_words_per_sentence=0.0,
            complex_word_count=0,
            word_count=0,
            syllable_per_word=0.0,
            personal_pronouns=0,
            avg_word_length=0.0
        )


class ProcessingStats(BaseModel):
    """Model for processing statistics."""
    
    total_urls: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    successful_analyses: int = 0
    failed_analyses: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    
    @property
    def extraction_success_rate(self) -> float:
        """Calculate extraction success rate."""
        if self.total_urls == 0:
            return 0.0
        return (self.successful_extractions / self.total_urls) * 100
    
    @property
    def analysis_success_rate(self) -> float:
        """Calculate analysis success rate."""
        if self.total_urls == 0:
            return 0.0
        return (self.successful_analyses / self.total_urls) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'total_urls': self.total_urls,
            'successful_extractions': self.successful_extractions,
            'failed_extractions': self.failed_extractions,
            'successful_analyses': self.successful_analyses,
            'failed_analyses': self.failed_analyses,
            'extraction_success_rate': round(self.extraction_success_rate, 2),
            'analysis_success_rate': round(self.analysis_success_rate, 2),
            'total_processing_time': round(self.total_processing_time, 2),
            'average_processing_time': round(self.average_processing_time, 2)
        }


class BatchProcessingResult(BaseModel):
    """Model for batch processing results."""
    
    results: list[TextAnalysisResult]
    stats: ProcessingStats
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    
    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)
    
    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)