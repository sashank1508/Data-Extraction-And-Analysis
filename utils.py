"""
Utility functions for the Blackcoffer Text Analysis System.
Contains helper functions for file operations, text processing, and validation.
"""

import re
import time
import asyncio
from pathlib import Path
from typing import Set, List, Optional, Dict, Any, Union
from functools import wraps
from contextlib import contextmanager

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import settings, file_paths
from exceptions import FileOperationError, ValidationError
from logger import get_logger

logger = get_logger(__name__)


def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    return wrapper


def async_timing_decorator(func):
    """Decorator to measure async function execution time."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"{func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    return wrapper


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(FileOperationError)
)
def safe_file_read(filepath: Union[str, Path], encoding: str = 'utf-8') -> str:
    """Safely read file with retry mechanism."""
    try:
        with open(filepath, 'r', encoding=encoding, errors='ignore') as f:
            return f.read()
    except Exception as e:
        raise FileOperationError(str(filepath), "read", str(e))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(FileOperationError)
)
def safe_file_write(filepath: Union[str, Path], content: str, encoding: str = 'utf-8') -> None:
    """Safely write file with retry mechanism."""
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding=encoding) as f:
            f.write(content)
    except Exception as e:
        raise FileOperationError(str(filepath), "write", str(e))


class FileManager:
    """Utility class for file operations."""
    
    @staticmethod
    @timing_decorator
    def load_word_list(filepath: Union[str, Path]) -> Set[str]:
        """Load word list from file and return as set."""
        try:
            content = safe_file_read(filepath)
            words = {word.strip().lower() for word in content.splitlines() if word.strip()}
            logger.debug(f"Loaded {len(words)} words from {filepath}")
            return words
        except Exception as e:
            logger.error(f"Failed to load word list from {filepath}: {e}")
            return set()
    
    @staticmethod
    @timing_decorator
    def load_stop_words() -> Set[str]:
        """Load all stop words from multiple files."""
        all_stop_words = set()
        
        for filepath in file_paths.STOP_WORDS_FILES:
            if filepath.exists():
                words = FileManager.load_word_list(filepath)
                all_stop_words.update(words)
            else:
                logger.warning(f"Stop words file not found: {filepath}")
        
        logger.info(f"Loaded {len(all_stop_words)} total stop words")
        return all_stop_words
    
    @staticmethod
    @timing_decorator
    def save_extracted_article(url_id: str, content: str) -> None:
        """Save extracted article content to file."""
        filepath = Path(settings.EXTRACTED_ARTICLES_DIR) / f"{url_id}.txt"
        safe_file_write(filepath, content)
        logger.debug(f"Saved article {url_id} to {filepath}")
    
    @staticmethod
    @timing_decorator
    def load_input_data(filepath: str = None) -> pd.DataFrame:
        """Load input data from Excel file."""
        if filepath is None:
            filepath = settings.INPUT_FILE
        
        try:
            df = pd.read_excel(filepath)
            if 'URL_ID' not in df.columns or 'URL' not in df.columns:
                raise ValidationError("Input file must contain 'URL_ID' and 'URL' columns")
            
            # Clean and validate data
            df = df.dropna(subset=['URL_ID', 'URL'])
            df['URL_ID'] = df['URL_ID'].astype(str).str.strip()
            df['URL'] = df['URL'].astype(str).str.strip()
            
            logger.info(f"Loaded {len(df)} URLs from {filepath}")
            return df
            
        except Exception as e:
            raise FileOperationError(filepath, "read", str(e))
    
    @staticmethod
    @timing_decorator
    def save_results(results: List[Dict[str, Any]], 
                    excel_path: str = None, 
                    csv_path: str = None) -> None:
        """Save results to Excel and CSV files."""
        if excel_path is None:
            excel_path = settings.OUTPUT_EXCEL
        if csv_path is None:
            csv_path = settings.OUTPUT_CSV
        
        try:
            df = pd.DataFrame(results)
            
            # Save to Excel
            df.to_excel(excel_path, index=False)
            logger.info(f"Results saved to {excel_path}")
            
            # Save to CSV
            df.to_csv(csv_path, index=False)
            logger.info(f"Results saved to {csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise FileOperationError("output files", "write", str(e))


class TextProcessor:
    """Utility class for text processing operations."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text by removing extra whitespace and normalizing."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-printable characters
        text = re.sub(r'[^\x20-\x7E\n\r\t]', '', text)
        
        return text.strip()
    
    @staticmethod
    def extract_words(text: str, remove_punctuation: bool = True) -> List[str]:
        """Extract words from text."""
        if not text:
            return []
        
        if remove_punctuation:
            # Remove punctuation and split
            text = re.sub(r'[^\w\s]', ' ', text)
        
        words = text.lower().split()
        
        # Filter words by minimum length
        words = [word for word in words if len(word) >= settings.MIN_WORD_LENGTH]
        
        return words
    
    @staticmethod
    def count_syllables(word: str) -> int:
        """Count syllables in a word using advanced heuristics."""
        if not word:
            return 0
        
        word = word.lower().strip()
        if not word:
            return 0
        
        # Handle common patterns
        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False
        
        # Count vowel groups
        for i, char in enumerate(word):
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle special cases
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        # Handle 'le' ending
        if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
            syllable_count += 1
        
        # Minimum of 1 syllable
        return max(1, syllable_count)
    
    @staticmethod
    def is_complex_word(word: str, threshold: int = None) -> bool:
        """Check if word is complex based on syllable count."""
        if threshold is None:
            threshold = 2  # Complex words have more than 2 syllables
        
        return TextProcessor.count_syllables(word) > threshold
    
    @staticmethod
    def extract_sentences(text: str) -> List[str]:
        """Extract sentences from text using regex patterns."""
        if not text:
            return []
        
        # Split on sentence endings
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    @staticmethod
    def count_personal_pronouns(text: str, pronouns: List[str] = None) -> int:
        """Count personal pronouns in text."""
        if pronouns is None:
            from config import text_constants
            pronouns = text_constants.PERSONAL_PRONOUNS
        
        if not text:
            return 0
        
        words = TextProcessor.extract_words(text.lower())
        count = sum(1 for word in words if word in pronouns)
        
        return count


class URLValidator:
    """Utility class for URL validation and processing."""
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Check if URL is valid."""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)',  # path
            re.IGNORECASE
        )
        
        return url_pattern.match(url) is not None
    
    @staticmethod
    def normalize_url(url: str) -> str:
        """Normalize URL format."""
        url = url.strip()
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        return url


class DataValidator:
    """Utility class for data validation."""
    
    @staticmethod
    def validate_input_data(df: pd.DataFrame) -> List[str]:
        """Validate input DataFrame and return list of issues."""
        issues = []
        
        # Check required columns
        required_columns = ['URL_ID', 'URL']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        
        # Check for empty values
        if df['URL_ID'].isna().any():
            issues.append("Found empty URL_ID values")
        
        if df['URL'].isna().any():
            issues.append("Found empty URL values")
        
        # Check for duplicate URL_IDs
        duplicates = df[df['URL_ID'].duplicated()]
        if not duplicates.empty:
            issues.append(f"Found duplicate URL_IDs: {duplicates['URL_ID'].tolist()}")
        
        # Validate URLs
        invalid_urls = []
        for idx, url in df['URL'].items():
            if not URLValidator.is_valid_url(str(url)):
                invalid_urls.append(f"Row {idx}: {url}")
        
        if invalid_urls:
            issues.append(f"Invalid URLs found: {invalid_urls[:5]}")  # Show first 5
        
        return issues


@contextmanager
def performance_monitor(operation_name: str):
    """Context manager for monitoring performance."""
    start_time = time.time()
    start_memory = get_memory_usage() if hasattr(__builtins__, 'get_memory_usage') else 0
    
    try:
        logger.info(f"Starting {operation_name}")
        yield
    finally:
        end_time = time.time()
        end_memory = get_memory_usage() if hasattr(__builtins__, 'get_memory_usage') else 0
        
        duration = end_time - start_time
        memory_delta = end_memory - start_memory
        
        logger.info(f"Completed {operation_name} in {duration:.2f}s")
        if memory_delta:
            logger.info(f"Memory usage change: {memory_delta:.2f}MB")


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def batch_process(items: List[Any], batch_size: int = 10):
    """Generator to process items in batches."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def create_progress_callback(total: int, logger_instance = None):
    """Create a progress callback function."""
    if logger_instance is None:
        logger_instance = logger
    
    def callback(current: int, item_id: str = "", success: bool = True):
        percentage = (current / total) * 100
        status = "✅" if success else "❌"
        logger_instance.info(f"{status} [{current:3d}/{total}] ({percentage:5.1f}%) {item_id}")
    
    return callback


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters."""
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Limit length
    if len(filename) > 200:
        filename = filename[:200]
    
    return filename


def chunks(lst: List[Any], n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retry with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed")
            
            raise last_exception
        return wrapper
    return decorator