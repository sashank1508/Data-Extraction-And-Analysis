"""
Text analysis module for the Blackcoffer Text Analysis System.
Performs comprehensive sentiment and readability analysis.
"""

import re
import time
from typing import List, Set, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

from config import text_constants, file_paths, settings
from exceptions import TextAnalysisError
from logger import get_logger
from utils import FileManager, TextProcessor, timing_decorator, performance_monitor
from models import TextAnalysisResult, ArticleContent

logger = get_logger(__name__)

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)


@dataclass
class TextMetrics:
    """Container for basic text metrics."""
    
    total_words: int = 0
    total_sentences: int = 0
    total_syllables: int = 0
    complex_words: int = 0
    personal_pronouns: int = 0
    positive_words: int = 0
    negative_words: int = 0
    avg_word_length: float = 0.0  # Changed from property to regular field
    
    @property
    def avg_sentence_length(self) -> float:
        """Calculate average sentence length."""
        return self.total_words / max(self.total_sentences, 1)
    
    @property
    def percentage_complex_words(self) -> float:
        """Calculate percentage of complex words."""
        return (self.complex_words / max(self.total_words, 1)) * 100
    
    @property
    def fog_index(self) -> float:
        """Calculate Fog readability index."""
        return text_constants.FOG_INDEX_MULTIPLIER * (
            self.avg_sentence_length + self.percentage_complex_words
        )
    
    @property
    def syllables_per_word(self) -> float:
        """Calculate average syllables per word."""
        return self.total_syllables / max(self.total_words, 1)


class SentimentAnalyzer:
    """Handles sentiment analysis using word dictionaries."""
    
    def __init__(self):
        self.positive_words = self._load_sentiment_words('positive')
        self.negative_words = self._load_sentiment_words('negative')
        logger.info(f"Loaded {len(self.positive_words)} positive and {len(self.negative_words)} negative words")
    
    def _load_sentiment_words(self, sentiment_type: str) -> Set[str]:
        """Load sentiment words from files."""
        if sentiment_type == 'positive':
            filepath = file_paths.POSITIVE_WORDS
        elif sentiment_type == 'negative':
            filepath = file_paths.NEGATIVE_WORDS
        else:
            raise ValueError(f"Invalid sentiment type: {sentiment_type}")
        
        return FileManager.load_word_list(filepath)
    
    def analyze_sentiment(self, words: List[str]) -> Dict[str, int]:
        """Analyze sentiment of word list."""
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        return {
            'positive_score': positive_count,
            'negative_score': negative_count
        }
    
    def calculate_polarity(self, positive_score: int, negative_score: int) -> float:
        """Calculate polarity score."""
        denominator = positive_score + negative_score + text_constants.POLARITY_EPSILON
        return (positive_score - negative_score) / denominator
    
    def calculate_subjectivity(self, positive_score: int, negative_score: int, total_words: int) -> float:
        """Calculate subjectivity score."""
        denominator = total_words + text_constants.SUBJECTIVITY_EPSILON
        return (positive_score + negative_score) / denominator


class ReadabilityAnalyzer:
    """Handles readability and complexity analysis."""
    
    def __init__(self):
        self.syllable_cache = {}  # Cache for syllable counts
        self.cache_lock = threading.Lock()
    
    def count_syllables(self, word: str) -> int:
        """Count syllables in a word with caching."""
        if not word:
            return 0
        
        word_lower = word.lower().strip()
        
        # Check cache first
        with self.cache_lock:
            if word_lower in self.syllable_cache:
                return self.syllable_cache[word_lower]
        
        # Calculate syllables
        syllable_count = self._calculate_syllables(word_lower)
        
        # Cache result
        with self.cache_lock:
            self.syllable_cache[word_lower] = syllable_count
        
        return syllable_count
    
    def _calculate_syllables(self, word: str) -> int:
        """Calculate syllables using advanced heuristics."""
        vowels = text_constants.VOWELS
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
        if (word.endswith('le') and len(word) > 2 and 
            word[-3] not in vowels):
            syllable_count += 1
        
        # Handle 'ed' ending
        if (word.endswith('ed') and len(word) > 2 and 
            word[-3] not in vowels):
            syllable_count -= 1
        
        # Handle 'es' ending
        if (word.endswith('es') and len(word) > 2 and 
            word[-3] in 'sxz'):
            syllable_count += 1
        
        # Minimum of 1 syllable
        return max(1, syllable_count)
    
    def is_complex_word(self, word: str) -> bool:
        """Check if word is complex (more than 2 syllables)."""
        return self.count_syllables(word) > text_constants.COMPLEX_WORD_SYLLABLE_THRESHOLD
    
    def analyze_word_complexity(self, words: List[str]) -> Dict[str, Any]:
        """Analyze word complexity metrics."""
        if not words:
            return {
                'complex_word_count': 0,
                'total_syllables': 0,
                'avg_word_length': 0.0
            }
        
        complex_count = 0
        total_syllables = 0
        total_length = 0
        
        for word in words:
            if self.is_complex_word(word):
                complex_count += 1
            total_syllables += self.count_syllables(word)
            total_length += len(word)
        
        return {
            'complex_word_count': complex_count,
            'total_syllables': total_syllables,
            'avg_word_length': total_length / len(words)
        }


class PersonalPronounAnalyzer:
    """Handles personal pronoun analysis."""
    
    def __init__(self):
        self.pronouns = set(text_constants.PERSONAL_PRONOUNS)
        self.pronoun_pattern = self._compile_pronoun_pattern()
    
    def _compile_pronoun_pattern(self) -> re.Pattern:
        """Compile regex pattern for personal pronouns."""
        # Create word boundary pattern for pronouns
        pronoun_alternatives = '|'.join(re.escape(p) for p in self.pronouns)
        pattern = rf'\b({pronoun_alternatives})\b'
        return re.compile(pattern, re.IGNORECASE)
    
    def count_personal_pronouns(self, text: str) -> int:
        """Count personal pronouns in text using regex."""
        if not text:
            return 0
        
        matches = self.pronoun_pattern.findall(text.lower())
        return len(matches)


class TextAnalyzer:
    """Main text analyzer class that coordinates all analysis components."""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.readability_analyzer = ReadabilityAnalyzer()
        self.pronoun_analyzer = PersonalPronounAnalyzer()
        self.stop_words = FileManager.load_stop_words()
        logger.info(f"Text analyzer initialized with {len(self.stop_words)} stop words")
    
    @timing_decorator
    def analyze(self, article: ArticleContent) -> TextAnalysisResult:
        """Perform complete text analysis on article content."""
        try:
            with performance_monitor(f"Text analysis for {article.url_id}"):
                start_time = time.time()
                
                # Preprocess text
                cleaned_text = self._preprocess_text(article.content)
                
                # Extract and filter words
                words = self._extract_words(cleaned_text)
                filtered_words = self._filter_words(words)
                
                # Extract sentences
                sentences = self._extract_sentences(article.content)
                
                # Perform analysis
                metrics = self._calculate_metrics(
                    article.content, filtered_words, sentences
                )
                
                # Calculate derived scores
                sentiment_scores = self.sentiment_analyzer.analyze_sentiment(filtered_words)
                polarity = self.sentiment_analyzer.calculate_polarity(
                    sentiment_scores['positive_score'],
                    sentiment_scores['negative_score']
                )
                subjectivity = self.sentiment_analyzer.calculate_subjectivity(
                    sentiment_scores['positive_score'],
                    sentiment_scores['negative_score'],
                    len(filtered_words)
                )
                
                processing_time = time.time() - start_time
                
                # Create result
                return TextAnalysisResult(
                    url_id=article.url_id,
                    url=article.url,
                    positive_score=sentiment_scores['positive_score'],
                    negative_score=sentiment_scores['negative_score'],
                    polarity_score=round(polarity, 2),
                    subjectivity_score=round(subjectivity, 2),
                    avg_sentence_length=round(metrics.avg_sentence_length, 2),
                    percentage_complex_words=round(metrics.percentage_complex_words, 2),
                    fog_index=round(metrics.fog_index, 2),
                    avg_words_per_sentence=round(metrics.avg_sentence_length, 2),
                    complex_word_count=metrics.complex_words,
                    word_count=metrics.total_words,
                    syllable_per_word=round(metrics.syllables_per_word, 2),
                    personal_pronouns=metrics.personal_pronouns,
                    avg_word_length=round(metrics.avg_word_length, 2),
                    processing_time_seconds=processing_time
                )
                
        except Exception as e:
            logger.error(f"Text analysis failed for {article.url_id}: {e}", exc_info=True)
            return TextAnalysisResult.create_default(article.url_id, article.url)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for analysis."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove non-ASCII characters that might interfere
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        
        return text.strip()
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract words from text using NLTK."""
        try:
            words = word_tokenize(text.lower())
            # Filter out punctuation and numbers
            words = [word for word in words if word.isalpha() and len(word) >= 2]
            return words
        except Exception as e:
            logger.warning(f"NLTK tokenization failed, falling back to simple split: {e}")
            return TextProcessor.extract_words(text)
    
    def _filter_words(self, words: List[str]) -> List[str]:
        """Filter out stop words."""
        return [word for word in words if word not in self.stop_words]
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text using NLTK."""
        try:
            sentences = sent_tokenize(text)
            # Filter out very short sentences
            sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
            return sentences
        except Exception as e:
            logger.warning(f"NLTK sentence tokenization failed, falling back to regex: {e}")
            return TextProcessor.extract_sentences(text)
    
    def _calculate_metrics(self, original_text: str, filtered_words: List[str], 
                          sentences: List[str]) -> TextMetrics:
        """Calculate comprehensive text metrics."""
        # Basic counts
        total_words = len(filtered_words)
        total_sentences = len(sentences)
        
        # Word complexity analysis
        complexity_metrics = self.readability_analyzer.analyze_word_complexity(filtered_words)
        
        # Personal pronouns (use original text to preserve case)
        personal_pronouns = self.pronoun_analyzer.count_personal_pronouns(original_text)
        
        # Create metrics object
        metrics = TextMetrics(
            total_words=total_words,
            total_sentences=total_sentences,
            total_syllables=complexity_metrics['total_syllables'],
            complex_words=complexity_metrics['complex_word_count'],
            personal_pronouns=personal_pronouns
        )
        
        # Set average word length
        metrics.avg_word_length = complexity_metrics['avg_word_length']
        
        return metrics


class BatchTextAnalyzer:
    """Handles batch processing of text analysis."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(4, (settings.MAX_CONCURRENT_REQUESTS // 2))
        self.analyzer = TextAnalyzer()
    
    def analyze_batch(self, articles: List[ArticleContent], 
                     progress_callback=None) -> List[TextAnalysisResult]:
        """Analyze multiple articles using thread pool."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_article = {
                executor.submit(self.analyzer.analyze, article): article 
                for article in articles if article is not None
            }
            
            # Process completed tasks
            for i, future in enumerate(as_completed(future_to_article)):
                article = future_to_article[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    if progress_callback:
                        progress_callback(i + 1, article.url_id, True)
                        
                except Exception as e:
                    logger.error(f"Analysis failed for {article.url_id}: {e}")
                    # Create default result for failed analysis
                    default_result = TextAnalysisResult.create_default(
                        article.url_id, article.url
                    )
                    results.append(default_result)
                    
                    if progress_callback:
                        progress_callback(i + 1, article.url_id, False)
        
        # Sort results by original order (URL_ID)
        article_order = {article.url_id: i for i, article in enumerate(articles) if article}
        results.sort(key=lambda x: article_order.get(x.url_id, float('inf')))
        
        return results


class AdvancedTextAnalyzer(TextAnalyzer):
    """Enhanced text analyzer with additional features."""
    
    def __init__(self):
        super().__init__()
        self.text_patterns = self._compile_text_patterns()
    
    def _compile_text_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for advanced text analysis."""
        return {
            'sentence_endings': re.compile(r'[.!?]+\s+'),
            'word_boundaries': re.compile(r'\b\w+\b'),
            'repeated_chars': re.compile(r'(.)\1{2,}'),
            'all_caps_words': re.compile(r'\b[A-Z]{2,}\b'),
            'numbers': re.compile(r'\b\d+\b'),
            'urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing."""
        text = super()._preprocess_text(text)
        
        # Remove URLs
        text = self.text_patterns['urls'].sub(' ', text)
        
        # Normalize repeated characters
        text = self.text_patterns['repeated_chars'].sub(r'\1', text)
        
        # Handle all-caps words (convert to lowercase)
        text = self.text_patterns['all_caps_words'].sub(
            lambda m: m.group(0).lower(), text
        )
        
        return text
    
    def analyze(self, article: ArticleContent) -> TextAnalysisResult:
        """Enhanced analysis with additional validation."""
        # Validate article content
        if not article.content or len(article.content.strip()) < 50:
            logger.warning(f"Article {article.url_id} has insufficient content")
            return TextAnalysisResult.create_default(article.url_id, article.url)
        
        # Check for content length limits
        if len(article.content) > settings.MAX_ARTICLE_LENGTH:
            logger.warning(f"Article {article.url_id} exceeds maximum length, truncating")
            article.content = article.content[:settings.MAX_ARTICLE_LENGTH]
        
        return super().analyze(article)