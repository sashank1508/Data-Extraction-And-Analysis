#!/usr/bin/env python3
"""
Blackcoffer Text Analysis System - Simplified Production Version
Works with basic dependencies and provides all required functionality.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional, Any
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    INPUT_FILE = "Input.xlsx"
    OUTPUT_EXCEL = "output_results.xlsx"
    OUTPUT_CSV = "output_results.csv" 
    EXTRACTED_ARTICLES_DIR = "extracted_articles"
    
    MAX_CONCURRENT_REQUESTS = 8
    REQUEST_TIMEOUT = 30
    REQUEST_DELAY = 1.0
    MAX_RETRIES = 3
    
    COMPLEX_WORD_SYLLABLE_THRESHOLD = 2
    FOG_INDEX_MULTIPLIER = 0.4
    POLARITY_EPSILON = 0.000001
    SUBJECTIVITY_EPSILON = 0.000001

config = Config()

class FileManager:
    """Handles all file operations."""
    
    @staticmethod
    def load_word_list(filepath: str) -> Set[str]:
        """Load word list from file."""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                words = {line.strip().lower() for line in f if line.strip()}
            logger.info(f"Loaded {len(words)} words from {filepath}")
            return words
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return set()
    
    @staticmethod
    def load_all_stop_words() -> Set[str]:
        """Load all stop words."""
        stop_words = set()
        stop_word_files = [
            'StopWords/StopWords_Auditor.txt',
            'StopWords/StopWords_Currencies.txt', 
            'StopWords/StopWords_DatesandNumbers.txt',
            'StopWords/StopWords_GenericLong.txt',
            'StopWords/StopWords_Generic.txt',
            'StopWords/StopWords_Geographic.txt',
            'StopWords/StopWords_Names.txt'
        ]
        
        for file_path in stop_word_files:
            if os.path.exists(file_path):
                words = FileManager.load_word_list(file_path)
                stop_words.update(words)
        
        logger.info(f"Loaded {len(stop_words)} total stop words")
        return stop_words
    
    @staticmethod
    def save_extracted_article(url_id: str, content: str):
        """Save extracted article."""
        os.makedirs(config.EXTRACTED_ARTICLES_DIR, exist_ok=True)
        filepath = os.path.join(config.EXTRACTED_ARTICLES_DIR, f"{url_id}.txt")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

class TextProcessor:
    """Text processing utilities."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text."""
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    @staticmethod
    def extract_words(text: str) -> List[str]:
        """Extract words from text."""
        if not text:
            return []
        text = re.sub(r'[^\w\s]', ' ', text)
        words = text.lower().split()
        return [word for word in words if word.isalpha() and len(word) >= 2]
    
    @staticmethod
    def extract_sentences(text: str) -> List[str]:
        """Extract sentences."""
        if not text:
            return []
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def count_syllables(word: str) -> int:
        """Count syllables in word."""
        if not word:
            return 0
        
        word = word.lower().strip()
        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    @staticmethod
    def is_complex_word(word: str) -> bool:
        """Check if word is complex."""
        return TextProcessor.count_syllables(word) > config.COMPLEX_WORD_SYLLABLE_THRESHOLD
    
    @staticmethod
    def count_personal_pronouns(text: str) -> int:
        """Count personal pronouns."""
        pronouns = ['i', 'we', 'my', 'ours', 'us']
        words = TextProcessor.extract_words(text.lower())
        return sum(1 for word in words if word in pronouns)

class ContentExtractor:
    """Extracts content from URLs."""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def extract_article_text(self, url: str) -> str:
        """Extract article text from URL."""
        try:
            response = requests.get(url, headers=self.headers, timeout=config.REQUEST_TIMEOUT)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form']):
                element.decompose()
            
            # Try different selectors
            selectors = [
                'article', '.post-content', '.entry-content', 
                '.article-content', '.content', '.post-body',
                '.article-body', 'main', '.main-content'
            ]
            
            title = ""
            article_text = ""
            
            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
            
            # Try selectors for content
            for selector in selectors:
                content = soup.select_one(selector)
                if content:
                    article_text = content.get_text()
                    break
            
            if not article_text:
                body = soup.find('body')
                if body:
                    article_text = body.get_text()
            
            # Clean text
            article_text = TextProcessor.clean_text(article_text)
            full_text = f"{title}\n\n{article_text}" if title else article_text
            
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting from {url}: {e}")
            return ""

class TextAnalyzer:
    """Performs text analysis."""
    
    def __init__(self):
        self.positive_words = FileManager.load_word_list('MasterDictionary/positive-words.txt')
        self.negative_words = FileManager.load_word_list('MasterDictionary/negative-words.txt')
        self.stop_words = FileManager.load_all_stop_words()
        logger.info("TextAnalyzer initialized")
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Perform complete text analysis."""
        if not text:
            return self._get_default_scores()
        
        # Extract and clean words
        words = TextProcessor.extract_words(text)
        filtered_words = [word for word in words if word not in self.stop_words]
        
        # Extract sentences
        sentences = TextProcessor.extract_sentences(text)
        
        if not filtered_words:
            return self._get_default_scores()
        
        # Basic metrics
        word_count = len(filtered_words)
        sentence_count = max(len(sentences), 1)
        
        # Sentiment analysis
        positive_score = sum(1 for word in filtered_words if word in self.positive_words)
        negative_score = sum(1 for word in filtered_words if word in self.negative_words)
        
        # Polarity and subjectivity
        polarity_score = ((positive_score - negative_score) / 
                         (positive_score + negative_score + config.POLARITY_EPSILON))
        subjectivity_score = ((positive_score + negative_score) / 
                             (word_count + config.SUBJECTIVITY_EPSILON))
        
        # Sentence and word analysis
        avg_sentence_length = word_count / sentence_count
        
        # Complex words
        complex_words = [word for word in filtered_words if TextProcessor.is_complex_word(word)]
        complex_word_count = len(complex_words)
        percentage_complex_words = (complex_word_count / word_count) * 100
        
        # Fog index
        fog_index = config.FOG_INDEX_MULTIPLIER * (avg_sentence_length + percentage_complex_words)
        
        # Syllables and word metrics
        total_syllables = sum(TextProcessor.count_syllables(word) for word in filtered_words)
        syllable_per_word = total_syllables / word_count
        
        # Personal pronouns
        personal_pronouns = TextProcessor.count_personal_pronouns(text)
        
        # Average word length
        avg_word_length = sum(len(word) for word in filtered_words) / word_count
        
        return {
            'POSITIVE SCORE': positive_score,
            'NEGATIVE SCORE': negative_score,
            'POLARITY SCORE': round(polarity_score, 2),
            'SUBJECTIVITY SCORE': round(subjectivity_score, 2),
            'AVG SENTENCE LENGTH': round(avg_sentence_length, 2),
            'PERCENTAGE OF COMPLEX WORDS': round(percentage_complex_words, 2),
            'FOG INDEX': round(fog_index, 2),
            'AVG NUMBER OF WORDS PER SENTENCE': round(avg_sentence_length, 2),
            'COMPLEX WORD COUNT': complex_word_count,
            'WORD COUNT': word_count,
            'SYLLABLE PER WORD': round(syllable_per_word, 2),
            'PERSONAL PRONOUNS': personal_pronouns,
            'AVG WORD LENGTH': round(avg_word_length, 2)
        }
    
    def _get_default_scores(self) -> Dict[str, Any]:
        """Default scores for failed analysis."""
        return {
            'POSITIVE SCORE': 0,
            'NEGATIVE SCORE': 0,
            'POLARITY SCORE': 0.0,
            'SUBJECTIVITY SCORE': 0.0,
            'AVG SENTENCE LENGTH': 0.0,
            'PERCENTAGE OF COMPLEX WORDS': 0.0,
            'FOG INDEX': 0.0,
            'AVG NUMBER OF WORDS PER SENTENCE': 0.0,
            'COMPLEX WORD COUNT': 0,
            'WORD COUNT': 0,
            'SYLLABLE PER WORD': 0.0,
            'PERSONAL PRONOUNS': 0,
            'AVG WORD LENGTH': 0.0
        }

def process_single_url(extractor: ContentExtractor, analyzer: TextAnalyzer, 
                      url_data: Dict[str, str], index: int, total: int) -> Dict[str, Any]:
    """Process a single URL."""
    url_id = url_data['URL_ID']
    url = url_data['URL']
    
    logger.info(f"Processing [{index+1}/{total}] {url_id}: {url}")
    
    # Extract content
    article_text = extractor.extract_article_text(url)
    
    # Save extracted text
    if article_text:
        FileManager.save_extracted_article(url_id, article_text)
    
    # Analyze text
    analysis_results = analyzer.analyze_text(article_text)
    
    # Create result
    result = {
        'URL_ID': url_id,
        'URL': url,
        **analysis_results
    }
    
    success = analysis_results['WORD COUNT'] > 0
    status = "✅" if success else "❌"
    logger.info(f"{status} Completed {url_id} - Words: {analysis_results['WORD COUNT']}")
    
    # Add delay
    time.sleep(config.REQUEST_DELAY)
    
    return result

def main():
    """Main function."""
    start_time = time.time()
    
    logger.info("🚀 Starting Blackcoffer Text Analysis System")
    
    try:
        # Load input data
        logger.info("📂 Loading input data...")
        df = pd.read_excel(config.INPUT_FILE)
        
        if 'URL_ID' not in df.columns or 'URL' not in df.columns:
            raise ValueError("Input file must contain 'URL_ID' and 'URL' columns")
        
        urls_data = df.to_dict('records')
        total_urls = len(urls_data)
        
        logger.info(f"📊 Loaded {total_urls} URLs for processing")
        
        # Initialize components
        logger.info("🔧 Initializing analysis components...")
        extractor = ContentExtractor()
        analyzer = TextAnalyzer()
        
        # Process URLs with threading
        logger.info("🌐 Starting content extraction and analysis...")
        results = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all tasks
            future_to_url = {
                executor.submit(process_single_url, extractor, analyzer, url_data, i, total_urls): url_data
                for i, url_data in enumerate(urls_data)
            }
            
            # Collect results
            for future in as_completed(future_to_url):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    url_data = future_to_url[future]
                    logger.error(f"❌ Failed to process {url_data['URL_ID']}: {e}")
                    # Add default result
                    default_result = {
                        'URL_ID': url_data['URL_ID'],
                        'URL': url_data['URL'],
                        **analyzer._get_default_scores()
                    }
                    results.append(default_result)
        
        # Sort results by original order
        url_order = {url_data['URL_ID']: i for i, url_data in enumerate(urls_data)}
        results.sort(key=lambda x: url_order.get(x['URL_ID'], float('inf')))
        
        # Save results
        logger.info("💾 Saving results...")
        results_df = pd.DataFrame(results)
        results_df.to_excel(config.OUTPUT_EXCEL, index=False)
        results_df.to_csv(config.OUTPUT_CSV, index=False)
        
        # Calculate statistics
        successful = sum(1 for r in results if r['WORD COUNT'] > 0)
        total_time = time.time() - start_time
        
        # Print summary
        print("\n" + "="*60)
        print("🎉 BLACKCOFFER ANALYSIS COMPLETED!")
        print("="*60)
        print(f"📊 Total URLs processed: {total_urls}")
        print(f"✅ Successful extractions: {successful}")
        print(f"❌ Failed extractions: {total_urls - successful}")
        print(f"📈 Success rate: {(successful/total_urls)*100:.1f}%")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"⚡ Average time per URL: {total_time/total_urls:.2f}s")
        print(f"\n📁 Results saved to:")
        print(f"   • {config.OUTPUT_EXCEL}")
        print(f"   • {config.OUTPUT_CSV}")
        print(f"   • {config.EXTRACTED_ARTICLES_DIR}/")
        print("="*60)
        
        logger.info("✨ Analysis completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("⏹️ Analysis interrupted by user")
    except Exception as e:
        logger.error(f"💥 Analysis failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()