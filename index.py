import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import string
import time
from urllib.parse import urljoin, urlparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextAnalyzer:
    def __init__(self):
        self.positive_words = self.load_word_list('MasterDictionary/positive-words.txt')
        self.negative_words = self.load_word_list('MasterDictionary/negative-words.txt')
        self.stop_words = self.load_all_stop_words()
        
    def load_word_list(self, filepath):
        """Load word list from file"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                words = [line.strip().lower() for line in f if line.strip()]
            return set(words)
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")
            return set()
    
    def load_all_stop_words(self):
        """Load all stop words from StopWords directory"""
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
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    words = [line.strip().lower() for line in f if line.strip()]
                    stop_words.update(words)
            except Exception as e:
                logger.warning(f"Could not load {file_path}: {e}")
        
        return stop_words
    
    def extract_article_text(self, url):
        """Extract article text from URL"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form']):
                element.decompose()
            
            # Try to find article content using common selectors
            article_selectors = [
                'article',
                '.post-content',
                '.entry-content',
                '.article-content',
                '.content',
                '.post-body',
                '.article-body',
                'main',
                '.main-content'
            ]
            
            article_text = ""
            title = ""
            
            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
            
            # Try different selectors to find article content
            for selector in article_selectors:
                content = soup.select_one(selector)
                if content:
                    article_text = content.get_text()
                    break
            
            # If no specific article content found, use body
            if not article_text:
                body = soup.find('body')
                if body:
                    article_text = body.get_text()
            
            # Clean the text
            article_text = re.sub(r'\s+', ' ', article_text).strip()
            
            # Combine title and article text
            full_text = f"{title}\n\n{article_text}" if title else article_text
            
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting text from {url}: {e}")
            return ""
    
    def clean_text(self, text):
        """Clean text by removing punctuation and converting to lowercase"""
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()
    
    def count_syllables(self, word):
        """Count syllables in a word"""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        prev_char_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_char_was_vowel:
                    syllable_count += 1
                prev_char_was_vowel = True
            else:
                prev_char_was_vowel = False
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        # Ensure at least one syllable
        if syllable_count == 0:
            syllable_count = 1
            
        return syllable_count
    
    def is_complex_word(self, word):
        """Check if word is complex (more than 2 syllables)"""
        return self.count_syllables(word) > 2
    
    def count_personal_pronouns(self, text):
        """Count personal pronouns in text"""
        personal_pronouns = [
            'i', 'we', 'my', 'ours', 'us'
        ]
        
        words = word_tokenize(text.lower())
        count = 0
        
        for word in words:
            if word in personal_pronouns:
                count += 1
        
        return count
    
    def analyze_text(self, text):
        """Perform complete text analysis"""
        if not text:
            return self.get_default_scores()
        
        # Clean text for word analysis
        cleaned_text = self.clean_text(text)
        words = word_tokenize(cleaned_text)
        
        # Remove stop words
        filtered_words = [word for word in words if word not in self.stop_words and word.isalpha()]
        
        # Sentence tokenization
        sentences = sent_tokenize(text)
        
        # Calculate basic metrics
        word_count = len(filtered_words)
        sentence_count = len(sentences)
        
        if word_count == 0:
            return self.get_default_scores()
        
        # Sentiment Analysis
        positive_score = sum(1 for word in filtered_words if word in self.positive_words)
        negative_score = sum(1 for word in filtered_words if word in self.negative_words)
        
        # Polarity and Subjectivity
        polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
        subjectivity_score = (positive_score + negative_score) / (word_count + 0.000001)
        
        # Average sentence length
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Complex words
        complex_words = [word for word in filtered_words if self.is_complex_word(word)]
        complex_word_count = len(complex_words)
        percentage_complex_words = (complex_word_count / word_count) * 100 if word_count > 0 else 0
        
        # Fog Index
        fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
        
        # Syllables
        total_syllables = sum(self.count_syllables(word) for word in filtered_words)
        syllable_per_word = total_syllables / word_count if word_count > 0 else 0
        
        # Personal pronouns
        personal_pronouns = self.count_personal_pronouns(text)
        
        # Average word length
        avg_word_length = sum(len(word) for word in filtered_words) / word_count if word_count > 0 else 0
        
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
    
    def get_default_scores(self):
        """Return default scores when text analysis fails"""
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

def main():
    """Main function to run the analysis"""
    logger.info("Starting text analysis...")
    
    # Initialize analyzer
    analyzer = TextAnalyzer()
    
    # Read input file
    try:
        input_df = pd.read_excel('Input.xlsx')
        logger.info(f"Loaded {len(input_df)} URLs from Input.xlsx")
    except Exception as e:
        logger.error(f"Error reading Input.xlsx: {e}")
        return
    
    # Create output directory for text files
    os.makedirs('extracted_articles', exist_ok=True)
    
    # Process each URL
    results = []
    
    for index, row in input_df.iterrows():
        url_id = row['URL_ID']
        url = row['URL']
        
        logger.info(f"Processing {index + 1}/{len(input_df)}: {url_id}")
        
        # Extract article text
        article_text = analyzer.extract_article_text(url)
        
        if article_text:
            # Save extracted text
            try:
                with open(f'extracted_articles/{url_id}.txt', 'w', encoding='utf-8') as f:
                    f.write(article_text)
                logger.info(f"Saved article text for {url_id}")
            except Exception as e:
                logger.error(f"Error saving text for {url_id}: {e}")
        
        # Perform analysis
        analysis_results = analyzer.analyze_text(article_text)
        
        # Create result row
        result_row = {
            'URL_ID': url_id,
            'URL': url,
            **analysis_results
        }
        
        results.append(result_row)
        
        # Add delay to be respectful to servers
        time.sleep(1)
    
    # Create output DataFrame
    output_df = pd.DataFrame(results)
    
    # Save results
    try:
        output_df.to_excel('output_results.xlsx', index=False)
        output_df.to_csv('output_results.csv', index=False)
        logger.info("Results saved to output_results.xlsx and output_results.csv")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
    
    # Print summary
    logger.info("Analysis completed!")
    logger.info(f"Total URLs processed: {len(results)}")
    logger.info(f"Successful extractions: {sum(1 for r in results if r['WORD COUNT'] > 0)}")

if __name__ == "__main__":
    main()