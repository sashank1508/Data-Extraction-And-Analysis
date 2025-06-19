"""
Web content extraction module for the Blackcoffer Text Analysis System.
Handles asynchronous web scraping with advanced content extraction techniques.
"""

import asyncio
import aiohttp
import time
from typing import Optional, Dict, List, Set
from urllib.parse import urljoin, urlparse
import re
from bs4 import BeautifulSoup, Comment
from aiohttp import ClientTimeout, ClientSession
from asyncio_throttle import Throttler

from config import settings, http_config, content_selectors
from exceptions import URLExtractionError, ContentParsingError, NetworkError
from logger import get_logger
from utils import async_timing_decorator, sanitize_filename
from models import ArticleContent

logger = get_logger(__name__)


class ContentExtractor:
    """Advanced content extraction with multiple strategies."""
    
    def __init__(self):
        self.selectors = content_selectors
        self.timeout = ClientTimeout(total=settings.REQUEST_TIMEOUT)
        self.throttler = Throttler(rate_limit=settings.MAX_CONCURRENT_REQUESTS, period=60)
    
    async def extract_from_url(self, url: str, url_id: str) -> Optional[ArticleContent]:
        """Extract article content from URL with multiple strategies."""
        async with self.throttler:
            try:
                async with aiohttp.ClientSession(
                    timeout=self.timeout,
                    headers=http_config.HEADERS
                ) as session:
                    return await self._extract_with_session(session, url, url_id)
            except Exception as e:
                logger.error(f"Failed to extract from {url}: {e}")
                return None
    
    @async_timing_decorator
    async def _extract_with_session(self, session: ClientSession, url: str, url_id: str) -> ArticleContent:
        """Extract content using aiohttp session."""
        try:
            async with session.get(url) as response:
                if response.status == 429:  # Rate limited
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited for {url}, waiting {retry_after}s")
                    await asyncio.sleep(retry_after)
                    return await self._extract_with_session(session, url, url_id)
                
                response.raise_for_status()
                
                # Get content with proper encoding
                content = await response.text()
                
                # Parse and extract
                soup = BeautifulSoup(content, 'lxml')
                
                # Extract title and content
                title = self._extract_title(soup)
                article_text = self._extract_article_content(soup)
                
                if not article_text:
                    raise ContentParsingError(f"No article content found for {url}")
                
                # Create article content object
                full_content = f"{title}\n\n{article_text}" if title else article_text
                
                return ArticleContent(
                    url_id=url_id,
                    url=url,
                    title=title,
                    content=full_content,
                    word_count=len(full_content.split()),
                    character_count=len(full_content)
                )
                
        except aiohttp.ClientError as e:
            raise NetworkError(f"Network error for {url}: {e}", url, response.status if 'response' in locals() else None)
        except Exception as e:
            raise URLExtractionError(url, str(e))
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract article title using multiple strategies."""
        for selector in self.selectors.TITLE_SELECTORS:
            try:
                element = soup.select_one(selector)
                if element and element.get_text().strip():
                    title = element.get_text().strip()
                    # Clean title
                    title = re.sub(r'\s+', ' ', title)
                    if len(title) > 10:  # Minimum title length
                        return title
            except Exception:
                continue
        
        return None
    
    def _extract_article_content(self, soup: BeautifulSoup) -> str:
        """Extract article content using multiple strategies."""
        # Remove unwanted elements first
        self._clean_soup(soup)
        
        # Try different extraction strategies
        strategies = [
            self._extract_by_selectors,
            self._extract_by_content_analysis,
            self._extract_by_text_density,
            self._extract_fallback
        ]
        
        for strategy in strategies:
            try:
                content = strategy(soup)
                if content and len(content.strip()) > 100:  # Minimum content length
                    return self._clean_content(content)
            except Exception as e:
                logger.debug(f"Strategy {strategy.__name__} failed: {e}")
                continue
        
        return ""
    
    def _clean_soup(self, soup: BeautifulSoup) -> None:
        """Remove unwanted elements from soup."""
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Remove unwanted tags
        for selector in self.selectors.REMOVE_SELECTORS:
            for element in soup.select(selector):
                element.decompose()
        
        # Remove empty paragraphs
        for p in soup.find_all('p'):
            if not p.get_text().strip():
                p.decompose()
    
    def _extract_by_selectors(self, soup: BeautifulSoup) -> str:
        """Extract content using predefined selectors."""
        for selector in self.selectors.ARTICLE_SELECTORS:
            elements = soup.select(selector)
            if elements:
                content = '\n'.join(elem.get_text() for elem in elements)
                if len(content.strip()) > 100:
                    return content
        return ""
    
    def _extract_by_content_analysis(self, soup: BeautifulSoup) -> str:
        """Extract content by analyzing paragraph density."""
        paragraphs = soup.find_all(['p', 'div'])
        
        # Score paragraphs by text density
        scored_paragraphs = []
        for p in paragraphs:
            text = p.get_text().strip()
            if len(text) > 50:  # Minimum paragraph length
                # Calculate score based on text density and length
                link_density = len(p.find_all('a', href=True)) / max(len(text.split()), 1)
                score = len(text) * (1 - link_density)
                scored_paragraphs.append((score, text))
        
        # Select top paragraphs
        scored_paragraphs.sort(reverse=True)
        top_paragraphs = [text for score, text in scored_paragraphs[:10]]
        
        return '\n'.join(top_paragraphs)
    
    def _extract_by_text_density(self, soup: BeautifulSoup) -> str:
        """Extract content by finding the area with highest text density."""
        body = soup.find('body')
        if not body:
            return ""
        
        # Find the container with most text
        max_text_length = 0
        best_container = None
        
        for container in body.find_all(['div', 'section', 'article', 'main']):
            text_length = len(container.get_text())
            child_containers = len(container.find_all(['div', 'section', 'article']))
            
            # Score based on text length and container complexity
            score = text_length / max(child_containers, 1)
            
            if score > max_text_length:
                max_text_length = score
                best_container = container
        
        if best_container:
            return best_container.get_text()
        
        return ""
    
    def _extract_fallback(self, soup: BeautifulSoup) -> str:
        """Fallback extraction method."""
        body = soup.find('body')
        if body:
            return body.get_text()
        return soup.get_text()
    
    def _clean_content(self, content: str) -> str:
        """Clean extracted content."""
        if not content:
            return ""
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove common unwanted patterns
        patterns_to_remove = [
            r'Subscribe to.*?newsletter',
            r'Follow us on.*?social media',
            r'Related Articles?',
            r'Advertisement',
            r'Cookie Policy',
            r'Privacy Policy',
            r'Terms of Service',
            r'Share this article',
            r'Print this article',
            r'Email this article',
        ]
        
        for pattern in patterns_to_remove:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        # Remove URLs
        content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', content)
        
        # Remove email addresses
        content = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', content)
        
        # Remove extra punctuation
        content = re.sub(r'[^\w\s.,!?;:()"\'-]', ' ', content)
        
        # Final cleanup
        content = re.sub(r'\s+', ' ', content).strip()
        
        return content


class BatchExtractor:
    """Batch processing for multiple URLs with concurrency control."""
    
    def __init__(self, max_concurrent: int = None):
        self.max_concurrent = max_concurrent or settings.MAX_CONCURRENT_REQUESTS
        self.extractor = ContentExtractor()
        self.results = []
        self.errors = []
    
    async def extract_batch(self, urls_data: List[Dict[str, str]], 
                           progress_callback=None) -> List[Optional[ArticleContent]]:
        """Extract content from multiple URLs concurrently."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def extract_single(url_data: Dict[str, str], index: int):
            async with semaphore:
                url_id = url_data['URL_ID']
                url = url_data['URL']
                
                try:
                    # Add delay between requests
                    if index > 0:
                        await asyncio.sleep(settings.REQUEST_DELAY)
                    
                    result = await self.extractor.extract_from_url(url, url_id)
                    
                    if progress_callback:
                        progress_callback(index + 1, url_id, result is not None)
                    
                    return result
                    
                except Exception as e:
                    error_msg = f"Failed to extract {url_id} ({url}): {e}"
                    self.errors.append(error_msg)
                    logger.error(error_msg)
                    
                    if progress_callback:
                        progress_callback(index + 1, url_id, False)
                    
                    return None
        
        # Create tasks for all URLs
        tasks = [
            extract_single(url_data, i) 
            for i, url_data in enumerate(urls_data)
        ]
        
        # Execute with concurrency control
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and None results
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {i} failed with exception: {result}")
                valid_results.append(None)
            else:
                valid_results.append(result)
        
        return valid_results
    
    def get_extraction_stats(self) -> Dict[str, int]:
        """Get extraction statistics."""
        successful = sum(1 for r in self.results if r is not None)
        failed = len(self.results) - successful
        
        return {
            'total': len(self.results),
            'successful': successful,
            'failed': failed,
            'success_rate': (successful / len(self.results) * 100) if self.results else 0
        }


class SmartContentExtractor(ContentExtractor):
    """Enhanced content extractor with ML-like features."""
    
    def __init__(self):
        super().__init__()
        self.content_patterns = self._compile_content_patterns()
    
    def _compile_content_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for content identification."""
        return {
            'article_start': re.compile(r'^(.*?)(PUBLISHED|By |Written by|Author:|Published on)', re.IGNORECASE | re.DOTALL),
            'article_end': re.compile(r'(Related Articles?|More Stories|Comments|Share this|Tags:|Categories:).*$', re.IGNORECASE | re.DOTALL),
            'noise_patterns': re.compile(r'(Advertisement|Subscribe|Newsletter|Follow us|Social media)', re.IGNORECASE),
            'paragraph_boundary': re.compile(r'\n\s*\n'),
        }
    
    def _extract_article_content(self, soup: BeautifulSoup) -> str:
        """Enhanced article extraction with pattern matching."""
        # Use parent class extraction first
        content = super()._extract_article_content(soup)
        
        if not content:
            return ""
        
        # Apply smart filtering
        content = self._apply_smart_filtering(content)
        
        return content
    
    def _apply_smart_filtering(self, content: str) -> str:
        """Apply intelligent content filtering."""
        if not content:
            return ""
        
        # Remove noise patterns
        content = self.content_patterns['noise_patterns'].sub('', content)
        
        # Find article boundaries
        article_start_match = self.content_patterns['article_start'].search(content)
        if article_start_match:
            content = content[article_start_match.end():]
        
        article_end_match = self.content_patterns['article_end'].search(content)
        if article_end_match:
            content = content[:article_end_match.start()]
        
        # Split into paragraphs and filter
        paragraphs = self.content_patterns['paragraph_boundary'].split(content)
        filtered_paragraphs = []
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if self._is_valid_paragraph(paragraph):
                filtered_paragraphs.append(paragraph)
        
        return '\n\n'.join(filtered_paragraphs)
    
    def _is_valid_paragraph(self, paragraph: str) -> bool:
        """Check if paragraph is valid content."""
        if len(paragraph) < 20:  # Too short
            return False
        
        # Check for high link density
        words = paragraph.split()
        if len([w for w in words if w.startswith(('http://', 'https://'))]) > len(words) * 0.3:
            return False
        
        # Check for excessive capitalization
        if sum(1 for c in paragraph if c.isupper()) > len(paragraph) * 0.5:
            return False
        
        return True