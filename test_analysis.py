"""
Test suite for the Blackcoffer Text Analysis System.
Production-ready tests with comprehensive coverage.
"""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import tempfile
import pandas as pd

from config import settings
from models import URLInput, ArticleContent, TextAnalysisResult
from analyzer import TextAnalyzer, SentimentAnalyzer, ReadabilityAnalyzer
from extractor import ContentExtractor
from utils import TextProcessor, FileManager


class TestTextProcessor:
    """Test cases for TextProcessor utility class."""
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        dirty_text = "  This   is    a   test  \n\n  with   extra   spaces  "
        expected = "This is a test with extra spaces"
        assert TextProcessor.clean_text(dirty_text) == expected
    
    def test_extract_words(self):
        """Test word extraction."""
        text = "Hello, World! This is a test."
        words = TextProcessor.extract_words(text)
        assert "hello" in words
        assert "world" in words
        assert "test" in words
        assert "," not in words
        assert "!" not in words
    
    def test_count_syllables(self):
        """Test syllable counting."""
        test_cases = [
            ("hello", 2),
            ("world", 1),
            ("beautiful", 3),
            ("a", 1),
            ("", 0),
            ("strength", 1),
            ("simple", 2)
        ]
        
        for word, expected in test_cases:
            assert TextProcessor.count_syllables(word) == expected
    
    def test_is_complex_word(self):
        """Test complex word identification."""
        assert TextProcessor.is_complex_word("beautiful")  # 3 syllables
        assert TextProcessor.is_complex_word("incredible")  # 4 syllables
        assert not TextProcessor.is_complex_word("hello")  # 2 syllables
        assert not TextProcessor.is_complex_word("cat")  # 1 syllable
    
    def test_count_personal_pronouns(self):
        """Test personal pronoun counting."""
        text = "I think we should go. My opinion is that ours is better."
        count = TextProcessor.count_personal_pronouns(text)
        assert count == 4  # I, we, my, ours


class TestSentimentAnalyzer:
    """Test cases for SentimentAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a sentiment analyzer for testing."""
        analyzer = SentimentAnalyzer()
        # Mock word lists for testing
        analyzer.positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful'}
        analyzer.negative_words = {'bad', 'terrible', 'awful', 'horrible', 'worst'}
        return analyzer
    
    def test_analyze_sentiment_positive(self, analyzer):
        """Test positive sentiment analysis."""
        words = ['good', 'great', 'excellent', 'test', 'hello']
        result = analyzer.analyze_sentiment(words)
        assert result['positive_score'] == 3
        assert result['negative_score'] == 0
    
    def test_analyze_sentiment_negative(self, analyzer):
        """Test negative sentiment analysis."""
        words = ['bad', 'terrible', 'awful', 'test', 'hello']
        result = analyzer.analyze_sentiment(words)
        assert result['positive_score'] == 0
        assert result['negative_score'] == 3
    
    def test_analyze_sentiment_mixed(self, analyzer):
        """Test mixed sentiment analysis."""
        words = ['good', 'bad', 'great', 'terrible', 'neutral']
        result = analyzer.analyze_sentiment(words)
        assert result['positive_score'] == 2
        assert result['negative_score'] == 2
    
    def test_calculate_polarity(self, analyzer):
        """Test polarity calculation."""
        # Positive polarity
        polarity = analyzer.calculate_polarity(3, 1)
        assert polarity > 0
        
        # Negative polarity
        polarity = analyzer.calculate_polarity(1, 3)
        assert polarity < 0
        
        # Neutral polarity
        polarity = analyzer.calculate_polarity(2, 2)
        assert abs(polarity) < 0.1  # Should be close to 0
    
    def test_calculate_subjectivity(self, analyzer):
        """Test subjectivity calculation."""
        subjectivity = analyzer.calculate_subjectivity(2, 1, 10)
        assert 0 <= subjectivity <= 1
        
        # High subjectivity
        subjectivity = analyzer.calculate_subjectivity(5, 5, 10)
        assert subjectivity == 1.0
        
        # Low subjectivity
        subjectivity = analyzer.calculate_subjectivity(1, 1, 100)
        assert subjectivity < 0.1


class TestReadabilityAnalyzer:
    """Test cases for ReadabilityAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a readability analyzer for testing."""
        return ReadabilityAnalyzer()
    
    def test_syllable_caching(self, analyzer):
        """Test syllable count caching."""
        word = "beautiful"
        
        # First call
        count1 = analyzer.count_syllables(word)
        
        # Second call should use cache
        count2 = analyzer.count_syllables(word)
        
        assert count1 == count2
        assert word.lower() in analyzer.syllable_cache
    
    def test_complex_word_analysis(self, analyzer):
        """Test word complexity analysis."""
        words = ["cat", "dog", "beautiful", "incredible", "amazing"]
        result = analyzer.analyze_word_complexity(words)
        
        assert result['complex_word_count'] >= 2  # beautiful, incredible, amazing
        assert result['total_syllables'] > 0
        assert result['avg_word_length'] > 0
    
    def test_empty_word_list(self, analyzer):
        """Test analysis with empty word list."""
        result = analyzer.analyze_word_complexity([])
        
        assert result['complex_word_count'] == 0
        assert result['total_syllables'] == 0
        assert result['avg_word_length'] == 0.0


class TestTextAnalyzer:
    """Test cases for TextAnalyzer."""
    
    @pytest.fixture
    def sample_article(self):
        """Create a sample article for testing."""
        return ArticleContent(
            url_id="TEST001",
            url="https://example.com/test",
            title="Test Article",
            content="This is a great test article. It contains good content for analysis. "
                   "The article is wonderful and amazing. However, some parts might be terrible.",
            word_count=20,
            character_count=150
        )
    
    @pytest.fixture
    def analyzer(self):
        """Create a text analyzer for testing."""
        analyzer = TextAnalyzer()
        # Mock sentiment words for consistent testing
        analyzer.sentiment_analyzer.positive_words = {'good', 'great', 'wonderful', 'amazing'}
        analyzer.sentiment_analyzer.negative_words = {'bad', 'terrible', 'awful', 'horrible'}
        analyzer.stop_words = {'the', 'is', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        return analyzer
    
    def test_analyze_article(self, analyzer, sample_article):
        """Test complete article analysis."""
        result = analyzer.analyze(sample_article)
        
        assert isinstance(result, TextAnalysisResult)
        assert result.url_id == "TEST001"
        assert result.url == "https://example.com/test"
        assert result.positive_score > 0
        assert result.negative_score > 0
        assert result.word_count > 0
        assert result.avg_sentence_length > 0
        assert result.fog_index >= 0
    
    def test_analyze_empty_article(self, analyzer):
        """Test analysis with empty article."""
        empty_article = ArticleContent(
            url_id="EMPTY001",
            url="https://example.com/empty",
            title="Empty Article",
            content="",
            word_count=0,
            character_count=0
        )
        
        result = analyzer.analyze(empty_article)
        
        # Should return default values
        assert result.word_count == 0
        assert result.positive_score == 0
        assert result.negative_score == 0
    
    def test_preprocess_text(self, analyzer):
        """Test text preprocessing."""
        dirty_text = "This   is    a   test\n\nwith   extra   spaces"
        clean_text = analyzer._preprocess_text(dirty_text)
        
        assert "  " not in clean_text  # No double spaces
        assert clean_text.strip() == clean_text  # No leading/trailing spaces
    
    def test_extract_words(self, analyzer):
        """Test word extraction."""
        text = "Hello, World! This is a test."
        words = analyzer._extract_words(text)
        
        assert all(word.isalpha() for word in words)
        assert all(len(word) >= 2 for word in words)
        assert "hello" in words
        assert "world" in words
    
    def test_filter_words(self, analyzer):
        """Test stop word filtering."""
        words = ["hello", "the", "world", "is", "great", "and", "amazing"]
        filtered = analyzer._filter_words(words)
        
        assert "hello" in filtered
        assert "world" in filtered
        assert "great" in filtered
        assert "amazing" in filtered
        assert "the" not in filtered
        assert "is" not in filtered
        assert "and" not in filtered


class TestContentExtractor:
    """Test cases for ContentExtractor."""
    
    @pytest.fixture
    def extractor(self):
        """Create a content extractor for testing."""
        return ContentExtractor()
    
    def test_extract_title(self, extractor):
        """Test title extraction."""
        html = """
        <html>
        <head><title>Test Page Title</title></head>
        <body>
        <h1>Main Heading</h1>
        <p>Content here</p>
        </body>
        </html>
        """
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        title = extractor._extract_title(soup)
        
        assert title in ["Test Page Title", "Main Heading"]
    
    def test_clean_content(self, extractor):
        """Test content cleaning."""
        dirty_content = """
        This is good content.
        Subscribe to our newsletter!
        Follow us on social media.
        More good content here.
        Advertisement
        """
        
        clean_content = extractor._clean_content(dirty_content)
        
        assert "Subscribe to" not in clean_content
        assert "Follow us on" not in clean_content
        assert "Advertisement" not in clean_content
        assert "good content" in clean_content
    
    @pytest.mark.asyncio
    async def test_extract_with_mock_response(self, extractor):
        """Test extraction with mocked HTTP response."""
        mock_html = """
        <html>
        <body>
        <article>
        <h1>Test Article Title</h1>
        <p>This is the main content of the article.</p>
        <p>It contains multiple paragraphs with good information.</p>
        </article>
        </body>
        </html>
        """
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value=mock_html)
            mock_response.raise_for_status = Mock()
            mock_get.return_value.__aenter__.return_value = mock_response
            
            result = await extractor.extract_from_url("https://example.com", "TEST001")
            
            assert result is not None
            assert result.url_id == "TEST001"
            assert "Test Article Title" in result.content
            assert "main content" in result.content


class TestFileManager:
    """Test cases for FileManager."""
    
    def test_load_word_list(self):
        """Test word list loading."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("good\ngreat\nexcellent\namazing\n")
            temp_path = f.name
        
        try:
            words = FileManager.load_word_list(temp_path)
            assert len(words) == 4
            assert 'good' in words
            assert 'great' in words
            assert 'excellent' in words
            assert 'amazing' in words
        finally:
            Path(temp_path).unlink()
    
    def test_load_input_data(self):
        """Test input data loading."""
        # Create temporary Excel file
        test_data = pd.DataFrame({
            'URL_ID': ['TEST001', 'TEST002'],
            'URL': ['https://example.com/1', 'https://example.com/2']
        })
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
            test_data.to_excel(f.name, index=False)
            temp_path = f.name
        
        try:
            df = FileManager.load_input_data(temp_path)
            assert len(df) == 2
            assert 'URL_ID' in df.columns
            assert 'URL' in df.columns
            assert df.iloc[0]['URL_ID'] == 'TEST001'
        finally:
            Path(temp_path).unlink()
    
    def test_save_results(self):
        """Test results saving."""
        test_results = [
            {
                'URL_ID': 'TEST001',
                'URL': 'https://example.com/1',
                'POSITIVE SCORE': 5,
                'NEGATIVE SCORE': 2,
                'WORD COUNT': 100
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            excel_path = Path(temp_dir) / "test_results.xlsx"
            csv_path = Path(temp_dir) / "test_results.csv"
            
            FileManager.save_results(test_results, str(excel_path), str(csv_path))
            
            assert excel_path.exists()
            assert csv_path.exists()
            
            # Verify content
            df = pd.read_excel(excel_path)
            assert len(df) == 1
            assert df.iloc[0]['URL_ID'] == 'TEST001'


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_analysis(self):
        """Test complete end-to-end analysis pipeline."""
        # Create test article
        test_article = ArticleContent(
            url_id="INTEGRATION001",
            url="https://example.com/integration",
            title="Integration Test Article",
            content="This is a comprehensive integration test article. "
                   "It contains good content for testing the complete analysis pipeline. "
                   "The article has excellent information and wonderful insights. "
                   "However, some aspects might be terrible or bad for testing negative sentiment.",
            word_count=50,
            character_count=300
        )
        
        # Initialize analyzer
        analyzer = TextAnalyzer()
        
        # Mock sentiment words
        analyzer.sentiment_analyzer.positive_words = {
            'good', 'excellent', 'wonderful', 'comprehensive', 'insights'
        }
        analyzer.sentiment_analyzer.negative_words = {
            'terrible', 'bad', 'awful', 'horrible'
        }
        
        # Run analysis
        result = analyzer.analyze(test_article)
        
        # Verify results
        assert isinstance(result, TextAnalysisResult)
        assert result.url_id == "INTEGRATION001"
        assert result.positive_score > 0
        assert result.negative_score > 0
        assert result.word_count > 0
        assert result.avg_sentence_length > 0
        assert -1 <= result.polarity_score <= 1
        assert 0 <= result.subjectivity_score <= 1
        assert result.fog_index >= 0
        assert result.complex_word_count >= 0
        assert result.personal_pronouns >= 0
        assert result.avg_word_length > 0
    
    def test_model_validation(self):
        """Test data model validation."""
        # Test valid TextAnalysisResult
        valid_result = TextAnalysisResult(
            url_id="TEST001",
            url="https://example.com/test",
            positive_score=5,
            negative_score=2,
            polarity_score=0.5,
            subjectivity_score=0.3,
            avg_sentence_length=15.5,
            percentage_complex_words=25.0,
            fog_index=12.5,
            avg_words_per_sentence=15.5,
            complex_word_count=10,
            word_count=50,
            syllable_per_word=1.8,
            personal_pronouns=3,
            avg_word_length=5.2
        )
        
        assert valid_result.url_id == "TEST001"
        assert valid_result.positive_score == 5
        
        # Test model conversion to dict
        result_dict = valid_result.to_dict()
        assert 'URL_ID' in result_dict
        assert 'POSITIVE SCORE' in result_dict
        assert result_dict['URL_ID'] == "TEST001"
        assert result_dict['POSITIVE SCORE'] == 5
    
    def test_error_handling(self):
        """Test error handling in various components."""
        # Test invalid URL input
        with pytest.raises(Exception):
            URLInput(url_id="", url="invalid-url")
        
        # Test empty article analysis
        analyzer = TextAnalyzer()
        empty_article = ArticleContent(
            url_id="EMPTY001",
            url="https://example.com/empty",
            title="",
            content="",
            word_count=0,
            character_count=0
        )
        
        result = analyzer.analyze(empty_article)
        assert result.word_count == 0
        assert result.positive_score == 0


# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])