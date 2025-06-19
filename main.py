"""
Blackcoffer Text Analysis System - Main Application
Production-ready text analysis system with enterprise-level architecture.
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import settings, file_paths
from exceptions import BlackcofferAnalysisError, ValidationError, FileOperationError
from logger import get_logger
from utils import FileManager, DataValidator, create_progress_callback, performance_monitor
from models import URLInput, ProcessingStats, BatchProcessingResult
from extractor import BatchExtractor, SmartContentExtractor
from analyzer import BatchTextAnalyzer, AdvancedTextAnalyzer

logger = get_logger(__name__)


class BlackcofferAnalysisSystem:
    """Main application class for the Blackcoffer Text Analysis System."""
    
    def __init__(self):
        self.extractor = BatchExtractor()
        self.analyzer = BatchTextAnalyzer()
        self.stats = ProcessingStats()
        self._validate_environment()
    
    def _validate_environment(self):
        """Validate system environment and dependencies."""
        logger.info("Validating system environment...")
        
        # Check required files
        required_files = [
            file_paths.POSITIVE_WORDS,
            file_paths.NEGATIVE_WORDS,
            Path(settings.INPUT_FILE)
        ]
        
        missing_files = [f for f in required_files if not f.exists()]
        if missing_files:
            raise FileOperationError(
                str(missing_files), 
                "validate", 
                "Required files are missing"
            )
        
        # Check stop words files
        available_stopwords = [f for f in file_paths.STOP_WORDS_FILES if f.exists()]
        if len(available_stopwords) < 3:  # At least 3 stop word files should exist
            logger.warning(f"Only {len(available_stopwords)} stop word files found")
        
        logger.info("Environment validation completed successfully")
    
    async def run_analysis(self) -> BatchProcessingResult:
        """Run the complete analysis pipeline."""
        try:
            with performance_monitor("Complete Analysis Pipeline"):
                # Load and validate input data
                input_data = await self._load_and_validate_input()
                
                # Extract content from URLs
                articles = await self._extract_content(input_data)
                
                # Analyze extracted content
                results = await self._analyze_content(articles)
                
                # Save results
                await self._save_results(results)
                
                # Generate final report
                return self._generate_report(results)
                
        except Exception as e:
            logger.critical(f"Analysis pipeline failed: {e}", exc_info=True)
            raise BlackcofferAnalysisError(f"Analysis pipeline failed: {e}")
    
    async def _load_and_validate_input(self) -> List[Dict[str, str]]:
        """Load and validate input data."""
        logger.info("Loading and validating input data...")
        
        try:
            # Load input DataFrame
            df = FileManager.load_input_data()
            
            # Validate data
            validation_issues = DataValidator.validate_input_data(df)
            if validation_issues:
                for issue in validation_issues:
                    logger.warning(f"Validation issue: {issue}")
                
                # Stop if critical issues found
                critical_issues = [i for i in validation_issues if 'Missing required columns' in i]
                if critical_issues:
                    raise ValidationError(f"Critical validation errors: {critical_issues}")
            
            # Convert to list of dictionaries
            input_data = df.to_dict('records')
            
            # Update stats
            self.stats.total_urls = len(input_data)
            
            logger.info(f"Successfully loaded {len(input_data)} URLs")
            return input_data
            
        except Exception as e:
            raise FileOperationError(settings.INPUT_FILE, "load", str(e))
    
    async def _extract_content(self, input_data: List[Dict[str, str]]) -> List:
        """Extract content from URLs."""
        logger.info("Starting content extraction...")
        logger.log_processing_start(len(input_data))
        
        # Create progress callback
        progress_callback = create_progress_callback(len(input_data), logger)
        
        try:
            # Extract content using batch extractor
            articles = await self.extractor.extract_batch(
                input_data, 
                progress_callback=progress_callback
            )
            
            # Update stats
            successful_extractions = sum(1 for a in articles if a is not None)
            self.stats.successful_extractions = successful_extractions
            self.stats.failed_extractions = len(articles) - successful_extractions
            
            logger.info(f"Content extraction completed: {successful_extractions}/{len(articles)} successful")
            
            # Save extracted articles to files
            await self._save_extracted_articles(articles)
            
            return articles
            
        except Exception as e:
            logger.error(f"Content extraction failed: {e}", exc_info=True)
            raise
    
    async def _save_extracted_articles(self, articles):
        """Save extracted articles to individual text files."""
        logger.info("Saving extracted articles to files...")
        
        saved_count = 0
        for article in articles:
            if article is not None:
                try:
                    FileManager.save_extracted_article(article.url_id, article.content)
                    saved_count += 1
                except Exception as e:
                    logger.error(f"Failed to save article {article.url_id}: {e}")
        
        logger.info(f"Saved {saved_count} article files")
    
    async def _analyze_content(self, articles) -> List:
        """Analyze extracted content."""
        logger.info("Starting text analysis...")
        
        # Filter out None articles
        valid_articles = [a for a in articles if a is not None]
        
        if not valid_articles:
            logger.error("No valid articles to analyze")
            return []
        
        # Create progress callback
        progress_callback = create_progress_callback(len(valid_articles), logger)
        
        try:
            # Run analysis in thread pool (wrapped in async)
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                self.analyzer.analyze_batch, 
                valid_articles, 
                progress_callback
            )
            
            # Update stats
            successful_analyses = sum(1 for r in results if r.word_count > 0)
            self.stats.successful_analyses = successful_analyses
            self.stats.failed_analyses = len(results) - successful_analyses
            
            logger.info(f"Text analysis completed: {successful_analyses}/{len(results)} successful")
            
            return results
            
        except Exception as e:
            logger.error(f"Text analysis failed: {e}", exc_info=True)
            raise
    
    async def _save_results(self, results) -> None:
        """Save analysis results to files."""
        logger.info("Saving analysis results...")
        
        try:
            # Convert results to dictionaries
            results_dicts = [result.to_dict() for result in results]
            
            # Save using FileManager
            FileManager.save_results(results_dicts)
            
            logger.info("Analysis results saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}", exc_info=True)
            raise
    
    def _generate_report(self, results) -> BatchProcessingResult:
        """Generate final processing report."""
        # Calculate final stats
        self.stats.total_processing_time = time.time() - self._start_time if hasattr(self, '_start_time') else 0
        self.stats.average_processing_time = (
            self.stats.total_processing_time / max(self.stats.total_urls, 1)
        )
        
        # Create batch result
        batch_result = BatchProcessingResult(
            results=results,
            stats=self.stats,
            errors=getattr(self.extractor, 'errors', []),
            warnings=[]
        )
        
        # Log completion
        logger.log_processing_complete(self.stats.to_dict())
        
        if batch_result.errors:
            logger.log_error_summary(batch_result.errors)
        
        return batch_result


class ProductionAnalysisSystem(BlackcofferAnalysisSystem):
    """Production-ready version with enhanced monitoring and error handling."""
    
    def __init__(self):
        super().__init__()
        self.extractor = BatchExtractor(max_concurrent=settings.MAX_CONCURRENT_REQUESTS)
        self.analyzer = BatchTextAnalyzer(max_workers=4)
        self._setup_monitoring()
    
    def _setup_monitoring(self):
        """Setup monitoring and health checks."""
        logger.info("Setting up production monitoring...")
        
        # Check system resources
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_count = psutil.cpu_count()
            logger.info(f"System resources: {memory_gb:.1f}GB RAM, {cpu_count} CPUs")
        except ImportError:
            logger.warning("psutil not available for system monitoring")
    
    async def run_analysis(self) -> BatchProcessingResult:
        """Enhanced analysis run with monitoring."""
        self._start_time = time.time()
        
        try:
            # Pre-flight checks
            await self._preflight_checks()
            
            # Run main analysis
            result = await super().run_analysis()
            
            # Post-processing validation
            await self._post_processing_validation(result)
            
            return result
            
        except Exception as e:
            await self._handle_critical_error(e)
            raise
    
    async def _preflight_checks(self):
        """Perform pre-flight system checks."""
        logger.info("Performing pre-flight checks...")
        
        # Check disk space
        try:
            import shutil
            free_space_gb = shutil.disk_usage('.').free / (1024**3)
            if free_space_gb < 1.0:  # Less than 1GB free
                logger.warning(f"Low disk space: {free_space_gb:.1f}GB free")
        except Exception:
            pass
        
        # Verify network connectivity
        import aiohttp
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('https://httpbin.org/get', timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status != 200:
                        logger.warning("Network connectivity test failed")
        except Exception as e:
            logger.warning(f"Network connectivity test failed: {e}")
    
    async def _post_processing_validation(self, result: BatchProcessingResult):
        """Validate results after processing."""
        logger.info("Performing post-processing validation...")
        
        # Check result quality
        if result.stats.extraction_success_rate < 50:
            logger.warning(f"Low extraction success rate: {result.stats.extraction_success_rate:.1f}%")
        
        if result.stats.analysis_success_rate < 80:
            logger.warning(f"Low analysis success rate: {result.stats.analysis_success_rate:.1f}%")
        
        # Validate output files
        output_files = [Path(settings.OUTPUT_EXCEL), Path(settings.OUTPUT_CSV)]
        for file_path in output_files:
            if not file_path.exists() or file_path.stat().st_size == 0:
                logger.error(f"Output file validation failed: {file_path}")
    
    async def _handle_critical_error(self, error: Exception):
        """Handle critical system errors."""
        logger.critical(f"Critical error in analysis system: {error}", exc_info=True)
        
        # Save partial results if available
        try:
            if hasattr(self, '_partial_results'):
                logger.info("Attempting to save partial results...")
                await self._save_results(self._partial_results)
        except Exception as e:
            logger.error(f"Failed to save partial results: {e}")


async def main():
    """Main entry point for the application."""
    try:
        # Initialize system
        system = ProductionAnalysisSystem()
        
        # Run analysis
        result = await system.run_analysis()
        
        # Print summary
        print("\n" + "="*60)
        print("BLACKCOFFER ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Total URLs processed: {result.stats.total_urls}")
        print(f"Successful extractions: {result.stats.successful_extractions}")
        print(f"Successful analyses: {result.stats.successful_analyses}")
        print(f"Overall success rate: {result.stats.analysis_success_rate:.1f}%")
        print(f"Total processing time: {result.stats.total_processing_time:.2f}s")
        print(f"Average time per URL: {result.stats.average_processing_time:.2f}s")
        print(f"\nResults saved to:")
        print(f"  - {settings.OUTPUT_EXCEL}")
        print(f"  - {settings.OUTPUT_CSV}")
        print(f"  - {settings.EXTRACTED_ARTICLES_DIR}/")
        print("="*60)
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 1
    except Exception as e:
        logger.critical(f"Application failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    # Set up event loop policy for Windows
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Run the application
    exit_code = asyncio.run(main())
    sys.exit(exit_code)