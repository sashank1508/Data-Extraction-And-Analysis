"""
Logging configuration for the Blackcoffer Text Analysis System.
Provides structured logging with file rotation and multiple handlers.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text

from config import settings


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding for different log levels."""
    
    COLORS = {
        'DEBUG': 'blue',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold red'
    }
    
    def format(self, record):
        # Format the base message
        formatted = super().format(record)
        
        # Add color if terminal supports it
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, 'white')
            return f"[{color}]{formatted}[/{color}]"
        
        return formatted


class BlackcofferLogger:
    """Centralized logging system for the application."""
    
    def __init__(self, name: str = "blackcoffer_analysis"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.console = Console()
        self._setup_logger()
    
    def _setup_logger(self):
        """Set up logger with multiple handlers."""
        self.logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
        
        # Prevent duplicate handlers
        if self.logger.handlers:
            return
        
        # Console handler with Rich formatting
        console_handler = RichHandler(
            console=self.console,
            show_time=True,
            show_path=True,
            rich_tracebacks=True
        )
        console_handler.setLevel(logging.INFO)
        
        # File handler with rotation
        log_file = Path(settings.LOGS_DIR) / f"{self.name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Error file handler
        error_log_file = Path(settings.LOGS_DIR) / f"{self.name}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        
        # Formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(settings.LOG_FORMAT)
        
        # Set formatters
        file_handler.setFormatter(detailed_formatter)
        error_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, exc_info: bool = False, **kwargs):
        """Log error message."""
        self.logger.error(message, exc_info=exc_info, **kwargs)
    
    def critical(self, message: str, exc_info: bool = False, **kwargs):
        """Log critical message."""
        self.logger.critical(message, exc_info=exc_info, **kwargs)
    
    def log_processing_start(self, total_urls: int):
        """Log the start of processing."""
        self.console.print(
            f"\n🚀 [bold green]Starting Text Analysis[/bold green]",
            f"\n📊 [blue]Total URLs to process: {total_urls}[/blue]",
            f"\n⏰ [yellow]Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/yellow]\n"
        )
    
    def log_processing_progress(self, current: int, total: int, url_id: str, success: bool):
        """Log processing progress."""
        percentage = (current / total) * 100
        status = "✅" if success else "❌"
        
        self.console.print(
            f"{status} [{current:3d}/{total}] ({percentage:5.1f}%) Processing: {url_id}"
        )
    
    def log_processing_complete(self, stats: dict):
        """Log processing completion with statistics."""
        self.console.print(
            f"\n🎉 [bold green]Processing Complete![/bold green]",
            f"\n📈 [blue]Statistics:[/blue]",
            f"\n   • Total URLs: {stats['total_urls']}",
            f"\n   • Successful extractions: {stats['successful_extractions']}",
            f"\n   • Failed extractions: {stats['failed_extractions']}",
            f"\n   • Success rate: {stats['extraction_success_rate']:.1f}%",
            f"\n   • Total time: {stats['total_processing_time']:.2f}s",
            f"\n   • Average time per URL: {stats['average_processing_time']:.2f}s"
        )
    
    def log_error_summary(self, errors: list):
        """Log error summary."""
        if not errors:
            return
        
        self.console.print(f"\n⚠️  [bold red]Errors encountered:[/bold red]")
        for i, error in enumerate(errors[:5], 1):  # Show first 5 errors
            self.console.print(f"   {i}. {error}")
        
        if len(errors) > 5:
            self.console.print(f"   ... and {len(errors) - 5} more errors (check log file)")


# Global logger instance
logger = BlackcofferLogger()


def get_logger(name: Optional[str] = None) -> BlackcofferLogger:
    """Get logger instance."""
    if name:
        return BlackcofferLogger(name)
    return logger