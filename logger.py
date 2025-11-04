"""
Structured Logging Module
==========================
Provides structured logging with JSON output support and proper error handling.

Usage:
    from logger import get_logger
    logger = get_logger(__name__)
    logger.info("Detection completed", prompt_length=len(prompt), is_jailbreak=result)
"""
import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from logging.handlers import RotatingFileHandler


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from the record
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


class StructuredLogger(logging.LoggerAdapter):
    """
    Logger adapter that supports structured logging with extra fields
    """

    def process(self, msg: str, kwargs: Any) -> tuple:
        """Process log message and add extra fields"""
        # Extract extra fields for JSON formatter
        extra_fields = {k: v for k, v in kwargs.items() if k not in ["extra", "exc_info", "stack_info", "stacklevel"]}

        if extra_fields:
            if "extra" not in kwargs:
                kwargs["extra"] = {}
            kwargs["extra"]["extra_fields"] = extra_fields

            # Remove extra fields from kwargs to avoid conflicts
            for key in extra_fields:
                kwargs.pop(key, None)

        return msg, kwargs


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "text",
    log_to_file: bool = False,
    log_file_path: str = "./logs/llm_abuse_patterns.log",
    log_to_console: bool = True,
) -> None:
    """
    Setup logging configuration

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type (text or json)
        log_to_file: Whether to log to file
        log_file_path: Path to log file
        log_to_console: Whether to log to console
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Choose formatter
    if log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        log_path = Path(log_file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file_path, maxBytes=10 * 1024 * 1024, backupCount=5  # 10MB
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> StructuredLogger:
    """
    Get a structured logger instance

    Args:
        name: Logger name (typically __name__)

    Returns:
        StructuredLogger instance
    """
    logger = logging.getLogger(name)
    return StructuredLogger(logger, {})


# Initialize default logging
def init_default_logging():
    """Initialize logging with default settings"""
    try:
        # Try to load from config if available
        from config import get_config

        config = get_config()
        setup_logging(
            log_level=config.logging.level,
            log_format=config.logging.format,
            log_to_file=config.logging.log_to_file,
            log_file_path=config.logging.log_file_path,
            log_to_console=config.logging.log_to_console,
        )
    except (ImportError, FileNotFoundError):
        # Fallback to simple logging
        setup_logging(log_level="INFO", log_format="text", log_to_console=True)


if __name__ == "__main__":
    # Demo logging
    print("="*70)
    print("Structured Logging Demo")
    print("="*70)

    # Setup logging
    setup_logging(log_level="DEBUG", log_format="text")

    logger = get_logger(__name__)

    # Standard logging
    logger.debug("Debug message")
    logger.info("Information message")
    logger.warning("Warning message")
    logger.error("Error message")

    # Structured logging with extra fields
    logger.info(
        "Detection completed",
        prompt_length=150,
        is_jailbreak=True,
        confidence=0.95,
        method="heuristic",
    )

    logger.info("Pattern matched", pattern_id="dan-style-001", severity="high")

    # Error with exception
    try:
        raise ValueError("Example error for demonstration")
    except ValueError as e:
        logger.error("An error occurred", exc_info=True)

    # JSON format demo
    print("\n" + "="*70)
    print("JSON Format Demo")
    print("="*70)

    setup_logging(log_level="INFO", log_format="json")
    json_logger = get_logger("json_demo")

    json_logger.info("Detection result", is_jailbreak=False, confidence=0.3)
    json_logger.warning("High latency detected", latency_ms=1500, threshold_ms=1000)
