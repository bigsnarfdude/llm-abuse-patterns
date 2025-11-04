"""
Configuration Management
========================
Load and validate configuration from config.yaml using Pydantic models.

Usage:
    from config import load_config
    config = load_config()
    print(config.detection.confidence_threshold)
"""
from pathlib import Path
from typing import Literal, Optional
from pydantic import BaseModel, Field
import yaml


class DetectionConfig(BaseModel):
    """Detection settings"""
    confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)
    min_signals_normal: int = Field(2, ge=1)
    min_signals_critical: int = Field(3, ge=1)
    enable_base64_detection: bool = True
    enable_special_tokens_detection: bool = True
    enable_pattern_matching: bool = True


class GuardrailsConfig(BaseModel):
    """Guardrails settings"""
    block_on_violation: bool = True
    category_threshold: float = Field(0.3, ge=0.0, le=1.0)
    allow_educational_violence: bool = True
    allow_medical_sexual: bool = True


class DatabaseConfig(BaseModel):
    """Database settings"""
    export_directory: str = "./exports"
    export_filename: str = "pattern_database.json"
    version: str = "2.1.0"


class LoggingConfig(BaseModel):
    """Logging settings"""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["text", "json"] = "text"
    log_to_file: bool = False
    log_file_path: str = "./logs/llm_abuse_patterns.log"
    log_to_console: bool = True


class APIConfig(BaseModel):
    """API settings"""
    default_model: str = "ollama/gpt-oss-safeguard:20b"
    ollama_base_url: str = "http://localhost:11434"
    vllm_base_url: str = "http://localhost:8000"
    timeout: int = Field(60, gt=0)
    reasoning_effort: Literal["low", "medium", "high"] = "medium"


class PerformanceConfig(BaseModel):
    """Performance settings"""
    max_latency_heuristic: int = Field(10, gt=0)
    max_latency_ml: int = Field(100, gt=0)
    max_latency_llm: int = Field(1000, gt=0)
    enable_caching: bool = True
    cache_ttl_seconds: int = Field(300, gt=0)


class Config(BaseModel):
    """Root configuration model"""
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to config.yaml in current directory.

    Returns:
        Config object with validated settings

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config validation fails
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        print(f"Config file not found at {config_path}, using defaults")
        return Config()

    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    try:
        config = Config(**data)
        return config
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}")


def get_default_config() -> Config:
    """Get default configuration without loading from file"""
    return Config()


# Global config instance (lazy loaded)
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance (singleton pattern)"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


if __name__ == "__main__":
    # Demo configuration loading
    print("="*70)
    print("Configuration Demo")
    print("="*70)

    config = load_config()
    print(f"\n✓ Configuration loaded successfully!")
    print(f"\nDetection Settings:")
    print(f"  Confidence threshold: {config.detection.confidence_threshold}")
    print(f"  Min signals (normal): {config.detection.min_signals_normal}")
    print(f"  Min signals (critical): {config.detection.min_signals_critical}")

    print(f"\nGuardrails Settings:")
    print(f"  Block on violation: {config.guardrails.block_on_violation}")
    print(f"  Category threshold: {config.guardrails.category_threshold}")

    print(f"\nAPI Settings:")
    print(f"  Default model: {config.api.default_model}")
    print(f"  Timeout: {config.api.timeout}s")
    print(f"  Reasoning effort: {config.api.reasoning_effort}")

    print(f"\nLogging Settings:")
    print(f"  Level: {config.logging.level}")
    print(f"  Format: {config.logging.format}")

    # Export configuration as JSON
    print(f"\n{'='*70}")
    print("Configuration as JSON:")
    print('='*70)
    print(config.model_dump_json(indent=2))
