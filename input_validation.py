"""
Input Validation Module
========================
Centralized input validation to prevent DoS and memory exhaustion.
"""

from typing import Optional, Tuple
from dataclasses import dataclass
from detection_constants import MAX_PROMPT_LENGTH, MAX_BASE64_LENGTH


@dataclass
class ValidationLimits:
    """Input validation limits"""
    max_prompt_length: int = MAX_PROMPT_LENGTH
    max_base64_length: int = MAX_BASE64_LENGTH
    max_pattern_count: int = 100
    max_signals_per_pattern: int = 50


class InputValidator:
    """Validates inputs to prevent abuse"""

    def __init__(self, limits: Optional[ValidationLimits] = None):
        self.limits = limits or ValidationLimits()

    def validate_prompt(self, prompt: str) -> Tuple[bool, Optional[str]]:
        """
        Validate prompt input.

        Args:
            prompt: User prompt to validate

        Returns:
            (is_valid, error_message)
        """
        if not isinstance(prompt, str):
            return False, "Prompt must be a string"

        if len(prompt) == 0:
            return False, "Prompt cannot be empty"

        if len(prompt) > self.limits.max_prompt_length:
            return False, f"Prompt exceeds maximum length of {self.limits.max_prompt_length} characters"

        # Check for null bytes (potential injection)
        if '\x00' in prompt:
            return False, "Prompt contains null bytes"

        return True, None

    def validate_base64(self, text: str) -> bool:
        """Validate base64 string size"""
        return len(text) <= self.limits.max_base64_length

    def sanitize_prompt(self, prompt: str) -> str:
        """
        Sanitize prompt by removing potentially problematic characters.

        Args:
            prompt: Raw prompt text

        Returns:
            Sanitized prompt
        """
        # Remove null bytes
        prompt = prompt.replace('\x00', '')

        # Normalize whitespace
        prompt = ' '.join(prompt.split())

        # Truncate if too long
        if len(prompt) > self.limits.max_prompt_length:
            prompt = prompt[:self.limits.max_prompt_length]

        return prompt


# Global validator instance
_validator: Optional[InputValidator] = None


def get_validator() -> InputValidator:
    """Get global validator instance (singleton)"""
    global _validator
    if _validator is None:
        _validator = InputValidator()
    return _validator
