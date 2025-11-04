"""Tests for input validation module"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from input_validation import InputValidator, ValidationLimits


def test_valid_prompt():
    """Test valid prompt passes validation"""
    validator = InputValidator()
    is_valid, error = validator.validate_prompt("What is the capital of France?")
    assert is_valid is True
    assert error is None
    print("✓ test_valid_prompt")


def test_empty_prompt():
    """Test empty prompt fails validation"""
    validator = InputValidator()
    is_valid, error = validator.validate_prompt("")
    assert is_valid is False
    assert "cannot be empty" in error
    print("✓ test_empty_prompt")


def test_prompt_too_long():
    """Test prompt exceeding maximum length fails"""
    validator = InputValidator(ValidationLimits(max_prompt_length=100))
    long_prompt = "A" * 101
    is_valid, error = validator.validate_prompt(long_prompt)
    assert is_valid is False
    assert "exceeds maximum length" in error
    print("✓ test_prompt_too_long")


def test_prompt_with_null_bytes():
    """Test prompt with null bytes fails"""
    validator = InputValidator()
    is_valid, error = validator.validate_prompt("Hello\x00World")
    assert is_valid is False
    assert "null bytes" in error
    print("✓ test_prompt_with_null_bytes")


def test_sanitize_removes_null_bytes():
    """Test sanitize removes null bytes"""
    validator = InputValidator()
    sanitized = validator.sanitize_prompt("Hello\x00World")
    assert "\x00" not in sanitized
    assert "Hello" in sanitized and "World" in sanitized
    print("✓ test_sanitize_removes_null_bytes")


def test_sanitize_truncates_long_prompt():
    """Test sanitize truncates prompt that's too long"""
    validator = InputValidator(ValidationLimits(max_prompt_length=50))
    long_prompt = "A" * 100
    sanitized = validator.sanitize_prompt(long_prompt)
    assert len(sanitized) == 50
    print("✓ test_sanitize_truncates_long_prompt")


def test_validate_base64_size():
    """Test base64 size validation"""
    validator = InputValidator(ValidationLimits(max_base64_length=100))

    # Valid size
    assert validator.validate_base64("A" * 50) is True

    # Too large
    assert validator.validate_base64("A" * 101) is False
    print("✓ test_validate_base64_size")


def test_non_string_prompt():
    """Test non-string input fails validation"""
    validator = InputValidator()
    is_valid, error = validator.validate_prompt(None)
    assert is_valid is False
    assert "must be a string" in error
    print("✓ test_non_string_prompt")


def main():
    """Run all validation tests"""
    print("=" * 70)
    print("Input Validation Tests")
    print("=" * 70)

    tests = [
        test_valid_prompt,
        test_empty_prompt,
        test_prompt_too_long,
        test_prompt_with_null_bytes,
        test_sanitize_removes_null_bytes,
        test_sanitize_truncates_long_prompt,
        test_validate_base64_size,
        test_non_string_prompt,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print("=" * 70)
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
