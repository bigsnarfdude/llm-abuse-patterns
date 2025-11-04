# Quick Reference Guide - Code Review Remediation

**Quick lookup for common fixes and best practices**

---

## Exception Handling

### ❌ Bad

```python
try:
    result = risky_operation()
except:  # Don't do this!
    pass
```

### ✅ Good

```python
try:
    result = risky_operation()
except (ValueError, TypeError) as e:
    logger.error(f"Operation failed: {e}")
    raise
except Exception as e:
    logger.exception("Unexpected error")
    raise OperationError(f"Failed: {e}") from e
```

---

## Input Validation

### ❌ Bad

```python
def detect(self, prompt: str):
    # No validation - could crash on None, empty string, huge input
    prompt_lower = prompt.lower()
```

### ✅ Good

```python
from input_validation import get_validator
from exceptions import InvalidInputError

def detect(self, prompt: str):
    validator = get_validator()
    is_valid, error = validator.validate_prompt(prompt)
    if not is_valid:
        raise InvalidInputError(error)

    prompt = validator.sanitize_prompt(prompt)
    prompt_lower = prompt.lower()
```

---

## Regex Patterns

### ❌ Bad (Recompiled every time)

```python
def check_pattern(self, text):
    if re.search(r'pattern', text, re.IGNORECASE):
        return True
```

### ✅ Good (Precompiled)

```python
class Detector:
    def __init__(self):
        self.pattern = re.compile(r'pattern', re.IGNORECASE)

    def check_pattern(self, text):
        if self.pattern.search(text):
            return True
```

---

## Magic Numbers

### ❌ Bad

```python
confidence = min(0.5 + (len(signals) * 0.1), 0.95)
```

### ✅ Good

```python
from detection_constants import BASE_CONFIDENCE, SIGNAL_WEIGHT, MAX_CONFIDENCE

confidence = min(
    BASE_CONFIDENCE + (len(signals) * SIGNAL_WEIGHT),
    MAX_CONFIDENCE
)
```

---

## Error Messages

### ❌ Bad

```python
raise ValueError("error")
```

### ✅ Good

```python
raise InvalidInputError(
    f"Prompt exceeds maximum length: {len(prompt)} > {MAX_LENGTH}"
)
```

---

## Logging

### ❌ Bad

```python
print(f"Error: {e}")
```

### ✅ Good

```python
from logger import get_logger

logger = get_logger(__name__)
logger.error("Detection failed", error=str(e), prompt_length=len(prompt))
```

---

## File Paths

### ❌ Bad

```python
sys.path.insert(0, '/home/user/llm-abuse-patterns')
```

### ✅ Good

```python
from pathlib import Path

project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))
```

---

## Database Operations

### ❌ Bad

```python
def add_pattern(self, pattern):
    try:
        # ... insert ...
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
```

### ✅ Good

```python
from exceptions import DatabaseError, PatternAlreadyExistsError
from logger import get_logger

logger = get_logger(__name__)

def add_pattern(self, pattern):
    try:
        # Check if exists
        if self._pattern_exists(pattern['pattern_id']):
            raise PatternAlreadyExistsError(
                f"Pattern {pattern['pattern_id']} already exists"
            )
        # ... insert ...
        logger.info("Pattern added", pattern_id=pattern['pattern_id'])
        return True
    except PatternAlreadyExistsError:
        logger.warning("Duplicate pattern", pattern_id=pattern['pattern_id'])
        raise
    except sqlite3.Error as e:
        logger.error("Database error", error=str(e))
        raise DatabaseError(f"Failed to add pattern: {e}") from e
```

---

## Type Hints

### ❌ Bad

```python
def detect(self, prompt):
    return {"is_jailbreak": True}
```

### ✅ Good

```python
from typing import Dict

def detect(self, prompt: str) -> Dict[str, Any]:
    """
    Detect jailbreak attempts.

    Args:
        prompt: User input to analyze

    Returns:
        Dictionary with detection results

    Raises:
        InvalidInputError: If prompt is invalid
    """
    return {"is_jailbreak": True}
```

---

## Configuration

### ❌ Bad (Hardcoded)

```python
MAX_REQUESTS = 100
THRESHOLD = 0.7
```

### ✅ Good (Config file)

```python
from config import get_config

config = get_config()
max_requests = config.performance.max_requests
threshold = config.detection.confidence_threshold
```

---

## Testing

### Minimum Test Coverage

```python
def test_function_name():
    """Test description"""
    # Arrange
    detector = LocalPatternDetector()

    # Act
    result = detector.detect("test input")

    # Assert
    assert result['is_jailbreak'] is False
```

### Test Edge Cases

```python
@pytest.mark.parametrize("input_value,expected", [
    ("", False),  # Empty string
    ("a" * 100000, False),  # Very long
    (None, ValueError),  # None
    ("普通话", False),  # Unicode
])
def test_edge_cases(input_value, expected):
    detector = LocalPatternDetector()
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            detector.detect(input_value)
    else:
        result = detector.detect(input_value)
        assert result['is_jailbreak'] == expected
```

---

## Common Imports

```python
# Standard library
import re
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Project modules
from config import get_config
from logger import get_logger
from input_validation import get_validator
from exceptions import InvalidInputError, DetectionError
from detection_constants import (
    BASE_CONFIDENCE,
    SIGNAL_WEIGHT,
    MAX_CONFIDENCE
)
```

---

## Checklist for New Functions

- [ ] Add type hints to all parameters and return values
- [ ] Add comprehensive docstring
- [ ] Validate all inputs
- [ ] Use specific exceptions (not bare `except:`)
- [ ] Add structured logging
- [ ] Use constants instead of magic numbers
- [ ] Write unit tests (including edge cases)
- [ ] Handle errors gracefully
- [ ] Log errors with context

---

## Before Committing

```bash
# Run tests
python tests/test_all.py
pytest tests/ -v

# Check for common issues
grep -r "except:" --include="*.py" .  # Should find none
grep -r "print(" --include="*.py" . | grep -v "# OK"  # Use logger instead

# Format code (if black is installed)
black .

# Check imports
# isort .
```

---

## Common Mistakes to Avoid

1. **Bare except clauses** - Always specify exception types
2. **No input validation** - Always validate user input
3. **Hardcoded paths** - Use `Path` or relative imports
4. **Magic numbers** - Extract to constants
5. **Print statements** - Use structured logging
6. **No error context** - Include relevant data in error messages
7. **Missing type hints** - Add them for better IDE support
8. **No tests** - Write tests as you code
9. **Ignoring exceptions** - At least log them
10. **SQL concatenation** - Use parameterized queries

---

## Performance Tips

1. **Precompile regex** - Compile once in `__init__`
2. **Cache results** - Use `@lru_cache` for expensive operations
3. **Limit input size** - Prevent memory exhaustion
4. **Use indices** - Add database indices for common queries
5. **Profile code** - Use `cProfile` to find bottlenecks

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def expensive_operation(param: str) -> str:
    # Results cached automatically
    return complex_calculation(param)
```

---

## Security Checklist

- [ ] Validate all user inputs
- [ ] Limit input sizes
- [ ] Use parameterized SQL queries
- [ ] Don't log sensitive data
- [ ] Set appropriate timeouts
- [ ] Implement rate limiting
- [ ] Sanitize error messages (don't leak internals)
- [ ] Use specific exceptions (not generic)

---

## Documentation Template

```python
def function_name(param1: str, param2: int = 0) -> Dict[str, Any]:
    """
    One-line summary of what the function does.

    Longer description if needed. Explain the purpose,
    algorithm, or any important details.

    Args:
        param1: Description of param1
        param2: Description of param2 (default: 0)

    Returns:
        Dictionary containing:
        - key1: Description
        - key2: Description

    Raises:
        InvalidInputError: If param1 is invalid
        DetectionError: If detection fails

    Example:
        >>> result = function_name("test", 5)
        >>> print(result['key1'])
        'value'
    """
    # Implementation
```

---

## Git Commit Messages

### Format

```
<type>: <subject>

<body>

<footer>
```

### Types

- `fix:` Bug fixes
- `feat:` New features
- `refactor:` Code refactoring
- `test:` Adding tests
- `docs:` Documentation
- `perf:` Performance improvements
- `chore:` Maintenance

### Example

```
fix: replace bare except clauses with specific exceptions

- Updated 01_safeguard_pattern_detector.py to catch specific exceptions
- Added proper error logging
- Added tests for exception handling

Fixes #123
```

---

## Need Help?

- Review `REMEDIATION_PLAN.md` for detailed fixes
- Check `CODE_REVIEW.md` for full analysis
- Look at existing code for patterns
- Run tests to verify changes
