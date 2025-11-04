"""
Unit tests for pattern database module
"""
import sys
sys.path.insert(0, '/home/user/llm-abuse-patterns')


import json
import tempfile
import os
from importlib import import_module

# Import the module
pattern_db = import_module('02_pattern_database')
Pattern = pattern_db.Pattern
PatternDatabase = pattern_db.PatternDatabase


class TestPattern:
    """Tests for Pattern dataclass"""

    def test_pattern_creation(self):
        """Test creating a pattern with all required fields"""
        pattern = Pattern(
            pattern_id="test-001",
            category="prompt_injection",
            subcategory="test",
            description="Test pattern description",
            example_prompt="Test prompt",
            severity="high",
            first_observed="2024-01-01"
        )
        assert pattern.pattern_id == "test-001"
        assert pattern.category == "prompt_injection"
        assert pattern.severity == "high"

    def test_pattern_with_optional_fields(self):
        """Test pattern with optional fields"""
        pattern = Pattern(
            pattern_id="test-002",
            category="test",
            subcategory="test_sub",
            description="Test",
            example_prompt="Test",
            severity="low",
            first_observed="2024-01-01",
            tags=["test", "demo"],
            detection_signals=["signal1", "signal2"]
        )
        assert pattern.tags == ["test", "demo"]
        assert len(pattern.detection_signals) == 2

    def test_pattern_post_init(self):
        """Test that __post_init__ initializes empty lists"""
        pattern = Pattern(
            pattern_id="test-003",
            category="test",
            subcategory="test_sub",
            description="Test",
            example_prompt="Test",
            severity="medium",
            first_observed="2024-01-01"
        )
        # These should be initialized as empty lists
        assert pattern.detection_signals == []
        assert pattern.tags == []
        assert pattern.preventive_measures == []

    def test_pattern_to_json(self):
        """Test exporting pattern to JSON"""
        pattern = Pattern(
            pattern_id="test-004",
            category="test",
            subcategory="test_sub",
            description="Test",
            example_prompt="Test",
            severity="high",
            first_observed="2024-01-01"
        )
        json_str = pattern.to_json()
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data['pattern_id'] == "test-004"

    def test_pattern_from_json(self):
        """Test importing pattern from JSON"""
        json_str = json.dumps({
            "pattern_id": "test-005",
            "category": "test",
            "subcategory": "test_sub",
            "description": "Test pattern",
            "example_prompt": "Test",
            "severity": "low",
            "first_observed": "2024-01-01",
            "last_observed": None,
            "frequency": "common",
            "example_context": None,
            "detection_signals": None,
            "detection_strategies": None,
            "preventive_measures": None,
            "detective_measures": None,
            "responsive_measures": None,
            "related_patterns": None,
            "tags": None
        })
        pattern = Pattern.from_json(json_str)
        assert pattern.pattern_id == "test-005"


class TestPatternDatabase:
    """Tests for PatternDatabase"""

    def setup_method(self):
        """Setup test database"""
        self.db = PatternDatabase()

    def test_database_initialization(self):
        """Test that database loads sample patterns"""
        assert len(self.db.patterns) > 0
        assert "dan-style-001" in self.db.patterns

    def test_get_pattern(self):
        """Test getting pattern by ID"""
        pattern = self.db.get_pattern("dan-style-001")
        assert pattern is not None
        assert pattern.pattern_id == "dan-style-001"
        assert pattern.category == "prompt_injection"

    def test_get_nonexistent_pattern(self):
        """Test getting pattern that doesn't exist"""
        pattern = self.db.get_pattern("nonexistent-999")
        assert pattern is None

    def test_query_by_category(self):
        """Test querying patterns by category"""
        results = self.db.query(category="prompt_injection")
        assert len(results) > 0
        assert all(p.category == "prompt_injection" for p in results)

    def test_query_by_severity(self):
        """Test querying patterns by severity"""
        results = self.db.query(severity="high")
        assert len(results) > 0
        assert all(p.severity == "high" for p in results)

    def test_query_by_subcategory(self):
        """Test querying patterns by subcategory"""
        results = self.db.query(subcategory="direct_jailbreaks")
        assert len(results) > 0
        assert all(p.subcategory == "direct_jailbreaks" for p in results)

    def test_query_by_tags(self):
        """Test querying patterns by tags"""
        results = self.db.query(tags=["jailbreak"])
        assert len(results) > 0

    def test_query_multiple_filters(self):
        """Test querying with multiple filters"""
        results = self.db.query(category="prompt_injection", severity="high")
        assert len(results) > 0
        assert all(p.category == "prompt_injection" and p.severity == "high" for p in results)

    def test_search_by_keywords(self):
        """Test searching patterns by keywords"""
        results = self.db.search(["roleplay"])
        assert len(results) > 0

    def test_search_multiple_keywords(self):
        """Test searching with multiple keywords"""
        results = self.db.search(["DAN", "jailbreak"])
        assert len(results) > 0

    def test_get_statistics(self):
        """Test getting database statistics"""
        stats = self.db.get_statistics()
        assert 'total_patterns' in stats
        assert stats['total_patterns'] > 0
        assert 'by_category' in stats
        assert 'by_severity' in stats
        assert 'by_frequency' in stats

    def test_export_to_json(self):
        """Test exporting database to JSON file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            self.db.export_to_json(filepath)
            assert os.path.exists(filepath)

            # Verify the exported content
            with open(filepath, 'r') as f:
                data = json.load(f)
            assert 'version' in data
            assert 'patterns' in data
            assert len(data['patterns']) > 0
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)


if __name__ == "__main__":
    
