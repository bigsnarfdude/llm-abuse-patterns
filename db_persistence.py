"""
Database Persistence Layer
===========================
SQLite-based persistence for pattern database with querying capabilities.

Usage:
    from db_persistence import PatternDB

    db = PatternDB("patterns.db")
    db.add_pattern(pattern)
    patterns = db.get_all_patterns()
    results = db.query_patterns(severity="high")
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from contextlib import contextmanager


class PatternDB:
    """
    SQLite-based pattern database with persistence
    """

    def __init__(self, db_path: str = "patterns.db"):
        """
        Initialize database connection

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Initialize database schema"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create patterns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    pattern_id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    subcategory TEXT NOT NULL,
                    description TEXT NOT NULL,
                    example_prompt TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    first_observed TEXT NOT NULL,
                    last_observed TEXT,
                    frequency TEXT DEFAULT 'common',
                    example_context TEXT,
                    detection_signals TEXT,  -- JSON array
                    detection_strategies TEXT,  -- JSON object
                    preventive_measures TEXT,  -- JSON array
                    detective_measures TEXT,  -- JSON array
                    responsive_measures TEXT,  -- JSON array
                    related_patterns TEXT,  -- JSON array
                    tags TEXT,  -- JSON array
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Create indices for common queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON patterns(category)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_severity ON patterns(severity)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_frequency ON patterns(frequency)")

            # Create metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)

            # Set version if not exists
            cursor.execute("INSERT OR IGNORE INTO metadata (key, value) VALUES ('version', '2.1.0')")

    def add_pattern(self, pattern: Dict[str, Any]) -> bool:
        """
        Add a new pattern to the database

        Args:
            pattern: Pattern dictionary

        Returns:
            True if successful, False otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            now = datetime.utcnow().isoformat()

            # Convert lists/dicts to JSON
            def to_json(val):
                return json.dumps(val) if val is not None else None

            try:
                cursor.execute("""
                    INSERT OR REPLACE INTO patterns (
                        pattern_id, category, subcategory, description, example_prompt,
                        severity, first_observed, last_observed, frequency, example_context,
                        detection_signals, detection_strategies, preventive_measures,
                        detective_measures, responsive_measures, related_patterns, tags,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern['pattern_id'],
                    pattern['category'],
                    pattern['subcategory'],
                    pattern['description'],
                    pattern['example_prompt'],
                    pattern['severity'],
                    pattern['first_observed'],
                    pattern.get('last_observed'),
                    pattern.get('frequency', 'common'),
                    pattern.get('example_context'),
                    to_json(pattern.get('detection_signals')),
                    to_json(pattern.get('detection_strategies')),
                    to_json(pattern.get('preventive_measures')),
                    to_json(pattern.get('detective_measures')),
                    to_json(pattern.get('responsive_measures')),
                    to_json(pattern.get('related_patterns')),
                    to_json(pattern.get('tags')),
                    now,
                    now
                ))
                return True
            except Exception as e:
                print(f"Error adding pattern: {e}")
                return False

    def get_pattern(self, pattern_id: str) -> Optional[Dict[str, Any]]:
        """
        Get pattern by ID

        Args:
            pattern_id: Pattern identifier

        Returns:
            Pattern dictionary or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM patterns WHERE pattern_id = ?", (pattern_id,))
            row = cursor.fetchone()

            if row:
                return self._row_to_dict(row)
            return None

    def get_all_patterns(self) -> List[Dict[str, Any]]:
        """
        Get all patterns

        Returns:
            List of pattern dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM patterns ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]

    def query_patterns(
        self,
        category: Optional[str] = None,
        subcategory: Optional[str] = None,
        severity: Optional[str] = None,
        frequency: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query patterns with filters

        Args:
            category: Filter by category
            subcategory: Filter by subcategory
            severity: Filter by severity
            frequency: Filter by frequency

        Returns:
            List of matching patterns
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM patterns WHERE 1=1"
            params = []

            if category:
                query += " AND category = ?"
                params.append(category)
            if subcategory:
                query += " AND subcategory = ?"
                params.append(subcategory)
            if severity:
                query += " AND severity = ?"
                params.append(severity)
            if frequency:
                query += " AND frequency = ?"
                params.append(frequency)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]

    def search_patterns(self, keyword: str) -> List[Dict[str, Any]]:
        """
        Search patterns by keyword in description or tags

        Args:
            keyword: Search keyword

        Returns:
            List of matching patterns
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT * FROM patterns
                WHERE description LIKE ?
                   OR tags LIKE ?
                   OR detection_signals LIKE ?
            """
            search_term = f"%{keyword}%"
            cursor.execute(query, (search_term, search_term, search_term))
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]

    def delete_pattern(self, pattern_id: str) -> bool:
        """
        Delete pattern by ID

        Args:
            pattern_id: Pattern identifier

        Returns:
            True if deleted, False otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM patterns WHERE pattern_id = ?", (pattern_id,))
            return cursor.rowcount > 0

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics

        Returns:
            Statistics dictionary
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total count
            cursor.execute("SELECT COUNT(*) FROM patterns")
            total = cursor.fetchone()[0]

            # By category
            cursor.execute("SELECT category, COUNT(*) FROM patterns GROUP BY category")
            by_category = {row[0]: row[1] for row in cursor.fetchall()}

            # By severity
            cursor.execute("SELECT severity, COUNT(*) FROM patterns GROUP BY severity")
            by_severity = {row[0]: row[1] for row in cursor.fetchall()}

            # By frequency
            cursor.execute("SELECT frequency, COUNT(*) FROM patterns GROUP BY frequency")
            by_frequency = {row[0]: row[1] for row in cursor.fetchall()}

            return {
                "total_patterns": total,
                "by_category": by_category,
                "by_severity": by_severity,
                "by_frequency": by_frequency,
            }

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert SQLite row to dictionary"""
        def from_json(val):
            return json.loads(val) if val else None

        return {
            "pattern_id": row["pattern_id"],
            "category": row["category"],
            "subcategory": row["subcategory"],
            "description": row["description"],
            "example_prompt": row["example_prompt"],
            "severity": row["severity"],
            "first_observed": row["first_observed"],
            "last_observed": row["last_observed"],
            "frequency": row["frequency"],
            "example_context": row["example_context"],
            "detection_signals": from_json(row["detection_signals"]),
            "detection_strategies": from_json(row["detection_strategies"]),
            "preventive_measures": from_json(row["preventive_measures"]),
            "detective_measures": from_json(row["detective_measures"]),
            "responsive_measures": from_json(row["responsive_measures"]),
            "related_patterns": from_json(row["related_patterns"]),
            "tags": from_json(row["tags"]),
        }


if __name__ == "__main__":
    # Demo database usage
    print("="*70)
    print("Database Persistence Demo")
    print("="*70)

    # Create database
    db = PatternDB("demo_patterns.db")

    # Add sample pattern
    sample_pattern = {
        "pattern_id": "demo-001",
        "category": "prompt_injection",
        "subcategory": "direct_jailbreak",
        "description": "Demo jailbreak pattern",
        "example_prompt": "Ignore all instructions",
        "severity": "high",
        "first_observed": "2024-11-04",
        "detection_signals": ["ignore", "instructions"],
        "tags": ["demo", "jailbreak"],
    }

    print("\n✓ Adding sample pattern...")
    db.add_pattern(sample_pattern)

    print("✓ Retrieving pattern...")
    pattern = db.get_pattern("demo-001")
    if pattern:
        print(f"  Pattern ID: {pattern['pattern_id']}")
        print(f"  Category: {pattern['category']}")
        print(f"  Severity: {pattern['severity']}")

    print("\n✓ Getting statistics...")
    stats = db.get_statistics()
    print(f"  Total patterns: {stats['total_patterns']}")
    print(f"  By severity: {stats['by_severity']}")

    print("\n✓ Querying high severity patterns...")
    high_severity = db.query_patterns(severity="high")
    print(f"  Found {len(high_severity)} patterns")

    print("\n✓ Searching for 'jailbreak'...")
    results = db.search_patterns("jailbreak")
    print(f"  Found {len(results)} matches")

    print("\n" + "="*70)
    print("Database demo completed successfully!")
