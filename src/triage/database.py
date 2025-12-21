"""
Database module for incident triage application.

Provides SQLite-based persistence for:
- Analysis history
- Bookmarks
- Tags
- User settings
- Analyst profiles
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Any
from contextlib import contextmanager


class TriageDatabase:
    """Manages SQLite database for triage application."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file.
                    Defaults to 'data/triage.db'
        """
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "data" / "triage.db"
        else:
            db_path = Path(db_path)

        # Ensure directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db_path = str(db_path)
        self._init_database()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)  # Add 30 second timeout
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_database(self):
        """Initialize database schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Analysis history table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    incident_text TEXT NOT NULL,
                    final_label TEXT NOT NULL,
                    max_prob REAL NOT NULL,
                    uncertainty_level TEXT,
                    analysis_mode TEXT,  -- 'single' or 'bulk'
                    difficulty TEXT,
                    threshold REAL,
                    use_llm INTEGER,  -- 0 or 1
                    raw_result TEXT,  -- JSON of full result
                    batch_id TEXT,  -- UUID for batch analyses
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Batch analyses table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS batch_analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT UNIQUE NOT NULL,
                    batch_name TEXT,
                    file_name TEXT,
                    total_incidents INTEGER,
                    timestamp TEXT NOT NULL,
                    use_preprocessing INTEGER,
                    use_llm INTEGER,
                    summary_stats TEXT,  -- JSON: {"avg_confidence": 0.85, ...}
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Bookmarks table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS bookmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER,
                    incident_text TEXT NOT NULL,
                    final_label TEXT,
                    note TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (analysis_id) REFERENCES analysis_history(id)
                )
            """
            )

            # Tags table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    color TEXT,  -- Hex color code
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Analysis-Tag junction table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_tags (
                    analysis_id INTEGER,
                    tag_id INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (analysis_id, tag_id),
                    FOREIGN KEY (analysis_id) REFERENCES analysis_history(id),
                    FOREIGN KEY (tag_id) REFERENCES tags(id)
                )
            """
            )

            # Notes table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_id INTEGER,
                    note_text TEXT NOT NULL,
                    author TEXT,  -- For future multi-user support
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT,
                    FOREIGN KEY (analysis_id) REFERENCES analysis_history(id)
                )
            """
            )

            # User settings table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,  -- JSON-encoded value
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # User profiles table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    default_difficulty TEXT,
                    default_threshold REAL,
                    default_max_classes INTEGER,
                    enable_llm INTEGER,
                    enable_advanced_viz INTEGER,
                    preferences TEXT,  -- JSON for other preferences
                    is_active INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT
                )
            """
            )

            # Feature flags table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS feature_flags (
                    flag_name TEXT PRIMARY KEY,
                    enabled INTEGER NOT NULL,  -- 0 or 1
                    description TEXT,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create indexes for performance
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_history_timestamp 
                ON analysis_history(timestamp)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_history_label 
                ON analysis_history(final_label)
            """
            )

            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_bookmarks_created 
                ON bookmarks(created_at)
            """
            )

    # Analysis History Methods

    def save_analysis(
        self,
        incident_text: str,
        final_label: str,
        max_prob: float,
        uncertainty_level: Optional[str] = None,
        analysis_mode: str = "single",
        difficulty: str = "default",
        threshold: float = 0.5,
        use_llm: bool = False,
        raw_result: Optional[Dict[str, Any]] = None,
        batch_id: Optional[str] = None,
    ) -> int:
        """
        Save analysis result to history.

        Args:
            batch_id: Optional UUID for batch analyses

        Returns:
            Analysis ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            timestamp = datetime.now().isoformat()
            raw_result_json = json.dumps(raw_result) if raw_result else None

            cursor.execute(
                """
                INSERT INTO analysis_history 
                (timestamp, incident_text, final_label, max_prob, uncertainty_level,
                 analysis_mode, difficulty, threshold, use_llm, raw_result, batch_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    timestamp,
                    incident_text,
                    final_label,
                    max_prob,
                    uncertainty_level,
                    analysis_mode,
                    difficulty,
                    threshold,
                    int(use_llm),
                    raw_result_json,
                    batch_id,
                ),
            )

            return cursor.lastrowid

    def save_batch_analysis(
        self,
        batch_id: str,
        batch_name: str,
        file_name: str,
        results: List[Dict[str, Any]],
        use_preprocessing: bool = False,
        use_llm: bool = False,
    ) -> int:
        """
        Save batch analysis metadata and all incidents.

        Args:
            batch_id: Unique UUID for this batch
            batch_name: User-friendly name for the batch
            file_name: Original filename
            results: List of analysis results
            use_preprocessing: Whether preprocessing was enabled
            use_llm: Whether LLM was enabled

        Returns:
            Batch record ID
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            timestamp = datetime.now().isoformat()
            total_incidents = len(results)

            # Calculate summary statistics
            avg_confidence = (
                sum(r.get("max_prob", 0) for r in results) / total_incidents
                if total_incidents > 0
                else 0
            )

            from collections import Counter

            label_counts = Counter(r.get("display_label", "unknown") for r in results)

            summary_stats = {
                "avg_confidence": avg_confidence,
                "total_incidents": total_incidents,
                "label_distribution": dict(label_counts),
                "high_confidence_count": len(
                    [r for r in results if r.get("max_prob", 0) >= 0.8]
                ),
                "low_confidence_count": len(
                    [r for r in results if r.get("max_prob", 0) < 0.5]
                ),
            }

            # Save batch metadata
            cursor.execute(
                """
                INSERT INTO batch_analyses
                (batch_id, batch_name, file_name, total_incidents, timestamp,
                 use_preprocessing, use_llm, summary_stats)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    batch_id,
                    batch_name,
                    file_name,
                    total_incidents,
                    timestamp,
                    int(use_preprocessing),
                    int(use_llm),
                    json.dumps(summary_stats),
                ),
            )

            batch_record_id = cursor.lastrowid

            # Save all incidents in the same connection (avoid nested connections)
            saved_ids = []
            for result in results:
                incident_timestamp = datetime.now().isoformat()
                raw_result_json = json.dumps(result)

                cursor.execute(
                    """
                    INSERT INTO analysis_history 
                    (timestamp, incident_text, final_label, max_prob, uncertainty_level,
                     analysis_mode, difficulty, threshold, use_llm, raw_result, batch_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        incident_timestamp,
                        result.get("incident_text", ""),
                        result.get("final_label", "unknown"),
                        result.get("max_prob", 0.0),
                        None,  # uncertainty_level
                        "batch",  # analysis_mode
                        "default",  # difficulty
                        0.0,  # threshold
                        int(use_llm),
                        raw_result_json,
                        batch_id,
                    ),
                )
                saved_ids.append(cursor.lastrowid)

            return batch_record_id

    def get_batch_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get list of all batch analyses.

        Returns:
            List of batch metadata records
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM batch_analyses
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (limit,),
            )

            rows = cursor.fetchall()
            results = []
            for row in rows:
                record = dict(row)
                # Parse JSON summary_stats
                if record.get("summary_stats"):
                    record["summary_stats"] = json.loads(record["summary_stats"])
                results.append(record)

            return results

    def get_batch_by_id(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Get batch metadata by batch_id.

        Returns:
            Batch metadata or None
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                "SELECT * FROM batch_analyses WHERE batch_id = ?",
                (batch_id,),
            )

            row = cursor.fetchone()
            if row:
                record = dict(row)
                if record.get("summary_stats"):
                    record["summary_stats"] = json.loads(record["summary_stats"])
                return record
            return None

    def get_batch_incidents(self, batch_id: str) -> List[Dict[str, Any]]:
        """
        Get all incidents from a specific batch.

        Returns:
            List of incident records
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM analysis_history
                WHERE batch_id = ?
                ORDER BY id
            """,
                (batch_id,),
            )

            rows = cursor.fetchall()
            results = []
            for row in rows:
                record = dict(row)
                # Parse JSON raw_result
                if record.get("raw_result"):
                    record["raw_result"] = json.loads(record["raw_result"])
                results.append(record)

            return results

    def update_analysis(
        self,
        analysis_id: int,
        final_label: str,
        max_prob: float,
        uncertainty_level: Optional[str] = None,
        use_llm: bool = False,
        raw_result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update an existing analysis record (e.g., when re-running with LLM).
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            timestamp = datetime.now().isoformat()
            raw_result_json = json.dumps(raw_result) if raw_result else None

            cursor.execute(
                """
                UPDATE analysis_history
                SET timestamp = ?, final_label = ?, max_prob = ?, 
                    uncertainty_level = ?, use_llm = ?, raw_result = ?
                WHERE id = ?
                """,
                (
                    timestamp,
                    final_label,
                    max_prob,
                    uncertainty_level,
                    int(use_llm),
                    raw_result_json,
                    analysis_id,
                ),
            )

    def get_analysis_history(
        self,
        limit: int = 100,
        offset: int = 0,
        label_filter: Optional[str] = None,
        mode_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get analysis history with optional filtering."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM analysis_history WHERE 1=1"
            params = []

            if label_filter:
                query += " AND final_label = ?"
                params.append(label_filter)

            if mode_filter:
                query += " AND analysis_mode = ?"
                params.append(mode_filter)

            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)
            rows = cursor.fetchall()

            return [dict(row) for row in rows]

    def get_analysis_by_id(self, analysis_id: int) -> Optional[Dict[str, Any]]:
        """Get a single analysis record by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM analysis_history WHERE id = ?", (analysis_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def clear_history(self):
        """
        Clear all analysis history and related data.

        WARNING: This permanently deletes:
        - All analysis history records
        - All batch analysis records
        - All bookmarks
        - All tags and tag associations
        - All analyst notes
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Delete in proper order to respect foreign key constraints
            cursor.execute("DELETE FROM analysis_tags")
            cursor.execute("DELETE FROM notes")
            cursor.execute("DELETE FROM bookmarks")
            cursor.execute("DELETE FROM tags")
            cursor.execute("DELETE FROM analysis_history")
            cursor.execute("DELETE FROM batch_analyses")

    def search_history(
        self, search_term: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Full-text search across incident text."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM analysis_history
                WHERE incident_text LIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (f"%{search_term}%", limit),
            )

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def advanced_search(
        self,
        search_term: str = "",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        label_filter: Optional[str] = None,
        min_confidence: Optional[float] = None,
        max_confidence: Optional[float] = None,
        tag_ids: Optional[List[int]] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Advanced search across analysis history with multiple filters.

        Args:
            search_term: Text to search in incident text
            start_date: ISO format date string (e.g., '2024-01-01')
            end_date: ISO format date string
            label_filter: Specific label to filter by
            min_confidence: Minimum confidence threshold (0.0-1.0)
            max_confidence: Maximum confidence threshold (0.0-1.0)
            tag_ids: List of tag IDs to filter by
            limit: Maximum results to return
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT DISTINCT ah.* FROM analysis_history ah"
            params = []
            conditions = []

            # Join with tags if filtering by tags
            if tag_ids:
                query += " LEFT JOIN analysis_tags at ON ah.id = at.analysis_id"
                conditions.append(f"at.tag_id IN ({','.join(['?'] * len(tag_ids))})")
                params.extend(tag_ids)

            # Text search
            if search_term:
                conditions.append("ah.incident_text LIKE ?")
                params.append(f"%{search_term}%")

            # Date range
            if start_date:
                conditions.append("ah.timestamp >= ?")
                params.append(start_date)
            if end_date:
                conditions.append("ah.timestamp <= ?")
                params.append(end_date)

            # Label filter
            if label_filter:
                conditions.append("ah.final_label = ?")
                params.append(label_filter)

            # Confidence range
            if min_confidence is not None:
                conditions.append("ah.max_prob >= ?")
                params.append(min_confidence)
            if max_confidence is not None:
                conditions.append("ah.max_prob <= ?")
                params.append(max_confidence)

            # Build WHERE clause
            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY ah.timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def search_bookmarks(
        self, search_term: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search bookmarks by incident text or notes."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM bookmarks
                WHERE incident_text LIKE ? OR note LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (f"%{search_term}%", f"%{search_term}%", limit),
            )

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def search_notes(self, search_term: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Search notes by content."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT n.*, ah.incident_text, ah.final_label, ah.timestamp
                FROM notes n
                LEFT JOIN analysis_history ah ON n.analysis_id = ah.id
                WHERE n.note_text LIKE ?
                ORDER BY n.created_at DESC
                LIMIT ?
            """,
                (f"%{search_term}%", limit),
            )

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    # Bookmark Methods

    def add_bookmark(
        self,
        incident_text: str,
        final_label: Optional[str] = None,
        note: Optional[str] = None,
        analysis_id: Optional[int] = None,
    ) -> int:
        """Add a bookmark."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO bookmarks (analysis_id, incident_text, final_label, note)
                VALUES (?, ?, ?, ?)
            """,
                (analysis_id, incident_text, final_label, note),
            )

            return cursor.lastrowid

    def get_bookmarks(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all bookmarks."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM bookmarks
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (limit,),
            )

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def delete_bookmark(self, bookmark_id: int):
        """Delete a bookmark."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM bookmarks WHERE id = ?", (bookmark_id,))

    def update_bookmark_note(self, bookmark_id: int, note: str):
        """Update the note for a bookmark."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE bookmarks SET note = ? WHERE id = ?", (note, bookmark_id)
            )

    # Tag Methods

    def create_tag(self, name: str, color: Optional[str] = None) -> int:
        """Create a new tag."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            try:
                cursor.execute(
                    """
                    INSERT INTO tags (name, color)
                    VALUES (?, ?)
                """,
                    (name, color),
                )
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                # Tag already exists, get its ID
                cursor.execute("SELECT id FROM tags WHERE name = ?", (name,))
                row = cursor.fetchone()
                return row[0] if row else None

    def get_all_tags(self) -> List[Dict[str, Any]]:
        """Get all tags."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tags ORDER BY name")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def delete_tag(self, tag_id: int):
        """
        Delete a tag and all its associations.

        Args:
            tag_id: ID of the tag to delete
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # First delete all associations
            cursor.execute(
                "DELETE FROM analysis_tags WHERE tag_id = ?",
                (tag_id,),
            )

            # Then delete the tag itself
            cursor.execute(
                "DELETE FROM tags WHERE id = ?",
                (tag_id,),
            )

    def add_tag_to_analysis(self, analysis_id: int, tag_id: int):
        """Associate a tag with an analysis."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            try:
                cursor.execute(
                    """
                    INSERT INTO analysis_tags (analysis_id, tag_id)
                    VALUES (?, ?)
                """,
                    (analysis_id, tag_id),
                )
            except sqlite3.IntegrityError:
                pass  # Tag already associated

    def remove_tag_from_analysis(self, analysis_id: int, tag_id: int):
        """Remove a tag from an analysis."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM analysis_tags 
                WHERE analysis_id = ? AND tag_id = ?
            """,
                (analysis_id, tag_id),
            )

    def remove_all_tags_from_analysis(self, analysis_id: int):
        """Remove all tags from an analysis."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM analysis_tags 
                WHERE analysis_id = ?
            """,
                (analysis_id,),
            )

    def get_tags_for_analysis(self, analysis_id: int) -> List[Dict[str, Any]]:
        """Get tags for a specific analysis."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT t.* FROM tags t
                JOIN analysis_tags at ON t.id = at.tag_id
                WHERE at.analysis_id = ?
            """,
                (analysis_id,),
            )

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    # Notes Methods

    def add_note(
        self,
        note_text: str,
        analysis_id: Optional[int] = None,
        author: Optional[str] = None,
    ) -> int:
        """Add a note, optionally linked to an analysis."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO notes (analysis_id, note_text, author)
                VALUES (?, ?, ?)
            """,
                (analysis_id, note_text, author),
            )

            return cursor.lastrowid

    def get_notes_for_analysis(self, analysis_id: int) -> List[Dict[str, Any]]:
        """Get notes for a specific analysis."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM notes
                WHERE analysis_id = ?
                ORDER BY created_at DESC
            """,
                (analysis_id,),
            )

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def update_note(self, note_id: int, note_text: str):
        """Update an existing note."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE notes
                SET note_text = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (note_text, note_id),
            )

    def get_all_notes(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get all notes across all analyses."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM notes
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (limit,),
            )

            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    # Settings Methods

    def save_setting(self, key: str, value: Any):
        """Save a user setting."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            value_json = json.dumps(value)

            cursor.execute(
                """
                INSERT OR REPLACE INTO user_settings (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
                (key, value_json),
            )

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a user setting."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT value FROM user_settings WHERE key = ?
            """,
                (key,),
            )

            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return default

    def get_all_settings(self) -> Dict[str, Any]:
        """Get all user settings."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT key, value FROM user_settings")
            rows = cursor.fetchall()

            return {row[0]: json.loads(row[1]) for row in rows}

    # Profile Methods

    def create_profile(
        self,
        name: str,
        role: str = "Analyst",
        email: Optional[str] = None,
        default_difficulty: str = "default",
        default_threshold: float = 0.5,
        default_max_classes: int = 5,
        enable_llm: bool = False,
        enable_advanced_viz: bool = True,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Create a user profile."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Merge role and email into preferences
            prefs = preferences or {}
            prefs["role"] = role
            if email:
                prefs["email"] = email

            prefs_json = json.dumps(prefs)

            cursor.execute(
                """
                INSERT INTO user_profiles 
                (name, default_difficulty, default_threshold, default_max_classes,
                 enable_llm, enable_advanced_viz, preferences)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    name,
                    default_difficulty,
                    default_threshold,
                    default_max_classes,
                    int(enable_llm),
                    int(enable_advanced_viz),
                    prefs_json,
                ),
            )

            return cursor.lastrowid

    def get_profile(self, profile_id: int) -> Optional[Dict[str, Any]]:
        """Get a profile by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM user_profiles WHERE id = ?
            """,
                (profile_id,),
            )

            row = cursor.fetchone()
            return dict(row) if row else None

    def get_active_profile(self) -> Optional[Dict[str, Any]]:
        """Get the currently active profile."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM user_profiles WHERE is_active = 1 LIMIT 1
            """
            )

            row = cursor.fetchone()
            return dict(row) if row else None

    def set_active_profile(self, profile_id: int):
        """Set a profile as active."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Deactivate all profiles
            cursor.execute("UPDATE user_profiles SET is_active = 0")

            # Activate selected profile
            cursor.execute(
                """
                UPDATE user_profiles SET is_active = 1, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (profile_id,),
            )

    def get_all_profiles(self) -> List[Dict[str, Any]]:
        """Get all user profiles."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT id, name, default_difficulty, default_threshold,
                       default_max_classes, enable_llm, enable_advanced_viz,
                       preferences, is_active, created_at, updated_at
                FROM user_profiles
                ORDER BY is_active DESC, name ASC
            """
            )

            rows = cursor.fetchall()
            profiles = []
            for row in rows:
                profile = dict(row)
                # Parse preferences JSON if present
                if profile.get("preferences"):
                    try:
                        profile["preferences"] = json.loads(profile["preferences"])
                    except:
                        profile["preferences"] = {}
                # Add role and email from preferences if they exist
                prefs = profile.get("preferences", {})
                profile["role"] = prefs.get("role", "Analyst")
                profile["email"] = prefs.get("email")
                profiles.append(profile)

            return profiles

    def update_profile(
        self,
        profile_id: int,
        name: Optional[str] = None,
        role: Optional[str] = None,
        email: Optional[str] = None,
        default_difficulty: Optional[str] = None,
        default_threshold: Optional[float] = None,
        default_max_classes: Optional[int] = None,
        enable_llm: Optional[bool] = None,
        enable_advanced_viz: Optional[bool] = None,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update an existing profile."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Get current profile
            cursor.execute("SELECT * FROM user_profiles WHERE id = ?", (profile_id,))
            row = cursor.fetchone()
            if not row:
                return False

            current_profile = dict(row)
            current_prefs = {}
            if current_profile.get("preferences"):
                try:
                    current_prefs = json.loads(current_profile["preferences"])
                except:
                    pass

            # Update preferences
            updated_prefs = preferences if preferences is not None else current_prefs
            if role is not None:
                updated_prefs["role"] = role
            if email is not None:
                updated_prefs["email"] = email

            prefs_json = json.dumps(updated_prefs)

            # Build update query
            updates = []
            params = []

            if name is not None:
                updates.append("name = ?")
                params.append(name)
            if default_difficulty is not None:
                updates.append("default_difficulty = ?")
                params.append(default_difficulty)
            if default_threshold is not None:
                updates.append("default_threshold = ?")
                params.append(default_threshold)
            if default_max_classes is not None:
                updates.append("default_max_classes = ?")
                params.append(default_max_classes)
            if enable_llm is not None:
                updates.append("enable_llm = ?")
                params.append(int(enable_llm))
            if enable_advanced_viz is not None:
                updates.append("enable_advanced_viz = ?")
                params.append(int(enable_advanced_viz))

            updates.append("preferences = ?")
            params.append(prefs_json)
            updates.append("updated_at = CURRENT_TIMESTAMP")

            params.append(profile_id)

            query = f"UPDATE user_profiles SET {', '.join(updates)} WHERE id = ?"
            cursor.execute(query, params)

            return cursor.rowcount > 0

    def delete_profile(self, profile_id: int) -> bool:
        """Delete a profile (cannot delete active profile)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Check if profile is active
            cursor.execute(
                "SELECT is_active FROM user_profiles WHERE id = ?", (profile_id,)
            )
            row = cursor.fetchone()
            if not row:
                return False

            if row["is_active"]:
                raise ValueError("Cannot delete the active profile")

            cursor.execute("DELETE FROM user_profiles WHERE id = ?", (profile_id,))

            return cursor.rowcount > 0

    def set_setting(self, key: str, value: Any):
        """Set a user setting (creates or updates)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            value_json = json.dumps(value)

            cursor.execute(
                """
                INSERT OR REPLACE INTO user_settings (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
                (key, value_json),
            )

    # Feature Flags Methods

    def set_feature_flag(self, flag_name: str, enabled: bool, description: str = ""):
        """Set a feature flag."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO feature_flags (flag_name, enabled, description, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (flag_name, int(enabled), description),
            )

    def is_feature_enabled(self, flag_name: str, default: bool = False) -> bool:
        """Check if a feature is enabled."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT enabled FROM feature_flags WHERE flag_name = ?
            """,
                (flag_name,),
            )

            row = cursor.fetchone()
            if row:
                return bool(row[0])
            return default

    def get_all_feature_flags(self) -> Dict[str, bool]:
        """Get all feature flags."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT flag_name, enabled FROM feature_flags")
            rows = cursor.fetchall()

            return {row[0]: bool(row[1]) for row in rows}

    # Search Helper Methods

    def get_search_facets(self) -> Dict[str, Any]:
        """
        Get available filter options with counts for search UI.

        Returns dictionary with:
        - classifications: List of (label, count) tuples
        - date_range: (earliest, latest) dates
        - confidence_range: (min, max) confidence
        - total_incidents: Total count
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Get classification counts
            cursor.execute(
                """
                SELECT final_label, COUNT(*) as count
                FROM analysis_history
                GROUP BY final_label
                ORDER BY count DESC
            """
            )
            classifications = [(row[0], row[1]) for row in cursor.fetchall()]

            # Get date range
            cursor.execute(
                """
                SELECT MIN(timestamp), MAX(timestamp)
                FROM analysis_history
            """
            )
            date_row = cursor.fetchone()
            date_range = (date_row[0], date_row[1]) if date_row else (None, None)

            # Get confidence range
            cursor.execute(
                """
                SELECT MIN(max_prob), MAX(max_prob)
                FROM analysis_history
            """
            )
            conf_row = cursor.fetchone()
            confidence_range = (conf_row[0], conf_row[1]) if conf_row else (0.0, 1.0)

            # Get total count
            cursor.execute("SELECT COUNT(*) FROM analysis_history")
            total_incidents = cursor.fetchone()[0]

            return {
                "classifications": classifications,
                "date_range": date_range,
                "confidence_range": confidence_range,
                "total_incidents": total_incidents,
            }

    def count_incidents(
        self,
        classification: Optional[str] = None,
        days: Optional[int] = None,
    ) -> int:
        """
        Count incidents with optional filters.

        Args:
            classification: Filter by classification
            days: Only count incidents from last N days
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT COUNT(*) FROM analysis_history WHERE 1=1"
            params = []

            if classification:
                query += " AND final_label = ?"
                params.append(classification)

            if days:
                from datetime import datetime, timedelta

                cutoff = (datetime.now() - timedelta(days=days)).isoformat()
                query += " AND timestamp >= ?"
                params.append(cutoff)

            cursor.execute(query, params)
            return cursor.fetchone()[0]

    def get_recent_incidents(
        self, limit: int = 10, classification: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get most recent incidents."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM analysis_history"
            params = []

            if classification:
                query += " WHERE final_label = ?"
                params.append(classification)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def execute_custom_query(
        self, sql_query: str, read_only: bool = True
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """
        Execute a custom SQL query (for power users).

        Args:
            sql_query: The SQL query to execute
            read_only: If True, only SELECT queries are allowed

        Returns:
            Tuple of (results as list of dicts, column names)
        """
        # Safety check - only allow SELECT queries in read-only mode
        if read_only:
            import re

            # Remove SQL comments (both -- and /* */ style) before validation
            # Remove single-line comments (-- ...)
            query_no_comments = re.sub(r"--[^\n]*", "", sql_query)
            # Remove multi-line comments (/* ... */)
            query_no_comments = re.sub(
                r"/\*.*?\*/", "", query_no_comments, flags=re.DOTALL
            )

            query_upper = query_no_comments.strip().upper()
            if not query_upper.startswith("SELECT"):
                raise ValueError("Only SELECT queries are allowed in read-only mode")

            # Prevent dangerous operations - use word boundaries to avoid false positives
            # Check for these keywords as separate words, not within column names like "created_at"
            dangerous_keywords = [
                "DROP",
                "DELETE",
                "UPDATE",
                "INSERT",
                "ALTER",
                "CREATE",
                "TRUNCATE",
                "EXEC",
                "EXECUTE",
            ]
            for keyword in dangerous_keywords:
                # Use word boundary regex to match keyword as a complete word
                # This prevents matching "CREATE" in "created_at" or "UPDATE" in "updated_at"
                pattern = r"\b" + keyword + r"\b"
                if re.search(pattern, query_upper):
                    raise ValueError(f"Query contains forbidden keyword: {keyword}")

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql_query)

            # Get column names
            column_names = (
                [description[0] for description in cursor.description]
                if cursor.description
                else []
            )

            # Fetch results
            rows = cursor.fetchall()

            # Convert to list of dicts
            results = []
            for row in rows:
                row_dict = {}
                for idx, col_name in enumerate(column_names):
                    row_dict[col_name] = row[idx]
                results.append(row_dict)

            return results, column_names
