from __future__ import annotations

import sqlite3
from pathlib import Path

from humetric_core import Err, Ok, Result

from humetric_store.errors import DbOpenFailed, StoreError

_SCHEMA: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS persons (
        id TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        name TEXT NOT NULL,
        headline TEXT NOT NULL DEFAULT '',
        about TEXT NOT NULL DEFAULT '',
        location TEXT NOT NULL DEFAULT '',
        follower_count INTEGER NOT NULL DEFAULT 0,
        last_active_days_ago INTEGER,
        raw_url TEXT NOT NULL DEFAULT ''
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS edges (
        src TEXT NOT NULL,
        dst TEXT NOT NULL,
        kind TEXT NOT NULL,
        weight REAL NOT NULL DEFAULT 1.0,
        PRIMARY KEY (src, dst, kind)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src)",
    "CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst)",
    """
    CREATE TABLE IF NOT EXISTS skills (
        name TEXT PRIMARY KEY,
        normalized TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_skills_normalized ON skills(normalized)",
    """
    CREATE TABLE IF NOT EXISTS person_skills (
        person_id TEXT NOT NULL,
        skill_name TEXT NOT NULL,
        PRIMARY KEY (person_id, skill_name)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_person_skills_skill ON person_skills(skill_name)",
    """
    CREATE TABLE IF NOT EXISTS vectors (
        person_id TEXT NOT NULL,
        kind TEXT NOT NULL,
        vec BLOB NOT NULL,
        PRIMARY KEY (person_id, kind)
    )
    """,
)


def open_db(path: str | Path) -> Result[sqlite3.Connection, StoreError]:
    """Open or create the database at `path`, run idempotent migrations.

    `:memory:` is allowed for tests.
    """
    p = str(path)
    try:
        conn = sqlite3.connect(p)
    except sqlite3.Error as e:
        return Err(DbOpenFailed(path=p, reason=str(e)))

    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL") if p != ":memory:" else None
        for stmt in _SCHEMA:
            conn.execute(stmt)
        conn.commit()
    except sqlite3.Error as e:
        conn.close()
        return Err(DbOpenFailed(path=p, reason=f"migration failed: {e}"))

    return Ok(conn)
