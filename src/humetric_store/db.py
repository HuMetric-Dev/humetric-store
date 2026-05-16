from __future__ import annotations

import psycopg
from humetric_core import Err, Ok, Result
from pgvector.psycopg import register_vector
from psycopg import sql

from humetric_store.errors import DbOpenFailed, StoreError

VECTOR_DIMS: dict[str, int] = {
    "text": 1024,
    "graph": 128,
    "tower": 256,
}

_DDL: tuple[sql.SQL, ...] = (
    sql.SQL("CREATE EXTENSION IF NOT EXISTS vector"),
    sql.SQL("""
        CREATE TABLE IF NOT EXISTS persons (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            name TEXT NOT NULL,
            headline TEXT NOT NULL DEFAULT '',
            about TEXT NOT NULL DEFAULT '',
            location TEXT NOT NULL DEFAULT '',
            follower_count INT NOT NULL DEFAULT 0,
            last_active_days_ago INT,
            raw_url TEXT NOT NULL DEFAULT '',
            vec_text vector(1024),
            vec_graph vector(128),
            vec_tower vector(256)
        )
    """),
    sql.SQL("""
        CREATE TABLE IF NOT EXISTS skills (
            name TEXT PRIMARY KEY,
            normalized TEXT NOT NULL
        )
    """),
    sql.SQL("CREATE INDEX IF NOT EXISTS idx_skills_normalized ON skills(normalized)"),
    sql.SQL("""
        CREATE TABLE IF NOT EXISTS person_skills (
            person_id TEXT NOT NULL,
            skill_name TEXT NOT NULL,
            PRIMARY KEY (person_id, skill_name)
        )
    """),
    sql.SQL("CREATE INDEX IF NOT EXISTS idx_person_skills_skill ON person_skills(skill_name)"),
    sql.SQL("""
        CREATE TABLE IF NOT EXISTS edges (
            src TEXT NOT NULL,
            dst TEXT NOT NULL,
            kind TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 1.0,
            PRIMARY KEY (src, dst, kind)
        )
    """),
    sql.SQL("CREATE INDEX IF NOT EXISTS idx_edges_src_kind ON edges(src, kind)"),
    sql.SQL("CREATE INDEX IF NOT EXISTS idx_edges_dst_kind ON edges(dst, kind)"),
)


def _vector_indexes() -> tuple[sql.Composed, ...]:
    return tuple(
        sql.SQL(
            "CREATE INDEX IF NOT EXISTS {idx} ON persons USING hnsw ({col} vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 64)"
        ).format(
            idx=sql.Identifier(f"idx_persons_vec_{kind}"),
            col=sql.Identifier(f"vec_{kind}"),
        )
        for kind in VECTOR_DIMS
    )


def open_db(dsn: str) -> Result[psycopg.Connection, StoreError]:
    """Open a connection to the Postgres store at `dsn` and run idempotent
    migrations (pgvector extension, base tables, HNSW indexes).
    """
    try:
        conn = psycopg.connect(dsn, autocommit=False)
    except psycopg.Error as e:
        return Err(DbOpenFailed(dsn=dsn, reason=str(e)))

    try:
        with conn.cursor() as cur:
            for stmt in _DDL:
                cur.execute(stmt)
            for stmt in _vector_indexes():
                cur.execute(stmt)
        conn.commit()
    except psycopg.Error as e:
        conn.close()
        return Err(DbOpenFailed(dsn=dsn, reason=f"migration failed: {e}"))

    try:
        register_vector(conn)
    except psycopg.Error as e:
        conn.close()
        return Err(DbOpenFailed(dsn=dsn, reason=f"register_vector failed: {e}"))

    return Ok(conn)
