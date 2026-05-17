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

ENTITY_TABLES: tuple[str, ...] = ("persons", "organizations")

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
        CREATE TABLE IF NOT EXISTS organizations (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            name TEXT NOT NULL,
            org_kind TEXT NOT NULL,
            headline TEXT NOT NULL DEFAULT '',
            about TEXT NOT NULL DEFAULT '',
            location TEXT NOT NULL DEFAULT '',
            hq_location TEXT NOT NULL DEFAULT '',
            founding_year INT,
            employee_count INT,
            raw_url TEXT NOT NULL DEFAULT '',
            vec_text vector(1024),
            vec_graph vector(128),
            vec_tower vector(256)
        )
    """),
    sql.SQL("CREATE INDEX IF NOT EXISTS idx_organizations_org_kind ON organizations(org_kind)"),
    sql.SQL("""
        CREATE TABLE IF NOT EXISTS org_industries (
            org_id TEXT NOT NULL,
            industry TEXT NOT NULL,
            PRIMARY KEY (org_id, industry)
        )
    """),
    sql.SQL("CREATE INDEX IF NOT EXISTS idx_org_industries_industry ON org_industries(industry)"),
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
            src_type TEXT,
            dst_type TEXT,
            PRIMARY KEY (src, dst, kind)
        )
    """),
    sql.SQL("CREATE INDEX IF NOT EXISTS idx_edges_src_kind ON edges(src, kind)"),
    sql.SQL("CREATE INDEX IF NOT EXISTS idx_edges_dst_kind ON edges(dst, kind)"),
    # Belt-and-suspenders for DBs created before the src_type/dst_type columns
    # existed. ADD COLUMN IF NOT EXISTS is idempotent on Postgres >= 9.6.
    sql.SQL("ALTER TABLE edges ADD COLUMN IF NOT EXISTS src_type TEXT"),
    sql.SQL("ALTER TABLE edges ADD COLUMN IF NOT EXISTS dst_type TEXT"),
    sql.SQL("""
        CREATE OR REPLACE VIEW entities AS
        SELECT id, 'person'::TEXT AS entity_type, source, name, location FROM persons
        UNION ALL
        SELECT id, 'organization'::TEXT AS entity_type, source, name, location
        FROM organizations
    """),
    # ------------------------------------------------------------------
    # Auth + per-user history tables. App users (the recruiters running
    # queries) are NOT the same thing as `persons` rows (the corpus being
    # searched). Each `users` row optionally claims one `persons.id` as
    # its "self" persona for personalization; the link is nullable so a
    # user can sign in immediately even when claim resolution is pending.
    # ------------------------------------------------------------------
    sql.SQL("ALTER TABLE persons ADD COLUMN IF NOT EXISTS github_username TEXT"),
    sql.SQL(
        "CREATE INDEX IF NOT EXISTS idx_persons_github_username ON persons(LOWER(github_username))"
    ),
    sql.SQL("CREATE INDEX IF NOT EXISTS idx_persons_name_lower ON persons(LOWER(name))"),
    sql.SQL("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL,
            person_id TEXT REFERENCES persons(id) ON DELETE SET NULL,
            created_at DOUBLE PRECISION NOT NULL,
            is_active BOOLEAN NOT NULL DEFAULT TRUE
        )
    """),
    sql.SQL(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_users_person_id_not_null "
        "ON users(person_id) WHERE person_id IS NOT NULL"
    ),
    sql.SQL("""
        CREATE TABLE IF NOT EXISTS auth_credentials (
            user_id TEXT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
            password_hash TEXT NOT NULL
        )
    """),
    sql.SQL("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            token_hash TEXT NOT NULL UNIQUE,
            created_at DOUBLE PRECISION NOT NULL,
            expires_at DOUBLE PRECISION NOT NULL,
            last_seen_at DOUBLE PRECISION NOT NULL,
            user_agent TEXT NOT NULL DEFAULT '',
            ip TEXT NOT NULL DEFAULT ''
        )
    """),
    sql.SQL("CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)"),
    sql.SQL("CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at)"),
    sql.SQL("""
        CREATE TABLE IF NOT EXISTS query_history (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            ts DOUBLE PRECISION NOT NULL,
            parsed JSONB NOT NULL,
            embedding vector(1024) NOT NULL
        )
    """),
    sql.SQL(
        "CREATE INDEX IF NOT EXISTS idx_query_history_user_ts ON query_history(user_id, ts DESC)"
    ),
)


def _vector_indexes() -> tuple[sql.Composed, ...]:
    return tuple(
        sql.SQL(
            "CREATE INDEX IF NOT EXISTS {idx} ON {tbl} USING hnsw ({col} vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 64)"
        ).format(
            idx=sql.Identifier(f"idx_{tbl}_vec_{kind}"),
            tbl=sql.Identifier(tbl),
            col=sql.Identifier(f"vec_{kind}"),
        )
        for tbl in ENTITY_TABLES
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
