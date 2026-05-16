from __future__ import annotations

import psycopg

VECTOR_DIMS: dict[str, int] = {
    "text": 1024,
    "graph": 128,
    "tower": 256,
}

EDGE_KINDS: tuple[str, ...] = ("follow", "co_contributor")


_DDL: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS person (
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
    """,
    """
    CREATE TABLE IF NOT EXISTS skill (
        normalized TEXT PRIMARY KEY
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS has_skill (
        person_id TEXT NOT NULL,
        skill_normalized TEXT NOT NULL,
        PRIMARY KEY (person_id, skill_normalized)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_has_skill_skill ON has_skill(skill_normalized)",
    """
    CREATE TABLE IF NOT EXISTS edge (
        src TEXT NOT NULL,
        dst TEXT NOT NULL,
        kind TEXT NOT NULL,
        weight REAL NOT NULL DEFAULT 1.0,
        PRIMARY KEY (src, dst, kind)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_edge_src_kind ON edge(src, kind)",
    "CREATE INDEX IF NOT EXISTS idx_edge_dst_kind ON edge(dst, kind)",
)


def _wipe_sql() -> tuple[str, ...]:
    return (
        "DROP TABLE IF EXISTS has_skill CASCADE",
        "DROP TABLE IF EXISTS edge CASCADE",
        "DROP TABLE IF EXISTS skill CASCADE",
        "DROP TABLE IF EXISTS person CASCADE",
    )


def wipe(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        for s in _wipe_sql():
            cur.execute(s)
    conn.commit()


def apply_schema(conn: psycopg.Connection, with_vector_indexes: bool = True) -> None:
    with conn.cursor() as cur:
        for s in _DDL:
            cur.execute(s)
        if with_vector_indexes:
            for kind in VECTOR_DIMS:
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_person_vec_{kind} "
                    f"ON person USING hnsw (vec_{kind} vector_cosine_ops) "
                    f"WITH (m = 16, ef_construction = 64)"
                )
    conn.commit()
