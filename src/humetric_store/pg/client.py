from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector


def connect(dsn: str = "postgresql://postgres:postgres@127.0.0.1:5433/humetric") -> psycopg.Connection:
    conn = psycopg.connect(dsn, autocommit=False)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn
