from __future__ import annotations

import psycopg
from humetric_core import Edge, Err, Ok, Result, entity_type_of

from humetric_store.errors import DbReadFailed, DbWriteFailed, StoreError


def _entity_type_or_null(s: str) -> str | None:
    r = entity_type_of(s)
    if isinstance(r, Err):
        return None
    return r.value


def upsert_edge(conn: psycopg.Connection, e: Edge) -> Result[None, StoreError]:
    try:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO edges (src, dst, kind, weight, src_type, dst_type)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (src, dst, kind) DO UPDATE SET
                    weight = EXCLUDED.weight,
                    src_type = EXCLUDED.src_type,
                    dst_type = EXCLUDED.dst_type
                """,
                (
                    e.src,
                    e.dst,
                    e.kind,
                    e.weight,
                    _entity_type_or_null(e.src),
                    _entity_type_or_null(e.dst),
                ),
            )
    except psycopg.Error as err:
        return Err(DbWriteFailed(table="edges", reason=str(err)))
    return Ok(None)


def list_edges_from(
    conn: psycopg.Connection, src: str, kind: str | None = None
) -> Result[list[Edge], StoreError]:
    try:
        with conn.cursor() as cur:
            if kind is None:
                cur.execute("SELECT src, dst, kind, weight FROM edges WHERE src = %s", (src,))
            else:
                cur.execute(
                    "SELECT src, dst, kind, weight FROM edges WHERE src = %s AND kind = %s",
                    (src, kind),
                )
            rows = cur.fetchall()
    except psycopg.Error as e:
        return Err(DbReadFailed(table="edges", reason=str(e)))
    return Ok([Edge(src=r[0], dst=r[1], kind=r[2], weight=r[3]) for r in rows])
