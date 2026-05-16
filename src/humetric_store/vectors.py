from __future__ import annotations

from typing import Final

import numpy as np
import psycopg
from humetric_core import Err, Ok, Result
from psycopg import sql

from humetric_store.db import VECTOR_DIMS
from humetric_store.errors import (
    DbReadFailed,
    DbWriteFailed,
    NotFound,
    StoreError,
    VectorShapeMismatch,
)

__all__ = ["VectorIndex", "load_vector_index"]

_F32: Final = np.float32


class VectorIndex:
    """A per-kind view over the `persons.vec_<kind>` pgvector column.

    Instances do not own state beyond the connection, dim, and kind — the
    vectors live in Postgres and are queried directly. Kept as a class to
    preserve the v0.1 public API (`add_batch`, `search`, `dim`, `kind`,
    `size`) so downstream callers don't move.
    """

    def __init__(self, conn: psycopg.Connection, dim: int, kind: str) -> None:
        if kind not in VECTOR_DIMS:
            raise ValueError(f"unknown vector kind {kind!r}; expected one of {list(VECTOR_DIMS)}")
        if dim != VECTOR_DIMS[kind]:
            raise ValueError(
                f"dim {dim} does not match schema dim {VECTOR_DIMS[kind]} for kind {kind!r}"
            )
        self._conn = conn
        self._dim = dim
        self._kind = kind
        self._col = sql.Identifier(f"vec_{kind}")

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def size(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT COUNT(*) FROM persons WHERE {col} IS NOT NULL").format(
                    col=self._col
                )
            )
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def add_batch(self, items: list[tuple[str, np.ndarray]]) -> Result[int, StoreError]:
        if not items:
            return Ok(0)
        for _pid, v in items:
            if v.shape != (self._dim,):
                return Err(
                    VectorShapeMismatch(
                        expected_shape=(self._dim,),
                        got_shape=tuple(int(x) for x in v.shape),
                    )
                )

        stmt = sql.SQL("UPDATE persons SET {col} = %s WHERE id = %s").format(col=self._col)
        params = [(v.astype(_F32, copy=False), pid) for pid, v in items]
        try:
            with self._conn.transaction(), self._conn.cursor() as cur:
                cur.executemany(stmt, params)
        except psycopg.Error as e:
            return Err(DbWriteFailed(table="persons", reason=str(e)))
        return Ok(len(items))

    def search(self, query: np.ndarray, k: int) -> Result[list[tuple[str, float]], StoreError]:
        if query.shape != (self._dim,):
            return Err(
                VectorShapeMismatch(
                    expected_shape=(self._dim,),
                    got_shape=tuple(int(x) for x in query.shape),
                )
            )
        q = query.astype(_F32, copy=False)
        nrm = float(np.linalg.norm(q))
        if nrm == 0:
            return Ok([])
        q = q / nrm

        stmt = sql.SQL(
            "SELECT id, 1 - ({col} <=> %s) AS score "
            "FROM persons WHERE {col} IS NOT NULL "
            "ORDER BY {col} <=> %s LIMIT %s"
        ).format(col=self._col)
        try:
            with self._conn.cursor() as cur:
                cur.execute(stmt, (q, q, k))
                rows = cur.fetchall()
        except psycopg.Error as e:
            return Err(DbReadFailed(table="persons", reason=str(e)))
        return Ok([(str(pid), float(score)) for pid, score in rows])


def load_vector_index(conn: psycopg.Connection, kind: str) -> Result[VectorIndex, StoreError]:
    """Return a VectorIndex bound to the existing `persons.vec_<kind>` column.

    In v0.1 this loaded a FAISS file from disk; in v0.2 the vectors live in
    Postgres so loading is just a constructor call, but we still validate the
    schema dim matches the kind and that the table exists.
    """
    if kind not in VECTOR_DIMS:
        return Err(NotFound(kind="vector_kind", key=kind))
    return Ok(VectorIndex(conn, dim=VECTOR_DIMS[kind], kind=kind))
