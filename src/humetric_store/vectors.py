from __future__ import annotations

from typing import Final

import numpy as np
import psycopg
from humetric_core import Err, Ok, Result
from psycopg import sql

from humetric_store.db import ENTITY_TABLES, VECTOR_DIMS
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
    """A per-(table, kind) view over the `<table>.vec_<kind>` pgvector column.

    `table` is the entity table the vectors live on: "persons" or
    "organizations". One VectorIndex covers exactly one entity type; retrieval
    fuses the per-type results via RRF.
    """

    def __init__(
        self,
        conn: psycopg.Connection,
        dim: int,
        kind: str,
        table: str = "persons",
    ) -> None:
        if kind not in VECTOR_DIMS:
            raise ValueError(f"unknown vector kind {kind!r}; expected one of {list(VECTOR_DIMS)}")
        if dim != VECTOR_DIMS[kind]:
            raise ValueError(
                f"dim {dim} does not match schema dim {VECTOR_DIMS[kind]} for kind {kind!r}"
            )
        if table not in ENTITY_TABLES:
            raise ValueError(
                f"unknown entity table {table!r}; expected one of {list(ENTITY_TABLES)}"
            )
        self._conn = conn
        self._dim = dim
        self._kind = kind
        self._table = table
        self._tbl = sql.Identifier(table)
        self._col = sql.Identifier(f"vec_{kind}")

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def table(self) -> str:
        return self._table

    @property
    def size(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT COUNT(*) FROM {tbl} WHERE {col} IS NOT NULL").format(
                    tbl=self._tbl, col=self._col
                )
            )
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def add_batch(self, items: list[tuple[str, np.ndarray]]) -> Result[int, StoreError]:
        if not items:
            return Ok(0)
        for _eid, v in items:
            if v.shape != (self._dim,):
                return Err(
                    VectorShapeMismatch(
                        expected_shape=(self._dim,),
                        got_shape=tuple(int(x) for x in v.shape),
                    )
                )

        stmt = sql.SQL("UPDATE {tbl} SET {col} = %s WHERE id = %s").format(
            tbl=self._tbl, col=self._col
        )
        params = [(v.astype(_F32, copy=False), eid) for eid, v in items]
        try:
            with self._conn.transaction(), self._conn.cursor() as cur:
                cur.executemany(stmt, params)
        except psycopg.Error as e:
            return Err(DbWriteFailed(table=self._table, reason=str(e)))
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
            "FROM {tbl} WHERE {col} IS NOT NULL "
            "ORDER BY {col} <=> %s LIMIT %s"
        ).format(tbl=self._tbl, col=self._col)
        try:
            with self._conn.cursor() as cur:
                cur.execute(stmt, (q, q, k))
                rows = cur.fetchall()
        except psycopg.Error as e:
            return Err(DbReadFailed(table=self._table, reason=str(e)))
        return Ok([(str(eid), float(score)) for eid, score in rows])


def load_vector_index(
    conn: psycopg.Connection, kind: str, table: str = "persons"
) -> Result[VectorIndex, StoreError]:
    """Return a VectorIndex bound to the `<table>.vec_<kind>` column.

    Validates the kind/dim pair and that `table` is a known entity table.
    """
    if kind not in VECTOR_DIMS:
        return Err(NotFound(kind="vector_kind", key=kind))
    if table not in ENTITY_TABLES:
        return Err(NotFound(kind="entity_table", key=table))
    return Ok(VectorIndex(conn, dim=VECTOR_DIMS[kind], kind=kind, table=table))
