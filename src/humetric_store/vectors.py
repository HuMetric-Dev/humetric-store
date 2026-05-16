from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any, Final, cast

import faiss
import numpy as np
from humetric_core import Err, Ok, Result

from humetric_store.errors import (
    DbReadFailed,
    DbWriteFailed,
    FaissReadFailed,
    FaissWriteFailed,
    StoreError,
    VectorShapeMismatch,
)

__all__ = ["VectorIndex", "load_vector_index"]

_F32: Final = np.float32


def _stable_int64_id(person_id: str) -> int:
    """Hash a person id to a stable int64 used as the FAISS row id.

    SHA-1 truncated to 63 bits (positive int64).
    """
    h = hashlib.sha1(person_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") & 0x7FFFFFFFFFFFFFFF


class VectorIndex:
    """Cosine-similarity FAISS index over L2-normalized embeddings, keyed by
    a stable int64 hash of `person_id`. SQLite stores the reverse map plus
    the raw vector (for joins) under (person_id, kind).

    The FAISS-id -> person-id reverse map is kept in-memory and updated on
    every `add_batch`, so `search` never has to scan the SQLite `vectors`
    table. On `load_vector_index` we repopulate it once from SQLite — that
    one-time scan is the price for not paying it on every query.
    """

    def __init__(self, conn: sqlite3.Connection, dim: int, kind: str) -> None:
        self._conn = conn
        self._dim = dim
        self._kind = kind
        # faiss is a SWIG binding; its real Python-level shape doesn't match the
        # generated stubs ty sees. Pin the index as Any at this single boundary.
        self._index: Any = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self._reverse: dict[int, str] = {}

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def kind(self) -> str:
        return self._kind

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

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

        raw_ids = [_stable_int64_id(pid) for pid, _ in items]
        ids = np.array(raw_ids, dtype=np.int64)
        mat = np.stack([v.astype(_F32, copy=False) for _, v in items])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms

        try:
            self._index.add_with_ids(mat, ids)
        except RuntimeError as e:
            return Err(FaissWriteFailed(path="<memory>", reason=str(e)))

        try:
            with self._conn:
                self._conn.executemany(
                    """
                    INSERT INTO vectors (person_id, kind, vec) VALUES (?, ?, ?)
                    ON CONFLICT(person_id, kind) DO UPDATE SET vec = excluded.vec
                    """,
                    [(pid, self._kind, mat[i].tobytes()) for i, (pid, _) in enumerate(items)],
                )
        except sqlite3.Error as e:
            return Err(DbWriteFailed(table="vectors", reason=str(e)))

        for raw_id, (pid, _) in zip(raw_ids, items, strict=True):
            self._reverse[raw_id] = pid

        return Ok(len(items))

    def search(self, query: np.ndarray, k: int) -> Result[list[tuple[str, float]], StoreError]:
        if query.shape != (self._dim,):
            return Err(
                VectorShapeMismatch(
                    expected_shape=(self._dim,),
                    got_shape=tuple(int(x) for x in query.shape),
                )
            )
        if self._index.ntotal == 0:
            return Ok([])

        q = query.astype(_F32, copy=False).reshape(1, -1)
        nrm = float(np.linalg.norm(q))
        if nrm == 0:
            return Ok([])
        q = q / nrm

        try:
            scores, ids = self._index.search(q, min(k, self._index.ntotal))
        except RuntimeError as e:
            return Err(FaissReadFailed(path="<memory>", reason=str(e)))

        out: list[tuple[str, float]] = []
        for raw_id, score in zip(ids[0].tolist(), scores[0].tolist(), strict=True):
            if raw_id == -1:
                continue
            pid = self._reverse.get(int(raw_id))
            if pid is not None:
                out.append((pid, float(score)))
        return Ok(out)

    def save(self, path: str | Path) -> Result[None, StoreError]:
        try:
            faiss.write_index(self._index, str(path))
        except RuntimeError as e:
            return Err(FaissWriteFailed(path=str(path), reason=str(e)))
        return Ok(None)

    def _replace_index(self, idx: Any) -> None:
        self._index = idx


def load_vector_index(
    conn: sqlite3.Connection, path: str | Path, kind: str
) -> Result[VectorIndex, StoreError]:
    p = str(path)
    try:
        idx = cast(Any, faiss.read_index(p))
    except RuntimeError as e:
        return Err(FaissReadFailed(path=p, reason=str(e)))
    out = VectorIndex(conn, dim=int(idx.d), kind=kind)
    out._replace_index(idx)

    # One-time scan of the SQLite reverse map. Amortized against the lifetime
    # of the process; `search` then runs in pure-memory dict lookups.
    try:
        rows = conn.execute("SELECT person_id FROM vectors WHERE kind = ?", (kind,)).fetchall()
    except sqlite3.Error as e:
        return Err(DbReadFailed(table="vectors", reason=str(e)))
    out._reverse = {_stable_int64_id(r["person_id"]): r["person_id"] for r in rows}

    return Ok(out)
