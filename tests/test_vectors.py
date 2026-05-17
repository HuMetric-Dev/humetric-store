from __future__ import annotations

import numpy as np
import psycopg
from humetric_core import Person

from humetric_store import (
    VectorIndex,
    VectorShapeMismatch,
    load_vector_index,
    upsert_person,
)
from humetric_store.db import VECTOR_DIMS


def _seed(conn: psycopg.Connection, ids: list[str]) -> None:
    for i in ids:
        upsert_person(conn, Person(id=i, source="github", name=i)).unwrap()


def _onehot(dim: int, idx: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = 1.0
    return v


def _twohot(dim: int, i: int, j: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    v[j] = 1.0
    return v


def test_add_batch_and_search_roundtrip(pg_conn: psycopg.Connection) -> None:
    dim = VECTOR_DIMS["graph"]
    _seed(pg_conn, ["a", "b", "c"])
    idx = VectorIndex(pg_conn, dim=dim, kind="graph")

    items = [
        ("a", _onehot(dim, 0)),
        ("b", _onehot(dim, 1)),
        ("c", _twohot(dim, 0, 1)),
    ]
    assert idx.add_batch(items).unwrap() == 3
    assert idx.size == 3

    results = idx.search(_onehot(dim, 0), k=2).unwrap()
    top_pids = [pid for pid, _ in results]
    assert top_pids[0] == "a"
    assert "c" in top_pids


def test_search_empty_index_returns_empty(pg_conn: psycopg.Connection) -> None:
    idx = VectorIndex(pg_conn, dim=VECTOR_DIMS["graph"], kind="graph")
    r = idx.search(_onehot(VECTOR_DIMS["graph"], 0), k=5).unwrap()
    assert r == []


def test_shape_mismatch_returns_err(pg_conn: psycopg.Connection) -> None:
    _seed(pg_conn, ["a"])
    idx = VectorIndex(pg_conn, dim=VECTOR_DIMS["graph"], kind="graph")
    bad = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    r = idx.add_batch([("a", bad)])
    assert r.is_err()
    assert isinstance(r.err(), VectorShapeMismatch)


def test_load_vector_index_returns_view(pg_conn: psycopg.Connection) -> None:
    _seed(pg_conn, ["a", "b"])
    idx = VectorIndex(pg_conn, dim=VECTOR_DIMS["graph"], kind="graph")
    idx.add_batch(
        [
            ("a", _onehot(VECTOR_DIMS["graph"], 0)),
            ("b", _onehot(VECTOR_DIMS["graph"], 1)),
        ]
    ).unwrap()

    loaded = load_vector_index(pg_conn, "graph").unwrap()
    assert loaded.dim == VECTOR_DIMS["graph"]
    assert loaded.kind == "graph"
    assert loaded.table == "persons"
    assert loaded.size == 2
    results = loaded.search(_onehot(VECTOR_DIMS["graph"], 0), k=1).unwrap()
    assert results[0][0] == "a"


def test_vector_index_against_organizations_table(pg_conn: psycopg.Connection) -> None:
    from humetric_core import Organization

    from humetric_store import upsert_organization

    dim = VECTOR_DIMS["graph"]
    for oid in ("o:gh:a", "o:gh:b"):
        upsert_organization(
            pg_conn,
            Organization(id=oid, source="github", name=oid, org_kind="company"),
        ).unwrap()

    idx = load_vector_index(pg_conn, "graph", table="organizations").unwrap()
    idx.add_batch(
        [
            ("o:gh:a", _onehot(dim, 0)),
            ("o:gh:b", _onehot(dim, 1)),
        ]
    ).unwrap()
    assert idx.size == 2
    results = idx.search(_onehot(dim, 0), k=1).unwrap()
    assert results[0][0] == "o:gh:a"


def test_load_vector_index_rejects_unknown_table(pg_conn: psycopg.Connection) -> None:
    from humetric_store import NotFound

    r = load_vector_index(pg_conn, "graph", table="bogus")
    assert r.is_err()
    err = r.err()
    assert isinstance(err, NotFound)
    assert err.kind == "entity_table"
