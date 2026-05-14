from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from humetric_core import Person

from humetric_store import (
    VectorIndex,
    VectorShapeMismatch,
    open_db,
    upsert_person,
)


def _conn_with_people(ids: list[str]):
    conn = open_db(":memory:").unwrap()
    for i in ids:
        upsert_person(conn, Person(id=i, source="github", name=i)).unwrap()
    return conn


def test_add_batch_and_search_roundtrip() -> None:
    ids = ["a", "b", "c"]
    conn = _conn_with_people(ids)
    idx = VectorIndex(conn, dim=4, kind="text")

    items = [
        ("a", np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
        ("b", np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)),
        ("c", np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)),
    ]
    assert idx.add_batch(items).unwrap() == 3
    assert idx.size == 3

    results = idx.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), k=2).unwrap()
    top_pids = [pid for pid, _ in results]
    assert top_pids[0] == "a"
    assert "c" in top_pids


def test_search_empty_index_returns_empty() -> None:
    conn = _conn_with_people([])
    idx = VectorIndex(conn, dim=4, kind="text")
    r = idx.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), k=5).unwrap()
    assert r == []


def test_shape_mismatch_returns_err() -> None:
    conn = _conn_with_people(["a"])
    idx = VectorIndex(conn, dim=4, kind="text")
    bad = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    r = idx.add_batch([("a", bad)])
    assert r.is_err()
    assert isinstance(r.err(), VectorShapeMismatch)


def test_save_and_load_index() -> None:
    ids = ["a", "b"]
    conn = _conn_with_people(ids)
    idx = VectorIndex(conn, dim=3, kind="text")
    idx.add_batch(
        [
            ("a", np.array([1.0, 0.0, 0.0], dtype=np.float32)),
            ("b", np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        ]
    ).unwrap()

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "idx.faiss"
        idx.save(p).unwrap()
        loaded = VectorIndex.load(conn, p, kind="text").unwrap()
        assert loaded.size == 2
        results = loaded.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), k=1).unwrap()
        assert results[0][0] == "a"
