from __future__ import annotations

import argparse
import sqlite3
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from surrealdb import RecordID, Surreal

from humetric_store.surreal import connect
from humetric_store.vectors import load_vector_index


def _bench(name: str, fn: Callable[[], Any], iters: int = 20, warmup: int = 3) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000.0)
    times.sort()
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)] if len(times) >= 20 else max(times)
    mean = statistics.mean(times)
    print(f"  {name:40s}  p50={p50:7.2f}ms  p95={p95:7.2f}ms  mean={mean:7.2f}ms  (n={iters})")
    return {"p50": p50, "p95": p95, "mean": mean}


def _random_query_vec(conn: sqlite3.Connection, kind: str, dim: int) -> np.ndarray:
    row = conn.execute(
        "SELECT vec FROM vectors WHERE kind = ? ORDER BY RANDOM() LIMIT 1", (kind,)
    ).fetchone()
    v = np.frombuffer(row[0], dtype=np.float32).copy()
    v += np.random.normal(0, 0.01, size=v.shape).astype(np.float32)
    return v


def _pick_busy_person(conn: sqlite3.Connection, kind: str = "follow") -> str:
    row = conn.execute(
        "SELECT src FROM edges WHERE kind = ? GROUP BY src ORDER BY COUNT(*) DESC LIMIT 1",
        (kind,),
    ).fetchone()
    return row[0]


def bench_vector_search(
    sqlite_conn: sqlite3.Connection, faiss_dir: Path, db: Surreal, kind: str, dim: int, k: int = 10
) -> None:
    print(f"\n[vector search, kind={kind}, dim={dim}, k={k}]")

    faiss_file = faiss_dir / {"text": "text.faiss.bge-small", "graph": "graph.faiss", "tower": "tower.faiss"}[kind]
    if not faiss_file.exists():
        faiss_file = faiss_dir / f"{kind}.faiss"
    if not faiss_file.exists():
        print(f"  (FAISS file {faiss_file} missing, skipping baseline)")
        idx = None
    else:
        r = load_vector_index(sqlite_conn, str(faiss_file), kind)
        if hasattr(r, "value"):
            idx = r.value
        else:
            print(f"  (FAISS load failed: {r}); skipping baseline")
            idx = None

    queries = [_random_query_vec(sqlite_conn, kind, dim) for _ in range(25)]
    q_iter = iter(queries)

    if idx is not None and idx.dim == dim:
        def faiss_one() -> Any:
            nonlocal q_iter
            try:
                q = next(q_iter)
            except StopIteration:
                q_iter = iter(queries)
                q = next(q_iter)
            return idx.search(q, k)
        q_iter = iter(queries)
        _bench("SQLite+FAISS kNN", faiss_one)

    q_iter2 = iter(queries)
    def surreal_one() -> Any:
        nonlocal q_iter2
        try:
            q = next(q_iter2)
        except StopIteration:
            q_iter2 = iter(queries)
            q = next(q_iter2)
        return db.query(
            f"SELECT id, vector::distance::knn() AS dist "
            f"FROM person WHERE vec_{kind} <|{k},40|> $q ORDER BY dist;",
            {"q": q.tolist()},
        )
    _bench("SurrealDB HNSW kNN", surreal_one)

    q_iter3 = iter(queries)
    def surreal_brute() -> Any:
        nonlocal q_iter3
        try:
            q = next(q_iter3)
        except StopIteration:
            q_iter3 = iter(queries)
            q = next(q_iter3)
        return db.query(
            f"SELECT id, vector::similarity::cosine(vec_{kind}, $q) AS score "
            f"FROM person WHERE vec_{kind} IS NOT NONE ORDER BY score DESC LIMIT {k};",
            {"q": q.tolist()},
        )
    _bench("SurrealDB brute-force kNN", surreal_brute)


def bench_graph_1hop(sqlite_conn: sqlite3.Connection, db: Surreal, person_id: str) -> None:
    print(f"\n[1-hop followers of {person_id}]")

    def sqlite_one() -> Any:
        return sqlite_conn.execute(
            "SELECT dst FROM edges WHERE src = ? AND kind = 'follow'", (person_id,)
        ).fetchall()
    _bench("SQLite 1-hop", sqlite_one)

    rid = RecordID("person", person_id)
    def surreal_one() -> Any:
        return db.query("SELECT ->follow->person AS f FROM $p;", {"p": rid})
    _bench("SurrealDB 1-hop", surreal_one)


def bench_graph_2hop(sqlite_conn: sqlite3.Connection, db: Surreal, person_id: str) -> None:
    print(f"\n[2-hop follow-of-follow of {person_id}]")

    def sqlite_one() -> Any:
        return sqlite_conn.execute(
            """
            SELECT DISTINCT e2.dst FROM edges e1
            JOIN edges e2 ON e2.src = e1.dst
            WHERE e1.src = ? AND e1.kind = 'follow' AND e2.kind = 'follow'
            """,
            (person_id,),
        ).fetchall()
    _bench("SQLite 2-hop (self-join)", sqlite_one)

    rid = RecordID("person", person_id)
    def surreal_one() -> Any:
        return db.query("SELECT array::distinct(->follow->person->follow->person) AS fof FROM $p;", {"p": rid})
    _bench("SurrealDB 2-hop (native)", surreal_one)


def bench_mutual_follows(sqlite_conn: sqlite3.Connection, db: Surreal, a: str, b: str) -> None:
    print(f"\n[mutual follows between {a} and {b}]")

    def sqlite_one() -> Any:
        return sqlite_conn.execute(
            """
            SELECT COUNT(*) FROM edges e1
            JOIN edges e2 ON e2.dst = e1.dst
            WHERE e1.src = ? AND e2.src = ? AND e1.kind = 'follow' AND e2.kind = 'follow'
            """,
            (a, b),
        ).fetchone()
    _bench("SQLite mutual-follow count", sqlite_one)

    ra, rb = RecordID("person", a), RecordID("person", b)
    def surreal_one() -> Any:
        return db.query(
            "LET $fa = (SELECT VALUE ->follow->person FROM ONLY $a)[0];"
            "LET $fb = (SELECT VALUE ->follow->person FROM ONLY $b)[0];"
            "RETURN array::len(array::intersect($fa, $fb));",
            {"a": ra, "b": rb},
        )
    _bench("SurrealDB mutual-follow count", surreal_one)


def bench_hybrid_lookup(sqlite_conn: sqlite3.Connection, db: Surreal, person_id: str) -> None:
    print(f"\n[hybrid: person + skills + 1-hop count for {person_id}]")

    def sqlite_one() -> Any:
        p = sqlite_conn.execute("SELECT * FROM persons WHERE id = ?", (person_id,)).fetchone()
        s = sqlite_conn.execute(
            "SELECT skill_name FROM person_skills WHERE person_id = ?", (person_id,)
        ).fetchall()
        c = sqlite_conn.execute(
            "SELECT COUNT(*) FROM edges WHERE src = ? AND kind = 'follow'", (person_id,)
        ).fetchone()
        return (p, s, c)
    _bench("SQLite hybrid (3 queries)", sqlite_one)

    rid = RecordID("person", person_id)
    def surreal_one() -> Any:
        return db.query(
            "SELECT *, ->has_skill->skill.normalized AS skills, "
            "count(->follow->person) AS follow_count FROM $p;",
            {"p": rid},
        )
    _bench("SurrealDB hybrid (1 query)", surreal_one)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sqlite", type=Path, default=Path("../humetric-cli/data/humetric.db"))
    p.add_argument("--faiss-dir", type=Path, default=Path("../humetric-cli/data"))
    args = p.parse_args()

    sqlite_conn = sqlite3.connect(args.sqlite)
    sqlite_conn.row_factory = sqlite3.Row
    db = connect()

    busy = _pick_busy_person(sqlite_conn)
    print(f"busy person (most follows): {busy}")
    other_row = sqlite_conn.execute(
        "SELECT src FROM edges WHERE kind = 'follow' GROUP BY src ORDER BY COUNT(*) DESC LIMIT 1 OFFSET 5"
    ).fetchone()
    other = other_row[0]
    print(f"other person: {other}")

    bench_vector_search(sqlite_conn, args.faiss_dir, db, "text", 1024, k=10)
    bench_vector_search(sqlite_conn, args.faiss_dir, db, "graph", 128, k=10)
    bench_vector_search(sqlite_conn, args.faiss_dir, db, "tower", 256, k=10)
    bench_graph_1hop(sqlite_conn, db, busy)
    bench_graph_2hop(sqlite_conn, db, busy)
    bench_mutual_follows(sqlite_conn, db, busy, other)
    bench_hybrid_lookup(sqlite_conn, db, busy)

    return 0


if __name__ == "__main__":
    sys.exit(main())
