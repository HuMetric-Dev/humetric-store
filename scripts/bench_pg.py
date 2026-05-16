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
import psycopg

from humetric_store.pg import connect


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


def _random_query_vec(conn: sqlite3.Connection, kind: str) -> np.ndarray:
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
    sqlite_conn: sqlite3.Connection, pg: psycopg.Connection, kind: str, dim: int, k: int = 10
) -> None:
    print(f"\n[vector search, kind={kind}, dim={dim}, k={k}]")

    queries = [_random_query_vec(sqlite_conn, kind) for _ in range(25)]
    q_iter = iter(queries)

    def hnsw_one() -> Any:
        nonlocal q_iter
        try:
            q = next(q_iter)
        except StopIteration:
            q_iter = iter(queries)
            q = next(q_iter)
        with pg.cursor() as cur:
            cur.execute(
                f"SELECT id, vec_{kind} <=> %s AS dist FROM person "
                f"WHERE vec_{kind} IS NOT NULL ORDER BY dist LIMIT %s",
                (q, k),
            )
            return cur.fetchall()
    _bench("pgvector HNSW kNN", hnsw_one)


def bench_graph_1hop(sqlite_conn: sqlite3.Connection, pg: psycopg.Connection, person_id: str) -> None:
    print(f"\n[1-hop followers of {person_id}]")

    def pg_one() -> Any:
        with pg.cursor() as cur:
            cur.execute("SELECT dst FROM edge WHERE src = %s AND kind = 'follow'", (person_id,))
            return cur.fetchall()
    _bench("pgvector 1-hop", pg_one)


def bench_graph_2hop(sqlite_conn: sqlite3.Connection, pg: psycopg.Connection, person_id: str) -> None:
    print(f"\n[2-hop follow-of-follow of {person_id}]")

    def pg_one() -> Any:
        with pg.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT e2.dst FROM edge e1
                JOIN edge e2 ON e2.src = e1.dst
                WHERE e1.src = %s AND e1.kind = 'follow' AND e2.kind = 'follow'
                """,
                (person_id,),
            )
            return cur.fetchall()
    _bench("pgvector 2-hop (self-join)", pg_one)


def bench_mutual_follows(sqlite_conn: sqlite3.Connection, pg: psycopg.Connection, a: str, b: str) -> None:
    print(f"\n[mutual follows between {a} and {b}]")

    def pg_one() -> Any:
        with pg.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) FROM edge e1
                JOIN edge e2 ON e2.dst = e1.dst
                WHERE e1.src = %s AND e2.src = %s AND e1.kind = 'follow' AND e2.kind = 'follow'
                """,
                (a, b),
            )
            return cur.fetchone()
    _bench("pgvector mutual-follow count", pg_one)


def bench_hybrid_lookup(sqlite_conn: sqlite3.Connection, pg: psycopg.Connection, person_id: str) -> None:
    print(f"\n[hybrid: person + skills + 1-hop count for {person_id}]")

    def pg_separate() -> Any:
        with pg.cursor() as cur:
            cur.execute("SELECT * FROM person WHERE id = %s", (person_id,))
            p = cur.fetchone()
            cur.execute("SELECT skill_normalized FROM has_skill WHERE person_id = %s", (person_id,))
            s = cur.fetchall()
            cur.execute("SELECT COUNT(*) FROM edge WHERE src = %s AND kind = 'follow'", (person_id,))
            c = cur.fetchone()
        return (p, s, c)
    _bench("pgvector hybrid (3 queries)", pg_separate)

    def pg_combined() -> Any:
        with pg.cursor() as cur:
            cur.execute(
                """
                SELECT
                    p.*,
                    (SELECT array_agg(skill_normalized) FROM has_skill WHERE person_id = p.id) AS skills,
                    (SELECT COUNT(*) FROM edge WHERE src = p.id AND kind = 'follow') AS follow_count
                FROM person p WHERE p.id = %s
                """,
                (person_id,),
            )
            return cur.fetchone()
    _bench("pgvector hybrid (1 query)", pg_combined)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sqlite", type=Path, default=Path("../humetric-cli/data/humetric.db"))
    args = p.parse_args()

    sqlite_conn = sqlite3.connect(args.sqlite)
    sqlite_conn.row_factory = sqlite3.Row
    pg = connect()

    busy = _pick_busy_person(sqlite_conn)
    print(f"busy person (most follows): {busy}")
    other_row = sqlite_conn.execute(
        "SELECT src FROM edges WHERE kind = 'follow' GROUP BY src ORDER BY COUNT(*) DESC LIMIT 1 OFFSET 5"
    ).fetchone()
    other = other_row[0]
    print(f"other person: {other}")

    bench_vector_search(sqlite_conn, pg, "text", 1024, k=10)
    bench_vector_search(sqlite_conn, pg, "graph", 128, k=10)
    bench_vector_search(sqlite_conn, pg, "tower", 256, k=10)
    bench_graph_1hop(sqlite_conn, pg, busy)
    bench_graph_2hop(sqlite_conn, pg, busy)
    bench_mutual_follows(sqlite_conn, pg, busy, other)
    bench_hybrid_lookup(sqlite_conn, pg, busy)

    return 0


if __name__ == "__main__":
    sys.exit(main())
