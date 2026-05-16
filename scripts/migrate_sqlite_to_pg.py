from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import psycopg
from psycopg import sql

from humetric_store.pg import apply_schema, connect
from humetric_store.pg.schema import EDGE_KINDS, VECTOR_DIMS, wipe


def _chunks[T](items: list[T], n: int) -> Iterator[list[T]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _open_sqlite(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _migrate_persons(src: sqlite3.Connection, dst: psycopg.Connection, batch: int) -> int:
    rows = src.execute("SELECT * FROM persons").fetchall()
    with dst.cursor() as cur:
        for chunk in _chunks(rows, batch):
            cur.executemany(
                "INSERT INTO person (id, source, name, headline, about, location, "
                "follower_count, last_active_days_ago, raw_url) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    (
                        r["id"], r["source"], r["name"], r["headline"] or "",
                        r["about"] or "", r["location"] or "",
                        int(r["follower_count"]), r["last_active_days_ago"],
                        r["raw_url"] or "",
                    )
                    for r in chunk
                ],
            )
    dst.commit()
    return len(rows)


def _migrate_skills(src: sqlite3.Connection, dst: psycopg.Connection, batch: int) -> tuple[int, int]:
    skills = src.execute("SELECT DISTINCT normalized FROM skills").fetchall()
    with dst.cursor() as cur:
        cur.executemany(
            "INSERT INTO skill (normalized) VALUES (%s) ON CONFLICT DO NOTHING",
            [(r["normalized"],) for r in skills],
        )
    dst.commit()
    n_skill = len(skills)

    ps_rows = src.execute(
        """
        SELECT ps.person_id, s.normalized
        FROM person_skills ps JOIN skills s ON s.name = ps.skill_name
        """
    ).fetchall()
    with dst.cursor() as cur:
        for chunk in _chunks(ps_rows, batch):
            cur.executemany(
                "INSERT INTO has_skill (person_id, skill_normalized) VALUES (%s, %s) "
                "ON CONFLICT DO NOTHING",
                [(r["person_id"], r["normalized"]) for r in chunk],
            )
    dst.commit()
    return n_skill, len(ps_rows)


def _migrate_edges(src: sqlite3.Connection, dst: psycopg.Connection, batch: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    with dst.cursor() as cur:
        for kind in EDGE_KINDS:
            rows = src.execute(
                "SELECT src, dst, weight FROM edges WHERE kind = ?", (kind,)
            ).fetchall()
            for chunk in _chunks(rows, batch):
                cur.executemany(
                    "INSERT INTO edge (src, dst, kind, weight) VALUES (%s, %s, %s, %s)",
                    [(r["src"], r["dst"], kind, float(r["weight"])) for r in chunk],
                )
            counts[kind] = len(rows)
    dst.commit()
    return counts


def _migrate_vectors(src: sqlite3.Connection, dst: psycopg.Connection, batch: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    with dst.cursor() as cur:
        for kind, dim in VECTOR_DIMS.items():
            rows = src.execute(
                "SELECT person_id, vec FROM vectors WHERE kind = ?", (kind,)
            ).fetchall()
            stmt = sql.SQL("UPDATE person SET {col} = %s WHERE id = %s").format(
                col=sql.Identifier(f"vec_{kind}")
            )
            n = 0
            for chunk in _chunks(rows, batch):
                params = []
                for r in chunk:
                    arr = np.frombuffer(r["vec"], dtype=np.float32)
                    if arr.shape[0] != dim:
                        raise RuntimeError(f"dim mismatch {r['person_id']}/{kind}: {arr.shape}")
                    params.append((arr, r["person_id"]))
                cur.executemany(stmt, params)
                n += len(chunk)
            counts[kind] = n
    dst.commit()
    return counts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sqlite", type=Path, default=Path("../humetric-cli/data/humetric.db"))
    p.add_argument("--batch", type=int, default=1000)
    p.add_argument("--skip-vectors", action="store_true")
    p.add_argument("--skip-vector-index", action="store_true")
    args = p.parse_args()

    src = _open_sqlite(args.sqlite)
    dst = connect()
    print("connected; wiping spike DB...")
    wipe(dst)
    print(f"applying schema (vector indexes: {not args.skip_vector_index and not args.skip_vectors})...")
    # Build vector indexes AFTER vector load — much faster than incremental
    apply_schema(dst, with_vector_indexes=False)

    t = time.perf_counter()
    n = _migrate_persons(src, dst, args.batch)
    dt = time.perf_counter() - t
    print(f"persons: {n} in {dt:.2f}s ({n / dt:.0f}/s)")

    t = time.perf_counter()
    n_s, n_ps = _migrate_skills(src, dst, args.batch)
    dt = time.perf_counter() - t
    print(f"skills: {n_s} + has_skill: {n_ps} in {dt:.2f}s")

    t = time.perf_counter()
    ec = _migrate_edges(src, dst, args.batch)
    dt = time.perf_counter() - t
    print(f"edges: {ec} in {dt:.2f}s ({sum(ec.values()) / dt:.0f}/s)")

    if not args.skip_vectors:
        t = time.perf_counter()
        vc = _migrate_vectors(src, dst, args.batch)
        dt = time.perf_counter() - t
        print(f"vectors: {vc} in {dt:.2f}s ({sum(vc.values()) / dt:.0f}/s)")

        if not args.skip_vector_index:
            print("building HNSW indexes...")
            with dst.cursor() as cur:
                for kind in VECTOR_DIMS:
                    t = time.perf_counter()
                    cur.execute(
                        f"CREATE INDEX idx_person_vec_{kind} ON person "
                        f"USING hnsw (vec_{kind} vector_cosine_ops) "
                        f"WITH (m = 16, ef_construction = 64)"
                    )
                    print(f"  vec_{kind}: {time.perf_counter() - t:.2f}s")
            dst.commit()

    return 0


if __name__ == "__main__":
    sys.exit(main())
