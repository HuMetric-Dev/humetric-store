"""One-shot migration of the legacy SQLite + FAISS store into Postgres+pgvector.

Reads from a SQLite file (default: ../humetric-cli/data/humetric.db) and writes
to the Postgres instance at $HUMETRIC_DB_URL. Idempotent in the sense that
re-running over an already-populated DB skips rows on conflict.

    export HUMETRIC_DB_URL='postgresql://postgres:pw@host:5432/humetric'
    uv run python scripts/migrate_sqlite_to_pg.py --sqlite path/to/humetric.db
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import psycopg
from psycopg import sql

from humetric_store import open_db
from humetric_store.db import VECTOR_DIMS


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
                "INSERT INTO persons (id, source, name, headline, about, location, "
                "follower_count, last_active_days_ago, raw_url) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (id) DO NOTHING",
                [
                    (
                        r["id"],
                        r["source"],
                        r["name"],
                        r["headline"] or "",
                        r["about"] or "",
                        r["location"] or "",
                        int(r["follower_count"]),
                        r["last_active_days_ago"],
                        r["raw_url"] or "",
                    )
                    for r in chunk
                ],
            )
    dst.commit()
    return len(rows)


def _migrate_skills(
    src: sqlite3.Connection, dst: psycopg.Connection, batch: int
) -> tuple[int, int]:
    skills = src.execute("SELECT name, normalized FROM skills").fetchall()
    with dst.cursor() as cur:
        cur.executemany(
            "INSERT INTO skills (name, normalized) VALUES (%s, %s) "
            "ON CONFLICT (name) DO NOTHING",
            [(r["name"], r["normalized"]) for r in skills],
        )
    dst.commit()

    ps_rows = src.execute("SELECT person_id, skill_name FROM person_skills").fetchall()
    with dst.cursor() as cur:
        for chunk in _chunks(ps_rows, batch):
            cur.executemany(
                "INSERT INTO person_skills (person_id, skill_name) VALUES (%s, %s) "
                "ON CONFLICT (person_id, skill_name) DO NOTHING",
                [(r["person_id"], r["skill_name"]) for r in chunk],
            )
    dst.commit()
    return len(skills), len(ps_rows)


def _migrate_edges(
    src: sqlite3.Connection, dst: psycopg.Connection, batch: int
) -> dict[str, int]:
    edge_kinds = [
        r["kind"] for r in src.execute("SELECT DISTINCT kind FROM edges").fetchall()
    ]
    counts: dict[str, int] = {}
    with dst.cursor() as cur:
        for kind in edge_kinds:
            rows = src.execute(
                "SELECT src, dst, weight FROM edges WHERE kind = ?", (kind,)
            ).fetchall()
            for chunk in _chunks(rows, batch):
                cur.executemany(
                    "INSERT INTO edges (src, dst, kind, weight) VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (src, dst, kind) DO NOTHING",
                    [(r["src"], r["dst"], kind, float(r["weight"])) for r in chunk],
                )
            counts[kind] = len(rows)
    dst.commit()
    return counts


def _migrate_vectors(
    src: sqlite3.Connection, dst: psycopg.Connection, batch: int
) -> dict[str, int]:
    counts: dict[str, int] = {}
    with dst.cursor() as cur:
        for kind, dim in VECTOR_DIMS.items():
            rows = src.execute(
                "SELECT person_id, vec FROM vectors WHERE kind = ?", (kind,)
            ).fetchall()
            stmt = sql.SQL("UPDATE persons SET {col} = %s WHERE id = %s").format(
                col=sql.Identifier(f"vec_{kind}")
            )
            n = 0
            for chunk in _chunks(rows, batch):
                params = []
                for r in chunk:
                    arr = np.frombuffer(r["vec"], dtype=np.float32)
                    if arr.shape[0] != dim:
                        raise RuntimeError(
                            f"dim mismatch {r['person_id']}/{kind}: got {arr.shape}, expected ({dim},)"
                        )
                    params.append((arr, r["person_id"]))
                cur.executemany(stmt, params)
                n += len(chunk)
            counts[kind] = n
    dst.commit()
    return counts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sqlite",
        type=Path,
        default=Path("../humetric-cli/data/humetric.db"),
        help="Source SQLite database.",
    )
    p.add_argument("--batch", type=int, default=1000)
    p.add_argument(
        "--skip-vectors",
        action="store_true",
        help="Migrate only persons/edges/skills; leave vec_* columns NULL.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Connect and count source rows; do not write to Postgres.",
    )
    args = p.parse_args()

    if not args.sqlite.exists():
        print(f"source SQLite not found: {args.sqlite}", file=sys.stderr)
        return 2

    src = _open_sqlite(args.sqlite)
    persons_n = src.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    edges_n = src.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    vectors_n = src.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
    print(f"source: persons={persons_n} edges={edges_n} vectors={vectors_n}")

    if args.dry_run:
        return 0

    dsn = os.environ.get("HUMETRIC_DB_URL")
    if not dsn:
        print("HUMETRIC_DB_URL is unset; export a Postgres DSN.", file=sys.stderr)
        return 2

    dst_r = open_db(dsn)
    if dst_r.is_err():
        print(f"open_db failed: {dst_r.err()!r}", file=sys.stderr)
        return 1
    dst = dst_r.value
    print(f"connected to {dsn.split('@')[-1]}; schema ready.")

    t = time.perf_counter()
    n = _migrate_persons(src, dst, args.batch)
    dt = time.perf_counter() - t
    print(f"persons: {n} in {dt:.2f}s ({n / dt:.0f}/s)")

    t = time.perf_counter()
    n_s, n_ps = _migrate_skills(src, dst, args.batch)
    dt = time.perf_counter() - t
    print(f"skills: {n_s} + person_skills: {n_ps} in {dt:.2f}s")

    t = time.perf_counter()
    ec = _migrate_edges(src, dst, args.batch)
    dt = time.perf_counter() - t
    print(f"edges: {ec} in {dt:.2f}s ({sum(ec.values()) / dt:.0f}/s)")

    if not args.skip_vectors:
        t = time.perf_counter()
        vc = _migrate_vectors(src, dst, args.batch)
        dt = time.perf_counter() - t
        print(f"vectors: {vc} in {dt:.2f}s ({sum(vc.values()) / dt:.0f}/s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
