from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from surrealdb import RecordID, Surreal

from humetric_store.surreal import apply_schema, connect
from humetric_store.surreal.schema import EDGE_KINDS, VECTOR_DIMS


def _chunks[T](items: list[T], n: int) -> Iterator[list[T]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def _open_sqlite(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _wipe(db: Surreal) -> None:
    for t in ("person", "skill", "has_skill", *EDGE_KINDS):
        db.query(f"REMOVE TABLE IF EXISTS {t};")


def _migrate_persons(src: sqlite3.Connection, db: Surreal, batch: int) -> int:
    rows = src.execute("SELECT * FROM persons").fetchall()
    docs = [
        {
            "id": RecordID("person", r["id"]),
            "source": r["source"],
            "name": r["name"],
            "headline": r["headline"] or "",
            "about": r["about"] or "",
            "location": r["location"] or "",
            "follower_count": int(r["follower_count"]),
            "last_active_days_ago": r["last_active_days_ago"],
            "raw_url": r["raw_url"] or "",
        }
        for r in rows
    ]
    n = 0
    for chunk in _chunks(docs, batch):
        db.insert("person", chunk)
        n += len(chunk)
    return n


def _migrate_skills(src: sqlite3.Connection, db: Surreal, batch: int) -> tuple[int, int]:
    skills = src.execute("SELECT name, normalized FROM skills").fetchall()
    skill_docs = [
        {"id": RecordID("skill", r["normalized"]), "normalized": r["normalized"]}
        for r in skills
    ]
    # Skills can collide on normalized form; dedupe
    seen: set[str] = set()
    deduped = []
    for d in skill_docs:
        key = d["normalized"]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(d)
    n_skill = 0
    for chunk in _chunks(deduped, batch):
        db.insert("skill", chunk)
        n_skill += len(chunk)

    ps_rows = src.execute(
        """
        SELECT ps.person_id, s.normalized
        FROM person_skills ps JOIN skills s ON s.name = ps.skill_name
        """
    ).fetchall()
    n_rel = 0
    for chunk in _chunks(ps_rows, batch):
        relations = [
            {
                "in": RecordID("person", r["person_id"]),
                "out": RecordID("skill", r["normalized"]),
            }
            for r in chunk
        ]
        db.insert_relation("has_skill", relations)
        n_rel += len(chunk)
    return n_skill, n_rel


def _migrate_edges(src: sqlite3.Connection, db: Surreal, batch: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for kind in EDGE_KINDS:
        rows = src.execute(
            "SELECT src, dst, weight FROM edges WHERE kind = ?", (kind,)
        ).fetchall()
        n = 0
        for chunk in _chunks(rows, batch):
            relations = [
                {
                    "in": RecordID("person", r["src"]),
                    "out": RecordID("person", r["dst"]),
                    "weight": float(r["weight"]),
                }
                for r in chunk
            ]
            db.insert_relation(kind, relations)
            n += len(chunk)
        counts[kind] = n
    return counts


def _migrate_vectors(src: sqlite3.Connection, db: Surreal, batch: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for kind, dim in VECTOR_DIMS.items():
        rows = src.execute(
            "SELECT person_id, vec FROM vectors WHERE kind = ?", (kind,)
        ).fetchall()
        n = 0
        for chunk in _chunks(rows, batch):
            updates = []
            for r in chunk:
                arr = np.frombuffer(r["vec"], dtype=np.float32)
                if arr.shape[0] != dim:
                    raise RuntimeError(f"dim mismatch for {r['person_id']}/{kind}: {arr.shape}")
                updates.append((RecordID("person", r["person_id"]), arr.tolist()))
            for rid, vec in updates:
                db.query(
                    f"UPDATE $id SET vec_{kind} = $v",
                    {"id": rid, "v": vec},
                )
            n += len(chunk)
        counts[kind] = n
    return counts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sqlite", type=Path, default=Path("../humetric-cli/data/humetric.db"))
    p.add_argument("--batch", type=int, default=500)
    p.add_argument("--skip-vectors", action="store_true")
    p.add_argument("--skip-vector-index", action="store_true",
                   help="Build vectors but skip HNSW index (huge time saver for iteration)")
    args = p.parse_args()

    src = _open_sqlite(args.sqlite)
    db = connect()
    print(f"connected; wiping spike DB...")
    _wipe(db)
    print(f"applying schema (vector indexes: {not args.skip_vector_index})...")
    apply_schema(db, with_vector_indexes=not args.skip_vector_index)

    t = time.perf_counter()
    np_persons = _migrate_persons(src, db, args.batch)
    t_persons = time.perf_counter() - t
    print(f"persons: {np_persons} in {t_persons:.2f}s ({np_persons / t_persons:.0f}/s)")

    t = time.perf_counter()
    n_skill, n_ps = _migrate_skills(src, db, args.batch)
    t_skills = time.perf_counter() - t
    print(f"skills: {n_skill} + has_skill: {n_ps} in {t_skills:.2f}s")

    t = time.perf_counter()
    edge_counts = _migrate_edges(src, db, args.batch)
    t_edges = time.perf_counter() - t
    print(f"edges: {edge_counts} in {t_edges:.2f}s ({sum(edge_counts.values()) / t_edges:.0f}/s)")

    if not args.skip_vectors:
        t = time.perf_counter()
        vc = _migrate_vectors(src, db, args.batch)
        t_vec = time.perf_counter() - t
        print(f"vectors: {vc} in {t_vec:.2f}s ({sum(vc.values()) / t_vec:.0f}/s)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
