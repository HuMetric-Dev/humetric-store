from __future__ import annotations

import sqlite3
from collections.abc import Iterable

from humetric_core import Edge, Err, Ok, Person, Result, Skill

from humetric_store.errors import (
    ConstraintViolated,
    DbReadFailed,
    DbWriteFailed,
    NotFound,
    StoreError,
)


def upsert_person(conn: sqlite3.Connection, p: Person) -> Result[None, StoreError]:
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO persons (
                    id, source, name, headline, about, location,
                    follower_count, last_active_days_ago, raw_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    source = excluded.source,
                    name = excluded.name,
                    headline = excluded.headline,
                    about = excluded.about,
                    location = excluded.location,
                    follower_count = excluded.follower_count,
                    last_active_days_ago = excluded.last_active_days_ago,
                    raw_url = excluded.raw_url
                """,
                (
                    p.id,
                    p.source,
                    p.name,
                    p.headline,
                    p.about,
                    p.location,
                    p.follower_count,
                    p.last_active_days_ago,
                    p.raw_url,
                ),
            )
            for s in p.skills:
                conn.execute(
                    "INSERT OR IGNORE INTO skills (name, normalized) VALUES (?, ?)",
                    (s.name, s.normalized),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO person_skills (person_id, skill_name) VALUES (?, ?)",
                    (p.id, s.name),
                )
    except sqlite3.IntegrityError as e:
        return Err(ConstraintViolated(table="persons", detail=str(e)))
    except sqlite3.Error as e:
        return Err(DbWriteFailed(table="persons", reason=str(e)))
    return Ok(None)


def bulk_upsert_persons(
    conn: sqlite3.Connection, persons: Iterable[Person]
) -> Result[int, StoreError]:
    count = 0
    for p in persons:
        r = upsert_person(conn, p)
        if isinstance(r, Err):
            return r
        count += 1
    return Ok(count)


def get_person(conn: sqlite3.Connection, person_id: str) -> Result[Person, StoreError]:
    try:
        row = conn.execute("SELECT * FROM persons WHERE id = ?", (person_id,)).fetchone()
        if row is None:
            return Err(NotFound(kind="person", key=person_id))
        skill_rows = conn.execute(
            """
            SELECT s.name, s.normalized FROM person_skills ps
            JOIN skills s ON s.name = ps.skill_name
            WHERE ps.person_id = ?
            ORDER BY s.name
            """,
            (person_id,),
        ).fetchall()
    except sqlite3.Error as e:
        return Err(DbReadFailed(table="persons", reason=str(e)))

    skills = tuple(Skill(name=r["name"], normalized=r["normalized"]) for r in skill_rows)
    return Ok(
        Person(
            id=row["id"],
            source=row["source"],
            name=row["name"],
            headline=row["headline"],
            about=row["about"],
            location=row["location"],
            follower_count=row["follower_count"],
            last_active_days_ago=row["last_active_days_ago"],
            raw_url=row["raw_url"],
            skills=skills,
        )
    )


def list_persons(
    conn: sqlite3.Connection,
    *,
    source: str | None = None,
    limit: int = 1000,
    offset: int = 0,
) -> Result[list[Person], StoreError]:
    try:
        if source is None:
            rows = conn.execute(
                "SELECT id FROM persons ORDER BY id LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id FROM persons WHERE source = ? ORDER BY id LIMIT ? OFFSET ?",
                (source, limit, offset),
            ).fetchall()
    except sqlite3.Error as e:
        return Err(DbReadFailed(table="persons", reason=str(e)))

    out: list[Person] = []
    for row in rows:
        r = get_person(conn, row["id"])
        if isinstance(r, Err):
            return r
        out.append(r.value)
    return Ok(out)


def count_persons(conn: sqlite3.Connection) -> Result[int, StoreError]:
    try:
        row = conn.execute("SELECT COUNT(*) AS n FROM persons").fetchone()
    except sqlite3.Error as e:
        return Err(DbReadFailed(table="persons", reason=str(e)))
    return Ok(int(row["n"]))


def upsert_edge(conn: sqlite3.Connection, e: Edge) -> Result[None, StoreError]:
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO edges (src, dst, kind, weight)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(src, dst, kind) DO UPDATE SET weight = excluded.weight
                """,
                (e.src, e.dst, e.kind, e.weight),
            )
    except sqlite3.Error as err:
        return Err(DbWriteFailed(table="edges", reason=str(err)))
    return Ok(None)


def list_edges_from(
    conn: sqlite3.Connection, src: str, kind: str | None = None
) -> Result[list[Edge], StoreError]:
    try:
        if kind is None:
            rows = conn.execute(
                "SELECT src, dst, kind, weight FROM edges WHERE src = ?", (src,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT src, dst, kind, weight FROM edges WHERE src = ? AND kind = ?",
                (src, kind),
            ).fetchall()
    except sqlite3.Error as e:
        return Err(DbReadFailed(table="edges", reason=str(e)))
    return Ok([Edge(src=r["src"], dst=r["dst"], kind=r["kind"], weight=r["weight"]) for r in rows])
