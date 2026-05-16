from __future__ import annotations

from collections.abc import Iterable

import psycopg
from humetric_core import Edge, Err, Ok, Person, Result, Skill

from humetric_store.errors import (
    ConstraintViolated,
    DbReadFailed,
    DbWriteFailed,
    NotFound,
    StoreError,
)


def upsert_person(conn: psycopg.Connection, p: Person) -> Result[None, StoreError]:
    try:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO persons (
                    id, source, name, headline, about, location,
                    follower_count, last_active_days_ago, raw_url
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(id) DO UPDATE SET
                    source = EXCLUDED.source,
                    name = EXCLUDED.name,
                    headline = EXCLUDED.headline,
                    about = EXCLUDED.about,
                    location = EXCLUDED.location,
                    follower_count = EXCLUDED.follower_count,
                    last_active_days_ago = EXCLUDED.last_active_days_ago,
                    raw_url = EXCLUDED.raw_url
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
                cur.execute(
                    "INSERT INTO skills (name, normalized) VALUES (%s, %s) "
                    "ON CONFLICT (name) DO NOTHING",
                    (s.name, s.normalized),
                )
                cur.execute(
                    "INSERT INTO person_skills (person_id, skill_name) VALUES (%s, %s) "
                    "ON CONFLICT (person_id, skill_name) DO NOTHING",
                    (p.id, s.name),
                )
    except psycopg.errors.IntegrityError as e:
        return Err(ConstraintViolated(table="persons", detail=str(e)))
    except psycopg.Error as e:
        return Err(DbWriteFailed(table="persons", reason=str(e)))
    return Ok(None)


def bulk_upsert_persons(
    conn: psycopg.Connection, persons: Iterable[Person]
) -> Result[int, StoreError]:
    count = 0
    for p in persons:
        r = upsert_person(conn, p)
        if isinstance(r, Err):
            return r
        count += 1
    return Ok(count)


def get_person(conn: psycopg.Connection, person_id: str) -> Result[Person, StoreError]:
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, source, name, headline, about, location, "
                "follower_count, last_active_days_ago, raw_url "
                "FROM persons WHERE id = %s",
                (person_id,),
            )
            row = cur.fetchone()
            if row is None:
                return Err(NotFound(kind="person", key=person_id))
            cur.execute(
                """
                SELECT s.name, s.normalized FROM person_skills ps
                JOIN skills s ON s.name = ps.skill_name
                WHERE ps.person_id = %s
                ORDER BY s.name
                """,
                (person_id,),
            )
            skill_rows = cur.fetchall()
    except psycopg.Error as e:
        return Err(DbReadFailed(table="persons", reason=str(e)))

    skills = tuple(Skill(name=r[0], normalized=r[1]) for r in skill_rows)
    return Ok(
        Person(
            id=row[0],
            source=row[1],
            name=row[2],
            headline=row[3],
            about=row[4],
            location=row[5],
            follower_count=row[6],
            last_active_days_ago=row[7],
            raw_url=row[8],
            skills=skills,
        )
    )


def list_persons(
    conn: psycopg.Connection,
    *,
    source: str | None = None,
    limit: int = 1000,
    offset: int = 0,
) -> Result[list[Person], StoreError]:
    try:
        with conn.cursor() as cur:
            if source is None:
                cur.execute(
                    "SELECT id FROM persons ORDER BY id LIMIT %s OFFSET %s",
                    (limit, offset),
                )
            else:
                cur.execute(
                    "SELECT id FROM persons WHERE source = %s ORDER BY id LIMIT %s OFFSET %s",
                    (source, limit, offset),
                )
            rows = cur.fetchall()
    except psycopg.Error as e:
        return Err(DbReadFailed(table="persons", reason=str(e)))

    out: list[Person] = []
    for row in rows:
        r = get_person(conn, row[0])
        if isinstance(r, Err):
            return r
        out.append(r.value)
    return Ok(out)


def count_persons(conn: psycopg.Connection) -> Result[int, StoreError]:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM persons")
            row = cur.fetchone()
    except psycopg.Error as e:
        return Err(DbReadFailed(table="persons", reason=str(e)))
    return Ok(int(row[0]) if row else 0)


def upsert_edge(conn: psycopg.Connection, e: Edge) -> Result[None, StoreError]:
    try:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO edges (src, dst, kind, weight)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (src, dst, kind) DO UPDATE SET weight = EXCLUDED.weight
                """,
                (e.src, e.dst, e.kind, e.weight),
            )
    except psycopg.Error as err:
        return Err(DbWriteFailed(table="edges", reason=str(err)))
    return Ok(None)


def list_edges_from(
    conn: psycopg.Connection, src: str, kind: str | None = None
) -> Result[list[Edge], StoreError]:
    try:
        with conn.cursor() as cur:
            if kind is None:
                cur.execute("SELECT src, dst, kind, weight FROM edges WHERE src = %s", (src,))
            else:
                cur.execute(
                    "SELECT src, dst, kind, weight FROM edges WHERE src = %s AND kind = %s",
                    (src, kind),
                )
            rows = cur.fetchall()
    except psycopg.Error as e:
        return Err(DbReadFailed(table="edges", reason=str(e)))
    return Ok([Edge(src=r[0], dst=r[1], kind=r[2], weight=r[3]) for r in rows])
