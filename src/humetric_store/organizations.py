from __future__ import annotations

from collections.abc import Iterable

import psycopg
from humetric_core import Err, Ok, Organization, Result
from psycopg import sql

from humetric_store.errors import (
    ConstraintViolated,
    DbReadFailed,
    DbWriteFailed,
    NotFound,
    StoreError,
)


def upsert_organization(conn: psycopg.Connection, o: Organization) -> Result[None, StoreError]:
    try:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO organizations (
                    id, source, name, org_kind, headline, about, location,
                    hq_location, founding_year, employee_count, raw_url
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(id) DO UPDATE SET
                    source = EXCLUDED.source,
                    name = EXCLUDED.name,
                    org_kind = EXCLUDED.org_kind,
                    headline = EXCLUDED.headline,
                    about = EXCLUDED.about,
                    location = EXCLUDED.location,
                    hq_location = EXCLUDED.hq_location,
                    founding_year = EXCLUDED.founding_year,
                    employee_count = EXCLUDED.employee_count,
                    raw_url = EXCLUDED.raw_url
                """,
                (
                    o.id,
                    o.source,
                    o.name,
                    o.org_kind,
                    o.headline,
                    o.about,
                    o.location,
                    o.hq_location,
                    o.founding_year,
                    o.employee_count,
                    o.raw_url,
                ),
            )
            # Reset industries on update so removed industries don't linger.
            cur.execute("DELETE FROM org_industries WHERE org_id = %s", (o.id,))
            for industry in o.industries:
                cur.execute(
                    "INSERT INTO org_industries (org_id, industry) VALUES (%s, %s)",
                    (o.id, industry),
                )
    except psycopg.errors.IntegrityError as e:
        return Err(ConstraintViolated(table="organizations", detail=str(e)))
    except psycopg.Error as e:
        return Err(DbWriteFailed(table="organizations", reason=str(e)))
    return Ok(None)


def bulk_upsert_organizations(
    conn: psycopg.Connection, orgs: Iterable[Organization]
) -> Result[int, StoreError]:
    count = 0
    for o in orgs:
        r = upsert_organization(conn, o)
        if isinstance(r, Err):
            return r
        count += 1
    return Ok(count)


def get_organization(conn: psycopg.Connection, org_id: str) -> Result[Organization, StoreError]:
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, source, name, org_kind, headline, about, location, "
                "hq_location, founding_year, employee_count, raw_url "
                "FROM organizations WHERE id = %s",
                (org_id,),
            )
            row = cur.fetchone()
            if row is None:
                return Err(NotFound(kind="organization", key=org_id))
            cur.execute(
                "SELECT industry FROM org_industries WHERE org_id = %s ORDER BY industry",
                (org_id,),
            )
            industries = tuple(r[0] for r in cur.fetchall())
    except psycopg.Error as e:
        return Err(DbReadFailed(table="organizations", reason=str(e)))

    return Ok(
        Organization(
            id=row[0],
            source=row[1],
            name=row[2],
            org_kind=row[3],
            headline=row[4],
            about=row[5],
            location=row[6],
            hq_location=row[7],
            founding_year=row[8],
            employee_count=row[9],
            raw_url=row[10],
            industries=industries,
        )
    )


def list_organizations(
    conn: psycopg.Connection,
    *,
    source: str | None = None,
    org_kind: str | None = None,
    limit: int = 1000,
    offset: int = 0,
) -> Result[list[Organization], StoreError]:
    try:
        with conn.cursor() as cur:
            clauses: list[sql.Composable] = []
            params: list[object] = []
            if source is not None:
                clauses.append(sql.SQL("source = %s"))
                params.append(source)
            if org_kind is not None:
                clauses.append(sql.SQL("org_kind = %s"))
                params.append(org_kind)
            where = sql.SQL("WHERE ") + sql.SQL(" AND ").join(clauses) if clauses else sql.SQL("")
            params.extend([limit, offset])
            stmt = sql.SQL(
                "SELECT id FROM organizations {where} ORDER BY id LIMIT %s OFFSET %s"
            ).format(where=where)
            cur.execute(stmt, params)
            rows = cur.fetchall()
    except psycopg.Error as e:
        return Err(DbReadFailed(table="organizations", reason=str(e)))

    out: list[Organization] = []
    for row in rows:
        r = get_organization(conn, row[0])
        if isinstance(r, Err):
            return r
        out.append(r.value)
    return Ok(out)


def count_organizations(conn: psycopg.Connection) -> Result[int, StoreError]:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM organizations")
            row = cur.fetchone()
    except psycopg.Error as e:
        return Err(DbReadFailed(table="organizations", reason=str(e)))
    return Ok(int(row[0]) if row else 0)
