from __future__ import annotations

from dataclasses import dataclass

import psycopg
from humetric_core import Err, Ok, ParsedQuery, Person, Result, Session, User

from humetric_store.errors import (
    ConstraintViolated,
    DbReadFailed,
    DbWriteFailed,
    NotFound,
    StoreError,
)


@dataclass(frozen=True, slots=True)
class AuthCredential:
    """Persistence-only record. Never crosses a route boundary; the raw
    password_hash string is opaque to everything but argon2 verify."""

    user_id: str
    password_hash: str


# --- users -----------------------------------------------------------------


def insert_user(conn: psycopg.Connection, user: User) -> Result[None, StoreError]:
    try:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (id, email, display_name, person_id, created_at, is_active) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (
                    user.id,
                    user.email,
                    user.display_name,
                    user.person_id,
                    user.created_at,
                    user.is_active,
                ),
            )
    except psycopg.errors.UniqueViolation as e:
        return Err(ConstraintViolated(table="users", detail=str(e)))
    except psycopg.errors.IntegrityError as e:
        return Err(ConstraintViolated(table="users", detail=str(e)))
    except psycopg.Error as e:
        return Err(DbWriteFailed(table="users", reason=str(e)))
    return Ok(None)


def _row_to_user(row: tuple) -> User:
    return User(
        id=row[0],
        email=row[1],
        display_name=row[2],
        person_id=row[3],
        created_at=float(row[4]),
        is_active=bool(row[5]),
    )


def get_user_by_id(conn: psycopg.Connection, user_id: str) -> Result[User, StoreError]:
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, email, display_name, person_id, created_at, is_active "
                "FROM users WHERE id = %s",
                (user_id,),
            )
            row = cur.fetchone()
    except psycopg.Error as e:
        return Err(DbReadFailed(table="users", reason=str(e)))
    if row is None:
        return Err(NotFound(kind="user", key=user_id))
    return Ok(_row_to_user(row))


def get_user_by_email(conn: psycopg.Connection, email: str) -> Result[User, StoreError]:
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, email, display_name, person_id, created_at, is_active "
                "FROM users WHERE email = %s",
                (email.lower(),),
            )
            row = cur.fetchone()
    except psycopg.Error as e:
        return Err(DbReadFailed(table="users", reason=str(e)))
    if row is None:
        return Err(NotFound(kind="user", key=email))
    return Ok(_row_to_user(row))


def set_user_person_id(
    conn: psycopg.Connection, user_id: str, person_id: str
) -> Result[None, StoreError]:
    """Claim a Person row as this user's 'self' persona. Idempotent in the
    sense that re-setting to the same value succeeds; setting a different
    value when already set is rejected at the call site (humetric-auth)."""
    try:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET person_id = %s WHERE id = %s",
                (person_id, user_id),
            )
            if cur.rowcount == 0:
                return Err(NotFound(kind="user", key=user_id))
    except psycopg.errors.UniqueViolation as e:
        # Hit by the partial unique index — another user already claims this person.
        return Err(ConstraintViolated(table="users", detail=str(e)))
    except psycopg.errors.IntegrityError as e:
        return Err(ConstraintViolated(table="users", detail=str(e)))
    except psycopg.Error as e:
        return Err(DbWriteFailed(table="users", reason=str(e)))
    return Ok(None)


def find_user_claiming_person(
    conn: psycopg.Connection, person_id: str
) -> Result[str | None, StoreError]:
    """Return user_id that currently claims this person, or None."""
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE person_id = %s", (person_id,))
            row = cur.fetchone()
    except psycopg.Error as e:
        return Err(DbReadFailed(table="users", reason=str(e)))
    return Ok(row[0] if row else None)


# --- credentials -----------------------------------------------------------


def insert_credential(conn: psycopg.Connection, cred: AuthCredential) -> Result[None, StoreError]:
    try:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                "INSERT INTO auth_credentials (user_id, password_hash) VALUES (%s, %s)",
                (cred.user_id, cred.password_hash),
            )
    except psycopg.errors.IntegrityError as e:
        return Err(ConstraintViolated(table="auth_credentials", detail=str(e)))
    except psycopg.Error as e:
        return Err(DbWriteFailed(table="auth_credentials", reason=str(e)))
    return Ok(None)


def get_credential(conn: psycopg.Connection, user_id: str) -> Result[AuthCredential, StoreError]:
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, password_hash FROM auth_credentials WHERE user_id = %s",
                (user_id,),
            )
            row = cur.fetchone()
    except psycopg.Error as e:
        return Err(DbReadFailed(table="auth_credentials", reason=str(e)))
    if row is None:
        return Err(NotFound(kind="credential", key=user_id))
    return Ok(AuthCredential(user_id=row[0], password_hash=row[1]))


# --- sessions --------------------------------------------------------------


def insert_session(conn: psycopg.Connection, session: Session) -> Result[None, StoreError]:
    try:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions ("
                "id, user_id, token_hash, created_at, expires_at, "
                "last_seen_at, user_agent, ip) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    session.id,
                    session.user_id,
                    session.token_hash,
                    session.created_at,
                    session.expires_at,
                    session.last_seen_at,
                    session.user_agent,
                    session.ip,
                ),
            )
    except psycopg.errors.IntegrityError as e:
        return Err(ConstraintViolated(table="sessions", detail=str(e)))
    except psycopg.Error as e:
        return Err(DbWriteFailed(table="sessions", reason=str(e)))
    return Ok(None)


def get_session_by_token_hash(
    conn: psycopg.Connection, token_hash: str
) -> Result[Session, StoreError]:
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, user_id, token_hash, created_at, expires_at, "
                "last_seen_at, user_agent, ip FROM sessions WHERE token_hash = %s",
                (token_hash,),
            )
            row = cur.fetchone()
    except psycopg.Error as e:
        return Err(DbReadFailed(table="sessions", reason=str(e)))
    if row is None:
        return Err(NotFound(kind="session", key=token_hash))
    return Ok(
        Session(
            id=row[0],
            user_id=row[1],
            token_hash=row[2],
            created_at=float(row[3]),
            expires_at=float(row[4]),
            last_seen_at=float(row[5]),
            user_agent=row[6],
            ip=row[7],
        )
    )


def touch_session(
    conn: psycopg.Connection, session_id: str, *, last_seen_at: float, expires_at: float
) -> Result[None, StoreError]:
    try:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                "UPDATE sessions SET last_seen_at = %s, expires_at = %s WHERE id = %s",
                (last_seen_at, expires_at, session_id),
            )
            if cur.rowcount == 0:
                return Err(NotFound(kind="session", key=session_id))
    except psycopg.Error as e:
        return Err(DbWriteFailed(table="sessions", reason=str(e)))
    return Ok(None)


def delete_session(conn: psycopg.Connection, session_id: str) -> Result[None, StoreError]:
    try:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
    except psycopg.Error as e:
        return Err(DbWriteFailed(table="sessions", reason=str(e)))
    return Ok(None)


def delete_sessions_for_user(conn: psycopg.Connection, user_id: str) -> Result[int, StoreError]:
    try:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute("DELETE FROM sessions WHERE user_id = %s", (user_id,))
            return Ok(cur.rowcount)
    except psycopg.Error as e:
        return Err(DbWriteFailed(table="sessions", reason=str(e)))


# --- person matching for the claim flow -------------------------------------


def _person_from_match_row(row: tuple) -> Person:
    """Match queries select the same 9-column projection as get_person, minus
    skills (which are joined per-row and cost more than they're worth at
    candidate-listing time). The candidate list is shown to the user; full
    skill detail is fetched only after they pick one."""
    return Person(
        id=row[0],
        source=row[1],
        name=row[2],
        headline=row[3],
        about=row[4],
        location=row[5],
        follower_count=row[6],
        last_active_days_ago=row[7],
        raw_url=row[8],
    )


def search_persons_by_github_username(
    conn: psycopg.Connection, github_username: str, *, limit: int = 5
) -> Result[tuple[Person, ...], StoreError]:
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, source, name, headline, about, location, "
                "follower_count, last_active_days_ago, raw_url "
                "FROM persons WHERE LOWER(github_username) = LOWER(%s) "
                "ORDER BY follower_count DESC LIMIT %s",
                (github_username, limit),
            )
            rows = cur.fetchall()
    except psycopg.Error as e:
        return Err(DbReadFailed(table="persons", reason=str(e)))
    return Ok(tuple(_person_from_match_row(r) for r in rows))


def search_persons_by_raw_url(
    conn: psycopg.Connection, raw_url: str, *, limit: int = 5
) -> Result[tuple[Person, ...], StoreError]:
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, source, name, headline, about, location, "
                "follower_count, last_active_days_ago, raw_url "
                "FROM persons WHERE LOWER(raw_url) = LOWER(%s) "
                "ORDER BY follower_count DESC LIMIT %s",
                (raw_url, limit),
            )
            rows = cur.fetchall()
    except psycopg.Error as e:
        return Err(DbReadFailed(table="persons", reason=str(e)))
    return Ok(tuple(_person_from_match_row(r) for r in rows))


def search_persons_by_name_lower(
    conn: psycopg.Connection, name: str, *, limit: int = 5
) -> Result[tuple[Person, ...], StoreError]:
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, source, name, headline, about, location, "
                "follower_count, last_active_days_ago, raw_url "
                "FROM persons WHERE LOWER(name) = LOWER(%s) "
                "ORDER BY follower_count DESC LIMIT %s",
                (name, limit),
            )
            rows = cur.fetchall()
    except psycopg.Error as e:
        return Err(DbReadFailed(table="persons", reason=str(e)))
    return Ok(tuple(_person_from_match_row(r) for r in rows))


# --- query history ---------------------------------------------------------


def append_query_history(
    conn: psycopg.Connection,
    user_id: str,
    ts: float,
    parsed: ParsedQuery,
    embedding: list[float],
) -> Result[None, StoreError]:
    """Insert one query into per-user history. `embedding` is the dense text
    vector (1024-d for bge-large; the column is fixed-width). Caller is
    responsible for size matching."""
    try:
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(
                "INSERT INTO query_history (user_id, ts, parsed, embedding) "
                "VALUES (%s, %s, %s::jsonb, %s)",
                (user_id, ts, parsed.model_dump_json(), embedding),
            )
    except psycopg.errors.IntegrityError as e:
        return Err(ConstraintViolated(table="query_history", detail=str(e)))
    except psycopg.Error as e:
        return Err(DbWriteFailed(table="query_history", reason=str(e)))
    return Ok(None)


def recent_query_embeddings(
    conn: psycopg.Connection, user_id: str, n: int
) -> Result[list[list[float]], StoreError]:
    """Return embeddings for the last `n` queries by this user, newest first.
    Used by the personalization centroid feature in humetric-orchestrator."""
    if n <= 0:
        return Ok([])
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT embedding FROM query_history WHERE user_id = %s ORDER BY ts DESC LIMIT %s",
                (user_id, n),
            )
            rows = cur.fetchall()
    except psycopg.Error as e:
        return Err(DbReadFailed(table="query_history", reason=str(e)))
    return Ok([list(r[0]) for r in rows])


def recent_query_history(
    conn: psycopg.Connection, user_id: str, n: int
) -> Result[list[tuple[float, dict, list[float]]], StoreError]:
    """Return (ts, parsed_dict, embedding) tuples for the last `n` queries by
    this user, newest first. The sidebar / history UI consumes this; for the
    centroid feature, `recent_query_embeddings` is cheaper."""
    if n <= 0:
        return Ok([])
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT ts, parsed, embedding FROM query_history "
                "WHERE user_id = %s ORDER BY ts DESC LIMIT %s",
                (user_id, n),
            )
            rows = cur.fetchall()
    except psycopg.Error as e:
        return Err(DbReadFailed(table="query_history", reason=str(e)))
    return Ok([(float(r[0]), dict(r[1]), list(r[2])) for r in rows])


__all__ = [
    "AuthCredential",
    "append_query_history",
    "delete_session",
    "delete_sessions_for_user",
    "find_user_claiming_person",
    "get_credential",
    "get_session_by_token_hash",
    "get_user_by_email",
    "get_user_by_id",
    "insert_credential",
    "insert_session",
    "insert_user",
    "recent_query_embeddings",
    "recent_query_history",
    "search_persons_by_github_username",
    "search_persons_by_name_lower",
    "search_persons_by_raw_url",
    "set_user_person_id",
    "touch_session",
]
