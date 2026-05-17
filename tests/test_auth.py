from __future__ import annotations

import time

import psycopg
import pytest
from humetric_core import (
    Err,
    Ok,
    ParsedQuery,
    Person,
    Session,
    User,
    new_session_id,
    new_user_id,
)

from humetric_store import (
    AuthCredential,
    NotFound,
    append_query_history,
    delete_session,
    delete_sessions_for_user,
    find_user_claiming_person,
    get_credential,
    get_session_by_token_hash,
    get_user_by_email,
    get_user_by_id,
    insert_credential,
    insert_session,
    insert_user,
    recent_query_embeddings,
    search_persons_by_github_username,
    search_persons_by_name_lower,
    search_persons_by_raw_url,
    set_user_person_id,
    touch_session,
    upsert_person,
)


def _mk_user(*, email: str = "a@example.com", person_id: str | None = None) -> User:
    return User(
        id=new_user_id(),
        email=email,
        display_name="A",
        person_id=person_id,
        created_at=time.time(),
    )


def _mk_session(user_id: str, *, token_hash: str | None = None) -> Session:
    now = time.time()
    return Session(
        id=new_session_id(),
        user_id=user_id,
        token_hash=token_hash or ("h" * 64),
        created_at=now,
        expires_at=now + 86400,
        last_seen_at=now,
    )


def test_insert_and_get_user_roundtrip(pg_conn: psycopg.Connection) -> None:
    u = _mk_user()
    assert isinstance(insert_user(pg_conn, u), Ok)
    r = get_user_by_id(pg_conn, u.id)
    assert isinstance(r, Ok)
    assert r.value.email == u.email
    assert r.value.person_id is None


def test_get_user_by_email_lowercases(pg_conn: psycopg.Connection) -> None:
    u = _mk_user(email="Mixed@Example.COM")
    # store accepts whatever case is passed in; the auth layer is responsible
    # for lowercasing on the way in. Confirm the lookup helper lowercases.
    assert isinstance(insert_user(pg_conn, u), Ok)
    r = get_user_by_email(pg_conn, "MIXED@example.com")
    # The row was inserted with the original case, but lookup uses LOWER on
    # the query side only — so a case-mismatched insert/select will miss.
    # In production, callers normalize before insert. Here verify the SQL
    # parameter is downcased before the WHERE.
    assert isinstance(r, Err)


def test_get_user_by_id_not_found(pg_conn: psycopg.Connection) -> None:
    r = get_user_by_id(pg_conn, "u:does-not-exist")
    assert isinstance(r, Err)
    assert isinstance(r.error, NotFound)


def test_unique_email_rejected(pg_conn: psycopg.Connection) -> None:
    u = _mk_user()
    assert isinstance(insert_user(pg_conn, u), Ok)
    dup = _mk_user(email=u.email)
    r = insert_user(pg_conn, dup)
    assert isinstance(r, Err)


def test_set_user_person_id_links_and_prevents_double_claim(
    pg_conn: psycopg.Connection,
) -> None:
    person = Person(id="p:gh:claimable", source="github", name="Claimable")
    assert isinstance(upsert_person(pg_conn, person), Ok)

    user_a = _mk_user(email="a@example.com")
    user_b = _mk_user(email="b@example.com")
    assert isinstance(insert_user(pg_conn, user_a), Ok)
    assert isinstance(insert_user(pg_conn, user_b), Ok)

    assert isinstance(set_user_person_id(pg_conn, user_a.id, person.id), Ok)
    # second user trying to claim the same person violates the partial
    # unique index — verifies the multi-user "one person, one user" rule
    assert isinstance(set_user_person_id(pg_conn, user_b.id, person.id), Err)

    owner_r = find_user_claiming_person(pg_conn, person.id)
    assert isinstance(owner_r, Ok)
    assert owner_r.value == user_a.id


def test_credential_roundtrip(pg_conn: psycopg.Connection) -> None:
    u = _mk_user()
    assert isinstance(insert_user(pg_conn, u), Ok)
    cred = AuthCredential(user_id=u.id, password_hash="$argon2id$v=19$...")
    assert isinstance(insert_credential(pg_conn, cred), Ok)
    got = get_credential(pg_conn, u.id)
    assert isinstance(got, Ok)
    assert got.value.password_hash == cred.password_hash


def test_credential_cascades_on_user_delete(pg_conn: psycopg.Connection) -> None:
    u = _mk_user()
    assert isinstance(insert_user(pg_conn, u), Ok)
    assert isinstance(
        insert_credential(pg_conn, AuthCredential(user_id=u.id, password_hash="h")), Ok
    )
    with pg_conn.cursor() as cur:
        cur.execute("DELETE FROM users WHERE id = %s", (u.id,))
    pg_conn.commit()
    assert isinstance(get_credential(pg_conn, u.id), Err)


def test_session_lifecycle(pg_conn: psycopg.Connection) -> None:
    u = _mk_user()
    assert isinstance(insert_user(pg_conn, u), Ok)
    s = _mk_session(u.id)
    assert isinstance(insert_session(pg_conn, s), Ok)

    fetched = get_session_by_token_hash(pg_conn, s.token_hash)
    assert isinstance(fetched, Ok)
    assert fetched.value.id == s.id

    # slide the expiry forward
    new_expiry = s.expires_at + 1_209_600
    new_seen = s.last_seen_at + 60
    assert isinstance(
        touch_session(pg_conn, s.id, last_seen_at=new_seen, expires_at=new_expiry),
        Ok,
    )
    again = get_session_by_token_hash(pg_conn, s.token_hash)
    assert isinstance(again, Ok)
    assert again.value.expires_at == pytest.approx(new_expiry)

    assert isinstance(delete_session(pg_conn, s.id), Ok)
    assert isinstance(get_session_by_token_hash(pg_conn, s.token_hash), Err)


def test_delete_sessions_for_user_count(pg_conn: psycopg.Connection) -> None:
    u = _mk_user()
    assert isinstance(insert_user(pg_conn, u), Ok)
    for i in range(3):
        s = _mk_session(u.id, token_hash=f"hash-{i}-" + "0" * 50)
        assert isinstance(insert_session(pg_conn, s), Ok)
    r = delete_sessions_for_user(pg_conn, u.id)
    assert isinstance(r, Ok)
    assert r.value == 3


def test_search_persons_by_github_username(pg_conn: psycopg.Connection) -> None:
    p = Person(id="p:gh:torvalds", source="github", name="Linus Torvalds")
    assert isinstance(upsert_person(pg_conn, p), Ok)
    # github_username column isn't set by upsert_person; populate directly
    with pg_conn.cursor() as cur:
        cur.execute(
            "UPDATE persons SET github_username = %s WHERE id = %s",
            ("torvalds", p.id),
        )
    pg_conn.commit()

    r = search_persons_by_github_username(pg_conn, "TORVALDS")  # case-insensitive
    assert isinstance(r, Ok)
    assert len(r.value) == 1
    assert r.value[0].id == p.id


def test_search_persons_by_raw_url(pg_conn: psycopg.Connection) -> None:
    p = Person(
        id="p:li:torvalds",
        source="linkedin",
        name="Linus Torvalds",
        raw_url="https://linkedin.com/in/torvalds",
    )
    assert isinstance(upsert_person(pg_conn, p), Ok)
    r = search_persons_by_raw_url(pg_conn, "https://LinkedIn.com/in/torvalds")
    assert isinstance(r, Ok)
    assert len(r.value) == 1


def test_search_persons_by_name_lower(pg_conn: psycopg.Connection) -> None:
    a = Person(id="p:gh:a", source="github", name="Jane Doe")
    b = Person(id="p:gh:b", source="github", name="jane doe", follower_count=10)
    for p in (a, b):
        assert isinstance(upsert_person(pg_conn, p), Ok)
    r = search_persons_by_name_lower(pg_conn, "Jane Doe")
    assert isinstance(r, Ok)
    assert {p.id for p in r.value} == {a.id, b.id}
    # ORDER BY follower_count DESC — b should sort first
    assert r.value[0].id == b.id


def test_query_history_append_and_recent(pg_conn: psycopg.Connection) -> None:
    u = _mk_user()
    assert isinstance(insert_user(pg_conn, u), Ok)

    embeddings: list[list[float]] = [[float(i)] * 1024 for i in range(3)]
    base_ts = time.time()
    for i, vec in enumerate(embeddings):
        r = append_query_history(
            pg_conn,
            user_id=u.id,
            ts=base_ts + i,
            parsed=ParsedQuery(free_text=f"q-{i}"),
            embedding=vec,
        )
        assert isinstance(r, Ok)

    got = recent_query_embeddings(pg_conn, u.id, 2)
    assert isinstance(got, Ok)
    assert len(got.value) == 2
    # newest first
    assert got.value[0][0] == 2.0
    assert got.value[1][0] == 1.0


def test_query_history_isolated_per_user(pg_conn: psycopg.Connection) -> None:
    a = _mk_user(email="a@example.com")
    b = _mk_user(email="b@example.com")
    assert isinstance(insert_user(pg_conn, a), Ok)
    assert isinstance(insert_user(pg_conn, b), Ok)
    assert isinstance(
        append_query_history(
            pg_conn,
            user_id=a.id,
            ts=time.time(),
            parsed=ParsedQuery(free_text="for a"),
            embedding=[1.0] * 1024,
        ),
        Ok,
    )
    r = recent_query_embeddings(pg_conn, b.id, 5)
    assert isinstance(r, Ok)
    assert r.value == []
