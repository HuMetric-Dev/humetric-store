from __future__ import annotations

from humetric_core import Edge, Person, Skill

from humetric_store import (
    NotFound,
    bulk_upsert_persons,
    count_persons,
    get_person,
    list_edges_from,
    list_persons,
    open_db,
    upsert_edge,
    upsert_person,
)


def _conn():
    r = open_db(":memory:")
    assert r.is_ok()
    return r.unwrap()


def _make_person(pid: str = "gh:torvalds") -> Person:
    return Person(
        id=pid,
        source="github",
        name="Linus Torvalds",
        headline="kernel maintainer",
        about="creator of linux and git",
        location="OR, USA",
        skills=(Skill.of("C"), Skill.of("Git")),
        follower_count=200_000,
    )


def test_open_db_runs_migrations_idempotently() -> None:
    conn = _conn()
    # Reopening over the same conn shouldn't break anything; tables already exist.
    r = open_db(":memory:")
    assert r.is_ok()
    assert count_persons(conn).unwrap() == 0


def test_upsert_and_get_person_roundtrip() -> None:
    conn = _conn()
    p = _make_person()
    assert upsert_person(conn, p).is_ok()

    got = get_person(conn, p.id).unwrap()
    assert got.name == p.name
    assert got.follower_count == p.follower_count
    assert {s.normalized for s in got.skills} == {"c", "git"}


def test_get_person_returns_not_found() -> None:
    conn = _conn()
    r = get_person(conn, "missing")
    assert r.is_err()
    err = r.err()
    assert isinstance(err, NotFound)
    assert err.kind == "person"


def test_upsert_is_idempotent_and_updates_fields() -> None:
    conn = _conn()
    p = _make_person()
    upsert_person(conn, p).unwrap()
    p2 = p.model_copy(update={"headline": "linux foundation fellow"})
    upsert_person(conn, p2).unwrap()
    assert get_person(conn, p.id).unwrap().headline == "linux foundation fellow"
    assert count_persons(conn).unwrap() == 1


def test_bulk_upsert_persons_counts() -> None:
    conn = _conn()
    people = [_make_person(f"gh:user{i}") for i in range(5)]
    n = bulk_upsert_persons(conn, people).unwrap()
    assert n == 5
    assert count_persons(conn).unwrap() == 5


def test_list_persons_filters_by_source_and_paginates() -> None:
    conn = _conn()
    for i in range(3):
        upsert_person(conn, Person(id=f"gh:{i}", source="github", name=f"g{i}")).unwrap()
    upsert_person(conn, Person(id="li:1", source="linkedin", name="L1")).unwrap()

    gh = list_persons(conn, source="github").unwrap()
    assert {p.id for p in gh} == {"gh:0", "gh:1", "gh:2"}

    page = list_persons(conn, limit=2, offset=1).unwrap()
    assert len(page) == 2


def test_edges_upsert_and_list() -> None:
    conn = _conn()
    upsert_edge(conn, Edge(src="a", dst="b", kind="follow")).unwrap()
    upsert_edge(conn, Edge(src="a", dst="c", kind="follow", weight=0.5)).unwrap()
    upsert_edge(conn, Edge(src="a", dst="b", kind="follow", weight=0.7)).unwrap()  # update

    edges = list_edges_from(conn, "a").unwrap()
    assert len(edges) == 2
    weights = {(e.dst, e.weight) for e in edges}
    assert (("b", 0.7) in weights) and (("c", 0.5) in weights)
