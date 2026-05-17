from __future__ import annotations

import psycopg
from humetric_core import Edge, Person, Skill

from humetric_store import (
    NotFound,
    bulk_upsert_persons,
    count_persons,
    get_person,
    list_edges_from,
    list_persons,
    upsert_edge,
    upsert_person,
)


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


def test_open_db_runs_migrations_idempotently(pg_conn: psycopg.Connection) -> None:
    assert count_persons(pg_conn).unwrap() == 0


def test_upsert_and_get_person_roundtrip(pg_conn: psycopg.Connection) -> None:
    p = _make_person()
    assert upsert_person(pg_conn, p).is_ok()

    got = get_person(pg_conn, p.id).unwrap()
    assert got.name == p.name
    assert got.follower_count == p.follower_count
    assert {s.normalized for s in got.skills} == {"c", "git"}


def test_get_person_returns_not_found(pg_conn: psycopg.Connection) -> None:
    r = get_person(pg_conn, "missing")
    assert r.is_err()
    err = r.err()
    assert isinstance(err, NotFound)
    assert err.kind == "person"


def test_upsert_is_idempotent_and_updates_fields(pg_conn: psycopg.Connection) -> None:
    p = _make_person()
    upsert_person(pg_conn, p).unwrap()
    p2 = p.model_copy(update={"headline": "linux foundation fellow"})
    upsert_person(pg_conn, p2).unwrap()
    assert get_person(pg_conn, p.id).unwrap().headline == "linux foundation fellow"
    assert count_persons(pg_conn).unwrap() == 1


def test_bulk_upsert_persons_counts(pg_conn: psycopg.Connection) -> None:
    people = [_make_person(f"gh:user{i}") for i in range(5)]
    n = bulk_upsert_persons(pg_conn, people).unwrap()
    assert n == 5
    assert count_persons(pg_conn).unwrap() == 5


def test_list_persons_filters_by_source_and_paginates(pg_conn: psycopg.Connection) -> None:
    for i in range(3):
        upsert_person(pg_conn, Person(id=f"gh:{i}", source="github", name=f"g{i}")).unwrap()
    upsert_person(pg_conn, Person(id="li:1", source="linkedin", name="L1")).unwrap()

    gh = list_persons(pg_conn, source="github").unwrap()
    assert {p.id for p in gh} == {"gh:0", "gh:1", "gh:2"}

    page = list_persons(pg_conn, limit=2, offset=1).unwrap()
    assert len(page) == 2


def test_edges_upsert_and_list(pg_conn: psycopg.Connection) -> None:
    upsert_edge(pg_conn, Edge(src="a", dst="b", kind="follow")).unwrap()
    upsert_edge(pg_conn, Edge(src="a", dst="c", kind="follow", weight=0.5)).unwrap()
    upsert_edge(pg_conn, Edge(src="a", dst="b", kind="follow", weight=0.7)).unwrap()

    edges = list_edges_from(pg_conn, "a").unwrap()
    assert len(edges) == 2
    weights = {(e.dst, e.weight) for e in edges}
    assert (("b", 0.7) in weights) and (("c", 0.5) in weights)


def test_edges_derive_src_dst_type_from_prefixed_ids(pg_conn: psycopg.Connection) -> None:
    upsert_edge(
        pg_conn,
        Edge(src="p:gh:octocat", dst="o:gh:anthropic", kind="works_at"),
    ).unwrap()
    upsert_edge(pg_conn, Edge(src="legacy_id", dst="other_legacy", kind="follow")).unwrap()

    with pg_conn.cursor() as cur:
        cur.execute(
            "SELECT src, dst, src_type, dst_type FROM edges ORDER BY src",
        )
        rows = dict((r[0], (r[2], r[3])) for r in cur.fetchall())

    assert rows["p:gh:octocat"] == ("person", "organization")
    # Legacy unprefixed IDs do not parse, so the denormalized type columns are NULL.
    assert rows["legacy_id"] == (None, None)


def test_entities_view_unions_persons_and_organizations(pg_conn: psycopg.Connection) -> None:
    from humetric_core import Organization

    from humetric_store import upsert_organization

    upsert_person(pg_conn, _make_person("p:gh:torvalds")).unwrap()
    upsert_organization(
        pg_conn,
        Organization(
            id="o:gh:anthropic",
            source="github",
            name="Anthropic",
            org_kind="company",
        ),
    ).unwrap()

    with pg_conn.cursor() as cur:
        cur.execute("SELECT id, entity_type FROM entities ORDER BY id")
        rows = cur.fetchall()

    assert rows == [
        ("o:gh:anthropic", "organization"),
        ("p:gh:torvalds", "person"),
    ]
