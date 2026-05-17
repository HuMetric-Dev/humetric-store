from __future__ import annotations

import psycopg
from humetric_core import Organization

from humetric_store import (
    NotFound,
    bulk_upsert_organizations,
    count_organizations,
    get_organization,
    list_organizations,
    upsert_organization,
)


def _make_company(oid: str = "o:gh:anthropic") -> Organization:
    return Organization(
        id=oid,
        source="github",
        name="Anthropic",
        org_kind="company",
        headline="AI safety lab",
        about="research and product",
        location="San Francisco, CA",
        hq_location="San Francisco, CA",
        founding_year=2021,
        employee_count=500,
        industries=("artificial-intelligence", "research"),
    )


def _make_institution(oid: str = "o:oa:I27837315") -> Organization:
    return Organization(
        id=oid,
        source="openalex",
        name="MIT",
        org_kind="institution",
        location="Cambridge, MA",
    )


def test_upsert_and_get_organization_roundtrip(pg_conn: psycopg.Connection) -> None:
    o = _make_company()
    assert upsert_organization(pg_conn, o).is_ok()

    got = get_organization(pg_conn, o.id).unwrap()
    assert got.name == "Anthropic"
    assert got.org_kind == "company"
    assert got.founding_year == 2021
    assert got.employee_count == 500
    assert set(got.industries) == {"artificial-intelligence", "research"}


def test_get_organization_returns_not_found(pg_conn: psycopg.Connection) -> None:
    r = get_organization(pg_conn, "o:gh:nope")
    assert r.is_err()
    err = r.err()
    assert isinstance(err, NotFound)
    assert err.kind == "organization"


def test_upsert_org_idempotent_and_industries_replaced(pg_conn: psycopg.Connection) -> None:
    o = _make_company()
    upsert_organization(pg_conn, o).unwrap()
    o2 = o.model_copy(update={"industries": ("artificial-intelligence",)})
    upsert_organization(pg_conn, o2).unwrap()
    got = get_organization(pg_conn, o.id).unwrap()
    # Old "research" industry should be gone after the second upsert.
    assert got.industries == ("artificial-intelligence",)
    assert count_organizations(pg_conn).unwrap() == 1


def test_bulk_upsert_organizations_counts(pg_conn: psycopg.Connection) -> None:
    orgs = [
        Organization(id=f"o:gh:org{i}", source="github", name=f"Org{i}", org_kind="company")
        for i in range(5)
    ]
    n = bulk_upsert_organizations(pg_conn, orgs).unwrap()
    assert n == 5
    assert count_organizations(pg_conn).unwrap() == 5


def test_list_organizations_filters_by_source_and_kind(pg_conn: psycopg.Connection) -> None:
    upsert_organization(pg_conn, _make_company()).unwrap()
    upsert_organization(pg_conn, _make_institution()).unwrap()
    upsert_organization(
        pg_conn,
        Organization(
            id="o:gh:other",
            source="github",
            name="Other",
            org_kind="company",
        ),
    ).unwrap()

    gh = list_organizations(pg_conn, source="github").unwrap()
    assert {o.id for o in gh} == {"o:gh:anthropic", "o:gh:other"}

    institutions = list_organizations(pg_conn, org_kind="institution").unwrap()
    assert {o.id for o in institutions} == {"o:oa:I27837315"}
