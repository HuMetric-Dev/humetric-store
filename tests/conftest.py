from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from urllib.parse import urlparse, urlunparse

import psycopg
import pytest

from humetric_store import open_db


def _admin_dsn() -> str:
    """DSN of the maintenance DB used to CREATE/DROP per-test databases.

    Override with HUMETRIC_TEST_DB_URL; defaults to the local spike container.
    """
    return os.environ.get(
        "HUMETRIC_TEST_DB_URL",
        "postgresql://postgres:postgres@127.0.0.1:5433/postgres",
    )


def _swap_dbname(dsn: str, dbname: str) -> str:
    p = urlparse(dsn)
    return urlunparse(p._replace(path=f"/{dbname}"))


@pytest.fixture
def pg_conn() -> Iterator[psycopg.Connection]:
    admin = _admin_dsn()
    test_db = f"humetric_test_{uuid.uuid4().hex[:12]}"

    admin_conn = psycopg.connect(admin, autocommit=True)
    try:
        with admin_conn.cursor() as cur:
            cur.execute(f'CREATE DATABASE "{test_db}"')
    finally:
        admin_conn.close()

    try:
        r = open_db(_swap_dbname(admin, test_db))
        assert r.is_ok(), r.err()
        conn = r.value
        yield conn
        conn.close()
    finally:
        admin_conn = psycopg.connect(admin, autocommit=True)
        try:
            with admin_conn.cursor() as cur:
                cur.execute(
                    f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    f"WHERE datname = '{test_db}' AND pid <> pg_backend_pid()"
                )
                cur.execute(f'DROP DATABASE IF EXISTS "{test_db}"')
        finally:
            admin_conn.close()
