"""Recompute skills.normalized using humetric_core.normalize_skill.

Needed because NPI / OpenAlex ingest rows were loaded before the
whitespace-hyphenation fix in humetric_core.types.normalize_skill, so their
stored `normalized` column still contains spaces (e.g.
'internal medicine, cardiovascular disease') even though the canonical form
is now 'internal-medicine,-cardiovascular-disease'. The function is
idempotent, so it's safe to run on every row.

Reads HUMETRIC_DB_URL. Prints a per-row diff for changed rows."""

from __future__ import annotations

import os
import sys

import psycopg
from humetric_core import normalize_skill


def main() -> int:
    dsn = os.environ.get("HUMETRIC_DB_URL")
    if not dsn:
        print("HUMETRIC_DB_URL is unset", file=sys.stderr)
        return 1

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute("SELECT name, normalized FROM skills ORDER BY name")
        rows = cur.fetchall()
        changed = 0
        for name, current in rows:
            new = normalize_skill(name)
            if new != current:
                cur.execute(
                    "UPDATE skills SET normalized = %s WHERE name = %s",
                    (new, name),
                )
                changed += 1
                print(f"  {name!r}: {current!r} -> {new!r}")
        conn.commit()
        print(f"\nrenormalized {changed} / {len(rows)} skill rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
