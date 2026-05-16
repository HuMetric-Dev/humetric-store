# humetric-store

SQLite + FAISS persistence for Humetric. No business logic — just typed, `Result`-returning CRUD.

## Public API

```python
from humetric_store import open_db, upsert_person, get_person, VectorIndex
```

- `open_db(path) -> Result[Connection, StoreError]` opens (or creates) the SQLite database and runs idempotent migrations.
- `upsert_person`, `get_person`, `list_persons`, `upsert_edge`, `bulk_upsert_persons` — all return `Result[..., StoreError]`.
- `VectorIndex` wraps a `faiss.IndexFlatIP` keyed by stable int64 ids (mapped to person ids in SQLite).

## Schema

`persons(id PK, source, name, headline, about, location, follower_count, last_active_days_ago, raw_url)`
`edges(src, dst, kind, weight, PK(src, dst, kind))`
`skills(name PK, normalized)`
`person_skills(person_id, skill_name, PK(person_id, skill_name))`
`vectors(person_id PK, kind, vec BLOB)` — used as a slow-path mirror of FAISS for joins; FAISS is the runtime read path.
