from __future__ import annotations

from surrealdb import Surreal


def connect(
    url: str = "ws://127.0.0.1:8765/rpc",
    user: str = "root",
    password: str = "root",
    namespace: str = "humetric",
    database: str = "spike",
) -> Surreal:
    db = Surreal(url)
    db.signin({"username": user, "password": password})
    db.use(namespace, database)
    return db
