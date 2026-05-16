from __future__ import annotations

from surrealdb import Surreal

VECTOR_DIMS: dict[str, int] = {
    "text": 1024,
    "graph": 128,
    "tower": 256,
}


def _person_schema() -> str:
    fields = """
        DEFINE TABLE person SCHEMAFULL;
        DEFINE FIELD source ON person TYPE string;
        DEFINE FIELD name ON person TYPE string;
        DEFINE FIELD headline ON person TYPE string DEFAULT '';
        DEFINE FIELD about ON person TYPE string DEFAULT '';
        DEFINE FIELD location ON person TYPE string DEFAULT '';
        DEFINE FIELD follower_count ON person TYPE int DEFAULT 0;
        DEFINE FIELD last_active_days_ago ON person TYPE option<int>;
        DEFINE FIELD raw_url ON person TYPE string DEFAULT '';
    """
    vec_fields = "\n".join(
        f"DEFINE FIELD vec_{kind} ON person TYPE option<array<float, {dim}>>;"
        for kind, dim in VECTOR_DIMS.items()
    )
    return fields + "\n" + vec_fields


def _vector_indexes() -> str:
    return "\n".join(
        f"DEFINE INDEX vec_{kind}_hnsw ON person FIELDS vec_{kind} "
        f"HNSW DIMENSION {dim} DIST COSINE TYPE F32;"
        for kind, dim in VECTOR_DIMS.items()
    )


def _edge_schema() -> str:
    return """
        DEFINE TABLE skill SCHEMAFULL;
        DEFINE FIELD normalized ON skill TYPE string;
        DEFINE INDEX skill_normalized_idx ON skill FIELDS normalized;

        DEFINE TABLE follow TYPE RELATION FROM person TO person SCHEMAFULL;
        DEFINE FIELD weight ON follow TYPE float DEFAULT 1.0;

        DEFINE TABLE co_contributor TYPE RELATION FROM person TO person SCHEMAFULL;
        DEFINE FIELD weight ON co_contributor TYPE float DEFAULT 1.0;

        DEFINE TABLE has_skill TYPE RELATION FROM person TO skill SCHEMAFULL;
    """


def apply_schema(db: Surreal, with_vector_indexes: bool = True) -> None:
    db.query(_person_schema())
    db.query(_edge_schema())
    if with_vector_indexes:
        db.query(_vector_indexes())


EDGE_KINDS: tuple[str, ...] = ("follow", "co_contributor")
