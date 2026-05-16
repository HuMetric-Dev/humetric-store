from __future__ import annotations

from dataclasses import dataclass

from humetric_core import HumetricError


@dataclass(frozen=True, slots=True)
class DbOpenFailed(HumetricError):
    dsn: str
    reason: str


@dataclass(frozen=True, slots=True)
class DbWriteFailed(HumetricError):
    table: str
    reason: str


@dataclass(frozen=True, slots=True)
class DbReadFailed(HumetricError):
    table: str
    reason: str


@dataclass(frozen=True, slots=True)
class NotFound(HumetricError):
    kind: str
    key: str


@dataclass(frozen=True, slots=True)
class ConstraintViolated(HumetricError):
    table: str
    detail: str


@dataclass(frozen=True, slots=True)
class VectorShapeMismatch(HumetricError):
    expected_shape: tuple[int, ...]
    got_shape: tuple[int, ...]


type StoreError = (
    DbOpenFailed
    | DbWriteFailed
    | DbReadFailed
    | NotFound
    | ConstraintViolated
    | VectorShapeMismatch
)
