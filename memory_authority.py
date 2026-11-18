"""Trusted-metadata authority routing for long-term agent memory.

The trust boundary is explicit: :class:`TrustedMetadata` must be built from
connector-controlled sidecar data, never from email or document text.  The
untrusted text in a :class:`MemoryItem` is deliberately absent from every
routing decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final, Iterable, Mapping


class AuthorityScope(str, Enum):
    """Authority granted by an immutable, connector-supplied sidecar."""

    PREFERENCE = "preference"
    OPERATIONAL = "operational"
    EXTERNAL = "external"
    QUARANTINE = "quarantine"


class MemoryCompartment(str, Enum):
    """Isolation compartments used by the memory store."""

    PREFERENCE = "preference"
    OPERATIONAL = "operational"
    EXTERNAL = "external"
    QUARANTINE = "quarantine"


class MemoryCapability(str, Enum):
    """Read capabilities issued to downstream tasks."""

    SHOPPING = "shopping"
    OPERATIONS = "operations"
    SOURCE_LOOKUP = "source_lookup"


_SCOPE_TO_COMPARTMENT: Final[Mapping[AuthorityScope, MemoryCompartment]] = (
    MappingProxyType(
        {
            AuthorityScope.PREFERENCE: MemoryCompartment.PREFERENCE,
            AuthorityScope.OPERATIONAL: MemoryCompartment.OPERATIONAL,
            AuthorityScope.EXTERNAL: MemoryCompartment.EXTERNAL,
            AuthorityScope.QUARANTINE: MemoryCompartment.QUARANTINE,
        }
    )
)

_CAPABILITY_GRANTS: Final[
    Mapping[MemoryCapability, frozenset[MemoryCompartment]]
] = MappingProxyType(
    {
        MemoryCapability.SHOPPING: frozenset({MemoryCompartment.PREFERENCE}),
        MemoryCapability.OPERATIONS: frozenset({MemoryCompartment.OPERATIONAL}),
        MemoryCapability.SOURCE_LOOKUP: frozenset(MemoryCompartment),
    }
)


def _sidecar_text(sidecar: Mapping[str, object], key: str, default: str) -> str:
    value = sidecar.get(key, default)
    return value if isinstance(value, str) else default


@dataclass(frozen=True, slots=True)
class TrustedMetadata:
    """Immutable snapshot of metadata supplied by a trusted integration.

    ``from_sidecar`` understands the repository's
    ``memory_authority_scope`` field.  A missing, malformed, or unknown scope
    is converted to ``QUARANTINE`` rather than guessed from the item text.
    """

    source_id: str
    source_type: str
    authority_scope: AuthorityScope
    source_label: str = ""

    @classmethod
    def from_sidecar(cls, sidecar: Mapping[str, object]) -> "TrustedMetadata":
        raw_scope = sidecar.get("memory_authority_scope")
        if isinstance(raw_scope, AuthorityScope):
            scope = raw_scope
        elif isinstance(raw_scope, str):
            try:
                scope = AuthorityScope(raw_scope.strip().lower())
            except ValueError:
                scope = AuthorityScope.QUARANTINE
        else:
            scope = AuthorityScope.QUARANTINE

        source_id = _sidecar_text(sidecar, "source_email_id", "")
        if not source_id:
            source_id = _sidecar_text(sidecar, "id", "unknown")

        return cls(
            source_id=source_id,
            source_type=_sidecar_text(sidecar, "source_type", "unknown"),
            authority_scope=scope,
            source_label=_sidecar_text(sidecar, "source_label", ""),
        )


@dataclass(frozen=True, slots=True)
class MemoryItem:
    """An untrusted memory payload bound to trusted metadata."""

    memory_id: str
    text: str
    trusted_metadata: TrustedMetadata


@dataclass(frozen=True, slots=True)
class RoutedMemory:
    """A memory item after deterministic authority routing."""

    item: MemoryItem
    compartment: MemoryCompartment

    def __post_init__(self) -> None:
        expected = _SCOPE_TO_COMPARTMENT.get(
            self.item.trusted_metadata.authority_scope,
            MemoryCompartment.QUARANTINE,
        )
        if self.compartment is not expected:
            raise ValueError(
                "compartment must match the item's trusted authority scope"
            )


def route_memory(item: MemoryItem) -> RoutedMemory:
    """Route one item using trusted metadata only."""

    compartment = _SCOPE_TO_COMPARTMENT.get(
        item.trusted_metadata.authority_scope,
        MemoryCompartment.QUARANTINE,
    )
    return RoutedMemory(item=item, compartment=compartment)


def allowed_compartments(
    capability: MemoryCapability,
) -> frozenset[MemoryCompartment]:
    """Return the immutable compartment grant for a read capability."""

    return _CAPABILITY_GRANTS[capability]


def filter_for_capability(
    records: Iterable[RoutedMemory],
    capability: MemoryCapability,
) -> tuple[RoutedMemory, ...]:
    """Apply the hard capability filter while preserving record order."""

    allowed = allowed_compartments(capability)
    return tuple(record for record in records if record.compartment in allowed)


@dataclass(frozen=True, slots=True)
class CompartmentedMemory:
    """Immutable routed-memory snapshot with capability-gated reads."""

    records: tuple[RoutedMemory, ...]

    @classmethod
    def from_items(cls, items: Iterable[MemoryItem]) -> "CompartmentedMemory":
        return cls(records=tuple(route_memory(item) for item in items))

    def read(self, capability: MemoryCapability) -> tuple[RoutedMemory, ...]:
        return filter_for_capability(self.records, capability)

    def in_compartment(
        self, compartment: MemoryCompartment
    ) -> tuple[RoutedMemory, ...]:
        return tuple(
            record for record in self.records
            if record.compartment is compartment
        )


__all__ = [
    "AuthorityScope",
    "CompartmentedMemory",
    "MemoryCapability",
    "MemoryCompartment",
    "MemoryItem",
    "RoutedMemory",
    "TrustedMetadata",
    "allowed_compartments",
    "filter_for_capability",
    "route_memory",
]
