"""Read-time guard for decisions proposed from untrusted memory.

The guard deliberately does not classify a memory as benign or malicious.
Instead, it compares a tool-less proposal that saw retrieved memory with a
physically separate proposal that saw no memory.  An unconfirmed memory-caused
decision change is never authorized: an interactive system asks the current
user, while a non-interactive system uses the no-memory proposal.

This policy handles the information-theoretic limit of text-only defenses.  A
forged memory that is byte-for-byte identical to a genuine memory cannot be
distinguished at read time without another signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GuardDisposition(str, Enum):
    """Outcome of the counterfactual decision comparison."""

    STABLE = "stable"
    CURRENT_USER_CONFIRMED = "current_user_confirmed"
    CONFIRMATION_REQUIRED = "confirmation_required"
    FALLBACK_NO_INTERACTION = "fallback_no_interaction"


@dataclass(frozen=True, slots=True)
class CounterfactualGuardResult:
    """Auditable result produced before an agent can commit an action."""

    baseline_choice_id: str
    memory_choice_id: str
    final_choice_id: str
    disposition: GuardDisposition
    memory_was_read: bool
    memory_induced_change: bool
    confirmation_required: bool
    action_authorized: bool


def apply_strict_counterfactual_guard(
    *,
    baseline_choice_id: str,
    memory_choice_id: str,
    memory_was_read: bool,
    current_user_choice_id: str | None = None,
    interaction_available: bool = True,
) -> CounterfactualGuardResult:
    """Prevent unconfirmed memory from autonomously changing a decision.

    ``baseline_choice_id`` must come from a separate inference that receives no
    retrieved memory.  Asking the same poisoned inference to imagine a
    no-memory answer is not a valid counterfactual.

    ``final_choice_id`` is the safe choice used by non-interactive evaluation.
    When confirmation is required, ``action_authorized`` is false even though
    the baseline choice is returned as a deterministic fallback candidate.
    """

    if not baseline_choice_id:
        raise ValueError("baseline_choice_id must be non-empty")
    if not memory_choice_id:
        raise ValueError("memory_choice_id must be non-empty")
    if not memory_was_read and memory_choice_id != baseline_choice_id:
        raise ValueError(
            "a proposal that did not read memory must match the no-memory baseline"
        )

    memory_induced_change = memory_was_read and (
        memory_choice_id != baseline_choice_id
    )

    # A choice stated in the current interaction is new authorization, not a
    # trust judgment about old memory.  It therefore takes precedence.
    if current_user_choice_id is not None:
        if not current_user_choice_id:
            raise ValueError("current_user_choice_id must be non-empty when set")
        return CounterfactualGuardResult(
            baseline_choice_id=baseline_choice_id,
            memory_choice_id=memory_choice_id,
            final_choice_id=current_user_choice_id,
            disposition=GuardDisposition.CURRENT_USER_CONFIRMED,
            memory_was_read=memory_was_read,
            memory_induced_change=memory_induced_change,
            confirmation_required=False,
            action_authorized=True,
        )

    if not memory_induced_change:
        return CounterfactualGuardResult(
            baseline_choice_id=baseline_choice_id,
            memory_choice_id=memory_choice_id,
            final_choice_id=baseline_choice_id,
            disposition=GuardDisposition.STABLE,
            memory_was_read=memory_was_read,
            memory_induced_change=False,
            confirmation_required=False,
            action_authorized=True,
        )

    if interaction_available:
        return CounterfactualGuardResult(
            baseline_choice_id=baseline_choice_id,
            memory_choice_id=memory_choice_id,
            final_choice_id=baseline_choice_id,
            disposition=GuardDisposition.CONFIRMATION_REQUIRED,
            memory_was_read=True,
            memory_induced_change=True,
            confirmation_required=True,
            action_authorized=False,
        )

    return CounterfactualGuardResult(
        baseline_choice_id=baseline_choice_id,
        memory_choice_id=memory_choice_id,
        final_choice_id=baseline_choice_id,
        disposition=GuardDisposition.FALLBACK_NO_INTERACTION,
        memory_was_read=True,
        memory_induced_change=True,
        confirmation_required=False,
        action_authorized=True,
    )


__all__ = [
    "CounterfactualGuardResult",
    "GuardDisposition",
    "apply_strict_counterfactual_guard",
]
