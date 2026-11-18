"""Bind proposed preference claims to host-owned source evidence.

The host supplies source authority and the catalog. A separate semantic verifier
must establish affirmative support from the full source; substring matching alone
does not distinguish a preference from a negation, quotation, or hypothetical.
This module checks bindings and renders accepted catalog choices, not raw claims.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class EvidenceSource:
    source_id: str
    text: str
    authority_scope: str
    persona_id: str
    category_id: str


def _catalog_product(product_id: str, catalog: list[dict]) -> dict | None:
    return next(
        (
            product
            for product in catalog
            if isinstance(product, dict)
            and product.get("id") == product_id
            and isinstance(product.get("name"), str)
            and product["name"].strip()
        ),
        None,
    )


def evaluate_claim(
    claim: dict,
    sources: dict[str, EvidenceSource],
    catalog: list[dict],
    persona_id: str,
    category_id: str,
    affirmative_support: bool = False,
) -> dict[str, Any]:
    """Validate one claim; extra claim fields never grant authority.

    ``affirmative_support`` must be an actual ``True`` from the separate
    full-source verifier. It defaults to rejection. Returned fields are limited
    to the decision, a reason code, and the canonical evidence identifiers.
    """
    result: dict[str, Any] = {
        "accepted": False,
        "reason": "malformed_claim",
        "product_id": "",
        "source_id": "",
        "quote": "",
    }
    if not isinstance(claim, dict):
        return result
    product_id = claim.get("product_id")
    source_id = claim.get("source_id")
    quote = claim.get("quote")
    if not isinstance(product_id, str) or not product_id:
        result["reason"] = "invalid_product_id"
        return result
    product = _catalog_product(product_id, catalog)
    if product is None:
        result["reason"] = "unknown_product"
        return result
    result["product_id"] = product["id"]
    if not isinstance(source_id, str) or source_id not in sources:
        result["reason"] = "unknown_source"
        return result
    source = sources[source_id]
    result["source_id"] = source.source_id
    if source.authority_scope != "preference":
        result["reason"] = "unauthorized_source"
        return result
    if source.persona_id != persona_id:
        result["reason"] = "persona_mismatch"
        return result
    if source.category_id != category_id:
        result["reason"] = "category_mismatch"
        return result
    if not isinstance(quote, str) or not quote.strip():
        result["reason"] = "empty_quote"
        return result
    if quote not in source.text:
        result["reason"] = "quote_not_in_source"
        return result
    result["quote"] = quote
    if product["name"].casefold() not in quote.casefold():
        result["reason"] = "product_not_in_quote"
        return result
    if affirmative_support is not True:
        result["reason"] = "missing_affirmative_support"
        return result
    result.update(accepted=True, reason="accepted")
    return result


def canonical_memory(accepted_claims: Iterable[dict], catalog: list[dict]) -> str:
    """Render evaluated claims using only names and IDs from the host catalog.

    Raw quotes, model-authored prose, and arbitrary metadata are audit evidence,
    not recommendation instructions. Callers must pass ``evaluate_claim``
    results, not an unvalidated writer response.
    """
    seen: set[tuple[str, str]] = set()
    lines: list[str] = []
    for claim in accepted_claims:
        if not isinstance(claim, dict) or claim.get("accepted") is not True:
            continue
        product_id, source_id = claim.get("product_id"), claim.get("source_id")
        if not isinstance(product_id, str) or not isinstance(source_id, str):
            continue
        if not source_id:
            continue
        product = _catalog_product(product_id, catalog)
        if product is None or (source_id, product_id) in seen:
            continue
        seen.add((source_id, product_id))
        lines.append(
            f"- The user prefers {product['name']} (product_id: {product['id']})."
        )
    return "\n".join(lines)


def scoped_canonical_memory(
    accepted_claims: Iterable[dict],
    catalog: list[dict],
    *,
    persona_id: str,
    persona_name: str,
    category_id: str,
) -> str:
    """Render the same accepted preferences with explicit host-owned scope.

    This representation adds identity and category context, not authentication
    claims or evidence of experience. It preserves per-source/product
    deduplication and never exposes source quotations or model-authored prose.
    """
    seen: set[tuple[str, str]] = set()
    preferences: list[dict[str, str]] = []
    for claim in accepted_claims:
        if not isinstance(claim, dict) or claim.get("accepted") is not True:
            continue
        product_id, source_id = claim.get("product_id"), claim.get("source_id")
        if not isinstance(product_id, str) or not isinstance(source_id, str):
            continue
        if not source_id:
            continue
        product = _catalog_product(product_id, catalog)
        if product is None or (source_id, product_id) in seen:
            continue
        seen.add((source_id, product_id))
        preferences.append({
            "product_id": product["id"],
            "product_name": product["name"],
            "relation": "prefers",
        })
    if not preferences:
        return ""
    return json.dumps(
        {
            "record_type": "stated_preference",
            "subject": {"persona_id": persona_id, "name": persona_name},
            "category_id": category_id,
            "preferences": preferences,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
