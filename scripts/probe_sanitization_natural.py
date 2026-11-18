#!/usr/bin/env python
"""Memory Sanitization decisions on the natural inbox, split by message kind (2026-11-17).

Replays the grid's classifier (same prompt, model, temperature 0) on the messages of the
natural inbox of each user: attack messages, genuine notes and the 300 everyday messages.
Writes logs/tables/sanitization_breakdown.json[<style>].

    source configs/vllm.env
    QWEN_CATALOG_V2=1 QWEN_HM_EMAILS_PER_CATEGORY=1 QWEN_SPOOF_COUNT=3 PYTHONNOUSERSITE=1 \
    QWEN_EMAIL_STYLE=spoofed_first_party python scripts/probe_sanitization_natural.py
"""
import asyncio
import collections
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
import exp_backends_qwen as X  # noqa: E402
import qwen_shopping_experiment as qse  # noqa: E402

OUT = REPO / "logs" / "tables" / "sanitization_breakdown.json"


def kind(email):
    if email.get("memory_authority_scope") == "preference":
        return "genuine"
    if (email.get("source_type") or "") == "verified_service_notification":
        return "everyday"
    return "attack"


async def main() -> None:
    sem = asyncio.Semaphore(12)
    total = collections.Counter()
    per = {}
    for pid in ("p01", "p02", "p03"):
        emails, *_ = X.build_neutral_daily_inbox(X.get_persona(pid), include_attack=True)
        flags = await asyncio.gather(*[qse.classify_email_for_block(e, sem) for e in emails])
        cnt = collections.Counter()
        for e, blocked in zip(emails, flags):
            cnt[f"{kind(e)}_{'blocked' if blocked else 'allowed'}"] += 1
        per[pid] = dict(sorted(cnt.items()))
        total.update(cnt)
        print(pid, per[pid], flush=True)
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data[X.EMAIL_STYLE] = {"total": dict(sorted(total.items())), "per_persona": per,
                           "model": X.QWEN_MODEL, "daily": X.DAILY_BENIGN_COUNT}
    OUT.write_text(json.dumps(data, indent=1))
    print("TOTAL", X.EMAIL_STYLE, dict(sorted(total.items())))


asyncio.run(main())
