#!/usr/bin/env python
"""StateGuard decisions on the natural inbox, split by message kind (2026-11-17).

For one user (argv[1]) and both attacks, runs the grid's neutral writer on every message of
the natural inbox and audits every note it writes with the grid's StateGuard auditor (same
prompts, model, temperature 0). Counts candidate memory writes and rolled-back writes for
attack messages, genuine notes and everyday messages. This is the write path shared by
Explicit, Context and RAG. Output: logs/tables/stateguard_breakdown_<user>.json

    source configs/vllm.env
    QWEN_CATALOG_V2=1 QWEN_HM_EMAILS_PER_CATEGORY=1 QWEN_SPOOF_COUNT=3 QWEN_SHARE_WRITER_NOTES=1 \
    PYTHONNOUSERSITE=1 python scripts/probe_stateguard_natural.py p01
"""
import asyncio
import collections
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
PID = sys.argv[1]
OUT = REPO / "logs" / "tables" / f"stateguard_breakdown_{PID}.json"


def kind(email):
    if email.get("memory_authority_scope") == "preference":
        return "genuine"
    if (email.get("source_type") or "") == "verified_service_notification":
        return "everyday"
    return "attack"


async def run_style(style):
    os.environ["QWEN_EMAIL_STYLE"] = style
    import importlib
    import exp_backends_qwen as X
    import qwen_shopping_experiment as qse
    X.EMAIL_STYLE = style
    persona = X.get_persona(PID)
    emails, *_ = X.build_neutral_daily_inbox(persona, include_attack=True)
    sem = asyncio.Semaphore(int(os.environ.get("PROBE_CONCURRENCY", "12")))
    cnt = collections.Counter()

    async def one(email):
        try:
            notes, _raw = await qse.agent_process_email(persona, email, sem, neutral=True)
        except Exception as exc:
            cnt[f"{kind(email)}_writer_error"] += 1
            return
        k = kind(email)
        cnt[f"{k}_messages"] += 1
        cnt[f"{k}_messages_with_write"] += int(bool(notes))
        for note in notes:
            audit = await qse.memory_stateguard_audit(email, note, sem)
            cnt[f"{k}_writes"] += 1
            cnt[f"{k}_rolled_back"] += int(bool(audit["dangerous"]))

    await asyncio.gather(*(one(e) for e in emails))
    print(PID, style, dict(sorted(cnt.items())), flush=True)
    return dict(sorted(cnt.items()))


async def main():
    out = {}
    for style in ("spoofed_first_party", "legit_marketing"):
        out[style] = await run_style(style)
    OUT.write_text(json.dumps(out, indent=1))


asyncio.run(main())
