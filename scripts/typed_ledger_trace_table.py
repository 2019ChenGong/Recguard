#!/usr/bin/env python
"""Where does each message stop inside the defense? (natural-inbox runs)

Reads logs/typed_ledger_trace_<run_tag>.json (written by exp_backends_qwen.py when the
typed ledger is on) and counts, per attack style and message kind, the stage at which a
message stopped:

    auth_fail     Step 1: the sender is not authenticated as the user
    no_category   authenticated, but not about a catalog category (everyday mail)
    no_claim      Step 2: the extractor returned no claim
    binding_fail  Step 2: a claim failed the code checks (catalog product, verbatim quote)
    stance_fail   Step 2: the stance checker did not confirm the user's own preference
    admitted      a preference record was written

Each message is counted once per (style, defense, noise, persona): the transform is cached,
so every backend of a run sees the same decision.

    python scripts/typed_ledger_trace_table.py [--since "2000-01-01 00:00"] [--defense typed_ledger]
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

LOGS = Path(__file__).resolve().parent.parent / "logs"
STAGES = ["auth_fail", "no_category", "no_claim", "binding_fail", "stance_fail", "conflict_rejected", "admitted"]
STYLE = {"spoofed_first_party": "SFP", "legit_marketing": "TPB", "forwarded_ad": "FWD"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2000-01-01 00:00")
    ap.add_argument("--defense", default="typed_ledger")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--min-messages", type=int, default=900,
                    help="skip traces of smoke runs (fewer messages than this)")
    args = ap.parse_args()
    since = time.mktime(time.strptime(args.since, "%Y-%m-%d %H:%M"))
    seen: dict[tuple, str] = {}
    kinds: dict[tuple, str] = {}
    for p in sorted(LOGS.glob("typed_ledger_trace_*.json")):
        if p.stat().st_mtime < since:
            continue
        d = json.loads(p.read_text())
        if d.get("defense") != args.defense or float(d.get("channel_noise") or 0) != args.noise:
            continue
        if "poisoned_neutral_daily_inbox" not in (d.get("conditions_run") or []):
            continue
        if len({(t["persona_id"], t["email_id"]) for t in d["trace"]}) < args.min_messages:
            continue
        style = STYLE.get(d.get("email_style"), d.get("email_style"))
        for t in d["trace"]:
            key = (style, t["persona_id"], t["email_id"])
            seen[key] = t["stage"]
            kinds[key] = t["kind"]
    table: dict[tuple, Counter] = defaultdict(Counter)
    for key, stage in seen.items():
        table[(key[0], kinds[key])][stage] += 1
    lines = [f"Defense = {args.defense}, label noise = {args.noise}. Messages by the stage where they stop.",
             f"{'Attack':<5} {'Message':<9} {'total':>6} | " + " ".join(f"{s:>12}" for s in STAGES)]
    for style in ("SFP", "TPB", "FWD"):
        for kind in ("attack", "genuine", "everyday"):
            c = table.get((style, kind))
            if not c:
                continue
            tot = sum(c.values())
            lines.append(f"{style:<5} {kind:<9} {tot:>6} | " + " ".join(f"{c.get(s, 0):>12}" for s in STAGES))
    text = "\n".join(lines)
    print(text)
    out = LOGS / "tables"
    out.mkdir(exist_ok=True)
    suffix = "" if (args.defense, args.noise) == ("typed_ledger", 0.0) else f"_{args.defense.replace('+', '_')}_n{args.noise}"
    (out / f"trace_table{suffix}.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
