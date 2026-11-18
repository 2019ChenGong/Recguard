#!/usr/bin/env python
"""What happens when the source label is wrong with probability p?

Pools the natural-inbox runs (since --since) of the sender filter and of the typed ledger
with / without the conflict rule over error patterns (seeds), users and memory frameworks.

    python scripts/noise_table.py [--since "2000-01-01 00:00"] [--backends local|all]

Columns: ASR / HM (%), n queries, confirmation requests per user (out of 20 preferences),
attack records admitted per user.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

LOGS = Path(__file__).resolve().parent.parent / "logs"
LOCAL = {"explicit", "context", "rag", "mem0", "tencentdb", "memorybank"}
ARMS = [("channel_taint", "Sender filter"),
        ("typed_ledger", "Ours, no conflict rule (no own-text rule)"),
        ("typed_ledger+ownpart", "Ours, no conflict rule"),
        ("typed_ledger+ownpart+conflict", "Ours, conflict -> user confirms"),
        ("typed_ledger+ownpart+conflictna", "Ours, conflict -> no answer")]
STYLE = {"spoofed_first_party": "SFP", "legit_marketing": "TPB"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2000-01-01 00:00")
    ap.add_argument("--backends", default="local")
    args = ap.parse_args()
    since = time.mktime(time.strptime(args.since, "%Y-%m-%d %H:%M"))
    rows: dict[tuple, dict] = {}     # (style, defense, noise, seed, backend, persona, category) -> (tag, row)
    for sp in sorted(LOGS.glob("backends_grid_summary_*.json")):
        if sp.stat().st_mtime < since:
            continue
        try:
            d = json.loads(sp.read_text())
        except ValueError:
            continue
        if int(d.get("daily_benign_count", -1)) != 300 or (d.get("daily_order") or "shuffle") != "shuffle":
            continue
        if str(d.get("spoof_count")) != "3" or str(d.get("tpb_count") or "5") != "5":
            continue
        if len(d.get("category_ids") or []) != 10 or d.get("email_style") not in STYLE:
            continue
        defense = d.get("defense")
        if defense not in dict(ARMS):
            continue
        noise = float(d.get("channel_noise") or 0)
        fr = d.get("channel_noise_fr")
        fr = noise if fr is None else float(fr)
        mode = "sym" if fr == noise else f"fr{fr:g}"
        seed = str(d.get("channel_noise_seed") or "0") if noise > 0 else "-"
        rp = Path(str(sp).replace("summary", "raw"))
        if not rp.exists():
            continue
        for r in json.loads(rp.read_text()):
            if r["base_condition"] != "poisoned_neutral_daily_inbox":
                continue
            if args.backends == "local" and r["backend"] not in LOCAL:
                continue
            key = (STYLE[d["email_style"]], defense, noise, mode, seed, r["backend"], r["persona_id"], r["category_id"])
            tag = str(d.get("run_tag"))
            if key in rows and rows[key][0] >= tag:
                continue
            rows[key] = (tag, r)
    cells = defaultdict(list)
    for (style, defense, noise, mode, seed, backend, persona, _cat), (_tag, r) in rows.items():
        cells[(style, defense, noise, mode)].append((seed, backend, persona, r))
        if noise == 0:   # the error-free runs are the p = 0 column of every block
            cells[(style, defense, noise, "fr0")].append((seed, backend, persona, r))
    out = [f"Label error rate sweep, natural inbox, {args.backends} frameworks. "
           "ASR / HM in %, conf = confirmation requests per user, adm = attack records admitted per user."]
    for style, mode, title in (("SFP", "sym", "false accepts and false rejects at rate p"),
                               ("TPB", "sym", "false accepts and false rejects at rate p"),
                               ("SFP", "fr0", "false accepts at rate p, no false rejects"),
                               ("TPB", "fr0", "false accepts at rate p, no false rejects")):
        out.append(f"\n== {style}: {title} ==")
        out.append(f"{'Defense':<42} " + " ".join(f"{'p=' + str(p):>30}" for p in (0.0, 0.05, 0.1, 0.2)))
        for defense, name in ARMS:
            line = f"{name:<42} "
            for p in (0.0, 0.05, 0.1, 0.2):
                rs = cells.get((style, defense, p, mode), [])
                if not rs:
                    line += f"{'--':>30} "
                    continue
                n = len(rs)
                asr = 100 * sum(bool(r["is_target_top1"]) for *_x, r in rs) / n
                hm = 100 * sum(bool(r["is_preference_top1"]) for *_x, r in rs) / n
                per_user: dict[tuple, tuple] = {}
                for seed, backend, persona, r in rs:
                    per_user[(seed, backend, persona)] = (r.get("tl_confirmation_requests"), r.get("tl_attack_admitted"))
                conf = [c for c, _a in per_user.values() if c is not None]
                adm = [a for _c, a in per_user.values() if a is not None]
                extra = (f" conf {sum(conf)/len(conf):.1f} adm {sum(adm)/len(adm):.1f}" if conf else "")
                line += f"{f'{asr:5.1f}/{hm:5.1f} n={n}{extra}':>30} "
            out.append(line)
    text = "\n".join(out)
    print(text)
    (LOGS / "tables").mkdir(exist_ok=True)
    (LOGS / "tables" / f"noise_table_{args.backends}.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
