#!/usr/bin/env python
"""What each defense's own check decided, natural-inbox runs.

For every defense, two shares per attack: the share of ATTACK items that its check stops and
the share of GENUINE items that its check stops. A check that separates the two has a high
first number and a low second one.

    python scripts/natural_decisions.py [--latex]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))
import natural_tables as NT  # noqa: E402

LOGS = REPO / "logs"


def pct(a, b):
    return f"{100.0 * a / b:5.1f}" if b else "   --"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    a = ap.parse_args()
    ns = argparse.Namespace(since="2000-01-01 00:00", daily=300, spoof="3", categories=10, order="shuffle",
                            tpb="5", noise=0.0, fwd_note="plain", latex=False, n_full=30)
    cells, _info = NT.load(ns)
    os.environ.setdefault("QWEN_CATALOG_V2", "1")
    from shopping_data_v2 import get_category_v2
    names: dict = {}

    def pname(cid, pid):
        if cid not in names:
            names[cid] = {p["id"]: p["name"] for p in get_category_v2(cid)["products"]}
        return names[cid].get(pid, "")

    san = {}
    sp = LOGS / "tables" / "sanitization_breakdown.json"
    if sp.exists():
        san = json.loads(sp.read_text())
    res: dict = {}
    for inbox, style in (("sfp", "spoofed_first_party"), ("tpb", "legit_marketing")):
        rows = defaultdict(list)
        for (i, d, b, _p, _c), r in cells.items():
            if i == inbox:
                rows[d].append(r)
        c: dict = {}
        t = (san.get(style) or {}).get("total") or {}
        if t:
            for k in ("attack", "genuine", "everyday"):
                c[f"san_{k}"] = (t.get(f"{k}_blocked", 0), t.get(f"{k}_blocked", 0) + t.get(f"{k}_allowed", 0))
        # StateGuard: per-kind counters when the run recorded them, totals otherwise
        per = {(r["backend"], r["persona_id"]): r for r in rows.get("memory_stateguard", [])}
        for k in ("attack", "genuine", "everyday"):
            if any(f"stateguard_n_candidate_{k}" in r for r in per.values()):
                c[f"sg_{k}"] = (sum(r.get(f"stateguard_n_rolled_back_{k}", 0) or 0 for r in per.values()),
                                sum(r.get(f"stateguard_n_candidate_{k}", 0) or 0 for r in per.values()))
        c["sg_all"] = (sum(r.get("stateguard_n_rolled_back", 0) or 0 for r in per.values()),
                       sum(r.get("stateguard_n_candidate", 0) or 0 for r in per.values()))
        c["sg_backends"] = len({b for b, _p in per})
        af = an = gf = gn = 0
        for r in rows.get("amemguard", []):
            tname = (r.get("target_product_name") or pname(r["category_id"], r["target_product_id"])).casefold()
            gname = pname(r["category_id"], r.get("preference_product_id")).casefold()
            for entries, flagged in ((r.get("amemguard_flagged_entries") or [], 1),
                                     (r.get("amemguard_kept_entries") or [], 0)):
                for e in entries:
                    e = str(e).casefold()
                    has_t, has_g = bool(tname) and tname in e, bool(gname) and gname in e
                    if has_t and not has_g:
                        an, af = an + 1, af + flagged
                    elif has_g and not has_t:
                        gn, gf = gn + 1, gf + flagged
        c["amg_attack"], c["amg_genuine"] = (af, an), (gf, gn)
        # action-level checks: proposals that name the target / the user's product
        for key, dname, stopped in (("ppmf", "action_gate", lambda r: r.get("action_gate_allowed") is False),
                                    ("guard", "strict_guard", lambda r: bool(r.get("read_guard_confirmation_required")))):
            rs = rows.get(dname, [])
            ta = [r for r in rs if r.get("read_guard_memory_is_target")]
            ge = [r for r in rs if r.get("read_guard_memory_is_preference")]
            c[f"{key}_attack"] = (sum(stopped(r) for r in ta), len(ta))
            c[f"{key}_genuine"] = (sum(stopped(r) for r in ge), len(ge))
            c[f"{key}_all"] = (sum(stopped(r) for r in rs), len(rs))
        res[inbox] = c
    lines = ["Share (%) of attack items / genuine items stopped by each defense's own check (natural inbox)."]
    for inbox in ("sfp", "tpb"):
        c = res[inbox]
        lines.append(f"\n== {inbox.upper()} ==")
        for label, ka, kg in (("Memory Sanitization (messages)", "san_attack", "san_genuine"),
                              ("StateGuard (memory writes)", "sg_attack", "sg_genuine"),
                              ("A-MemGuard (memory entries)", "amg_attack", "amg_genuine"),
                              ("PPMF (recommendations)", "ppmf_attack", "ppmf_genuine"),
                              ("Counterfactual guard (recommendations)", "guard_attack", "guard_genuine")):
            x, y = c.get(ka, (0, 0)), c.get(kg, (0, 0))
            lines.append(f"{label:<42} attack {pct(*x)} ({x[0]}/{x[1]})   genuine {pct(*y)} ({y[0]}/{y[1]})")
        if "san_everyday" in c:
            x = c["san_everyday"]
            lines.append(f"{'  Sanitization, everyday messages':<42} blocked {pct(*x)} ({x[0]}/{x[1]})")
        x = c["sg_all"]
        lines.append(f"{'  StateGuard, all writes':<42} rolled back {pct(*x)} ({x[0]}/{x[1]}) on {c['sg_backends']} frameworks")
        for k in ("ppmf_all", "guard_all"):
            x = c[k]
            lines.append(f"{'  ' + k:<42} {pct(*x)} ({x[0]}/{x[1]})")
    text = "\n".join(lines)
    print(text)
    (LOGS / "tables" / "decisions.txt").write_text(text + "\n")
    (LOGS / "tables" / "decisions.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
