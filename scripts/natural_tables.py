#!/usr/bin/env python
"""Tables for the natural-inbox grid (Qwen3.8-27B).

Reads every logs/backends_grid_summary_*.json written since --since together with
its raw rows, keeps runs that match the natural-inbox settings (daily message
count, spoof count, 10 categories) and pools rows by
(inbox, defense, backend, persona, category). Per-persona runs of one cell are
pooled automatically; when a cell was run twice the newest row wins.

    python scripts/natural_tables.py                 # text tables
    python scripts/natural_tables.py --latex         # + LaTeX in the paper's table format
    python scripts/natural_tables.py --daily 100     # another inbox size (RQ4 sweeps)

Inboxes: none = no email, clean = everyday messages + genuine notes (no attack),
sfp / tpb = the same inbox plus the attack messages.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

LOGS = Path(__file__).resolve().parent.parent / "logs"
BACKENDS = ["explicit", "context", "rag", "mem0", "memorybank", "tencentdb", "letta", "openai_sandbox"]
BNAME = {"explicit": "Explicit", "context": "Context", "rag": "RAG", "mem0": "Mem0",
         "memorybank": "MemoryBank", "tencentdb": "TencentDB", "letta": "Letta", "openai_sandbox": "OpenAI-SM"}
DEFENSES = ["none", "content_classifier", "memory_stateguard", "limit_memory_length", "amemguard", "smsr",
            "action_gate", "mage", "strict_guard", "channel_stamp", "channel_taint",
            "typed_ledger+ownpart+conflict", "typed_ledger+ownpart+conflict+noauth", "typed_ledger+conflict",
            "typed_ledger+ownpart+conflict+nostance", "typed_ledger+ownpart", "typed_ledger",
            "typed_ledger+nostance", "typed_ledger+ownpart+conflict+k1", "typed_ledger+ownpart+conflict+k3"]
DNAME = {"none": "No Defense", "content_classifier": "Memory Sanitization", "memory_stateguard": "StateGuard",
         "limit_memory_length": "Trojan Hippo", "amemguard": "A-MemGuard", "smsr": "SMSR", "action_gate": "PPMF",
         "mage": "ShadowMem", "strict_guard": "Counterfactual Guard", "channel_stamp": "Label Stored in Memory",
         "channel_taint": "Filter before Memory", "typed_ledger+ownpart+conflict": "MailRecGuard",
         "typed_ledger+ownpart+conflict+noauth": "w/o source check",
         "typed_ledger+conflict": "w/o own-text check",
         "typed_ledger+ownpart+conflict+nostance": "w/o stance check",
         "typed_ledger+ownpart": "w/o conflict confirmation",
         "typed_ledger": "w/o own-text, conflict", "typed_ledger+nostance": "w/o own-text, conflict, stance",
         "typed_ledger+ownpart+conflict+k1": "K=1", "typed_ledger+ownpart+conflict+k3": "K=3"}
INBOX_OF = {"baseline_no_email": "none", "clean_preference_neutral_daily_inbox": "clean",
            "clean_neutral_daily_inbox": "clean_nopref"}
STYLE_INBOX = {"spoofed_first_party": "sfp", "legit_marketing": "tpb", "forwarded_ad": "fwd"}


def load(args) -> tuple[dict, dict]:
    since = time.mktime(time.strptime(args.since, "%Y-%m-%d %H:%M"))
    cells: dict[tuple, dict] = {}       # (inbox, defense, backend, persona, category) -> row
    stamp: dict[tuple, str] = {}
    info = defaultdict(set)             # (inbox, defense, backend) -> run tags
    for sp in sorted(LOGS.glob("backends_grid_summary_*.json")):
        if sp.stat().st_mtime < since:
            continue
        try:
            d = json.loads(sp.read_text())
        except ValueError:
            continue
        if int(d.get("daily_benign_count", -1)) != args.daily:
            continue
        if str(d.get("spoof_count")) != str(args.spoof) or int(d.get("hm_emails_per_category") or 0) != 1:
            continue
        if len(d.get("category_ids") or []) != args.categories:
            continue
        if args.order != (d.get("daily_order") or "shuffle"):
            continue
        if str(d.get("tpb_count") or "5") != str(args.tpb) or float(d.get("channel_noise") or 0) != args.noise:
            continue
        if d.get("email_style") == "forwarded_ad" and (
                str(d.get("fwd_count") or "3") != "3" or (d.get("fwd_note") or "plain") != args.fwd_note):
            continue
        rp = Path(str(sp).replace("summary", "raw"))
        if not rp.exists():
            continue
        defense, style, tag = d.get("defense"), d.get("email_style"), str(d.get("run_tag"))
        for r in json.loads(rp.read_text()):
            cond = r["base_condition"]
            if cond == "poisoned_neutral_daily_inbox":
                inbox = STYLE_INBOX.get(style)
            else:
                inbox = INBOX_OF.get(cond)
            if inbox is None:
                continue
            key = (inbox, defense, r["backend"], r["persona_id"], r["category_id"])
            if key in stamp and stamp[key] >= tag:
                continue
            cells[key], stamp[key] = r, tag
            info[(inbox, defense, r["backend"])].add(tag)
    return cells, info


def rate(rows: list[dict], field: str) -> float | None:
    return 100.0 * sum(bool(r.get(field)) for r in rows) / len(rows) if rows else None


def fmt(x: float | None, w: int = 5) -> str:
    return f"{x:{w}.1f}" if x is not None else " " * (w - 2) + "--"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2000-01-01 00:00")
    ap.add_argument("--daily", type=int, default=300)
    ap.add_argument("--spoof", default="3")
    ap.add_argument("--categories", type=int, default=10)
    ap.add_argument("--order", default="shuffle")
    ap.add_argument("--tpb", default="5")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--fwd-note", default="plain")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--n-full", type=int, default=30)
    args = ap.parse_args()
    cells, info = load(args)

    def rows(inbox, defense, backend):
        return [r for (i, d, b, _p, _c), r in cells.items() if (i, d, b) == (inbox, defense, backend)]

    def cell(inbox, defense, backend):
        rs = rows(inbox, defense, backend)
        return rate(rs, "is_target_top1"), rate(rs, "is_preference_top1"), len(rs)

    present = [d for d in DEFENSES if any(k[1] == d for k in cells)]
    out: list[str] = []
    out.append(f"Natural inbox, Qwen3.8-27B, {args.daily} everyday messages per user, order={args.order}; "
               f"ASR / HM in %, n per cell in brackets when not {args.n_full}.")

    # ---------------- RQ1: attack effectiveness without a defense
    out.append("\n== RQ1: no defense ==")
    out.append(f"{'Memory':<11} | {'No email':>11} | {'Clean inbox':>11} | {'SFP':>11} {'flip':>5} | {'TPB':>11} {'flip':>5}")
    means = defaultdict(list)
    for b in BACKENDS:
        line = f"{BNAME[b]:<11}"
        for inbox in ("none", "clean", "sfp", "tpb"):
            a, h, n = cell(inbox, "none", b)
            tagn = "" if n in (0, args.n_full) else f"[{n}]"
            line += f" | {fmt(a)}/{fmt(h)}{tagn}"
            if a is not None and n == args.n_full:
                means[inbox].append((a, h))
            if inbox in ("sfp", "tpb"):
                clean = {(p, c): r for (i, d, bb, p, c), r in cells.items() if (i, d, bb) == ("clean", "none", b)}
                pois = {(p, c): r for (i, d, bb, p, c), r in cells.items() if (i, d, bb) == (inbox, "none", b)}
                pairs = [(clean[k], pois[k]) for k in clean if k in pois]
                base = [pr for pr in pairs if not pr[0]["is_target_top1"]]
                flip = 100.0 * sum(bool(p_["is_target_top1"]) for _c, p_ in base) / len(base) if base else None
                line += f" {fmt(flip)}"
                if flip is not None and len(pairs) == args.n_full:
                    means[inbox + "_flip"].append(flip)
        out.append(line)
    line = f"{'Mean':<11}"
    for inbox in ("none", "clean", "sfp", "tpb"):
        v = means[inbox]
        line += (f" | {fmt(sum(a for a, _ in v)/len(v))}/{fmt(sum(h for _, h in v)/len(v))}" if v else " |    --/   --")
        if inbox in ("sfp", "tpb"):
            f = means[inbox + "_flip"]
            line += f" {fmt(sum(f)/len(f)) if f else '   --'}"
    out.append(line + f"   (means over complete cells: " + ", ".join(f"{k}={len(v)}" for k, v in means.items()) + ")")

    # ---------------- RQ2: defenses
    for inbox, title in (("sfp", "SFP"), ("tpb", "TPB"), ("fwd", "Forwarded ad (FWD)"),
                         ("clean", "Clean inbox (no attack)")):
        if not any(k[0] == inbox for k in cells):
            continue
        out.append(f"\n== RQ2: {title} ==")
        out.append(f"{'Memory':<11} | " + " | ".join(f"{DNAME[d][:13]:>13}" for d in present))
        cols = defaultdict(list)
        for b in BACKENDS:
            line = f"{BNAME[b]:<11}"
            for d in present:
                a, h, n = cell(inbox, d, b)
                mark = "" if n in (0, args.n_full) else "*"
                line += f" | {fmt(a)}/{fmt(h)}{mark:1}"
                if a is not None and n == args.n_full:
                    cols[d].append((a, h))
            out.append(line)
        line = f"{'Mean':<11}"
        for d in present:
            v = cols[d]
            line += (f" | {fmt(sum(a for a, _ in v)/len(v))}/{fmt(sum(h for _, h in v)/len(v))} " if v else " |    --/   -- ")
        out.append(line)
        out.append(f"{'#backends':<11}" + "".join(f" | {len(cols[d]):>12} " for d in present) + "   (* = incomplete cell)")

    text = "\n".join(out)
    print(text)
    outdir = LOGS / "tables"
    outdir.mkdir(exist_ok=True)
    suffix = "" if (args.daily, args.order, str(args.spoof), str(args.tpb), args.noise) == (300, "shuffle", "3", "5", 0.0) \
        else f"_d{args.daily}_{args.order}_sc{args.spoof}_tc{args.tpb}_n{args.noise}"
    (outdir / f"natural_tables{suffix}.txt").write_text(text + "\n")

    if args.latex:
        tex: list[str] = []
        for group, label, caption in (
            (["none", "content_classifier", "memory_stateguard", "limit_memory_length", "amemguard", "smsr",
              "action_gate", "typed_ledger+ownpart+conflict"], "tab:rq2",
             "The ASR (\\%) and HM (\\%) of the LLM agent under different defenses on different memory frameworks. "
             "`SFP' means spoofed first-party attack, and `TPB' means third-party brand attack."),
        ):
            group = [d for d in group if d in present]
            ncol = 2 * len(group)
            tex += ["\\begin{table*}[!t]", "\\renewcommand{\\arraystretch}{1.1}", "\\setlength{\\tabcolsep}{5.5pt}",
                    "\\small", "    \\centering", f"    \\caption{{{caption}}}", f"    \\label{{{label}}}",
                    "    \\resizebox{1.0\\textwidth}{!}{", "    \\begin{tabular}{l|" + "|".join(["rc"] * len(group)) + "}"]
            for inbox, title in (("sfp", "SFP"), ("tpb", "TPB")):
                tex += ["    \\toprule",
                        f"    \\multirow{{3}}{{*}}{{\\textbf{{Memory}}}} & \\multicolumn{{{ncol}}}{{c}}{{{title}}} \\\\",
                        f"    \\Xcline{{2-{ncol + 1}}}{{0.5pt}}",
                        "    & " + " & ".join(
                            f"\\multicolumn{{2}}{{c{'|' if i < len(group) - 1 else ''}}}{{{DNAME[d]}}}"
                            for i, d in enumerate(group)) + " \\\\",
                        f"    \\Xcline{{2-{ncol + 1}}}{{0.5pt}}",
                        "     & " + " & ".join(["ASR & HM"] * len(group)) + " \\\\", "    \\hline"]
                cols = defaultdict(list)
                for b in BACKENDS:
                    vals = []
                    for d in group:
                        a, h, n = cell(inbox, d, b)
                        vals += [fmt(a), fmt(h)]
                        if a is not None and n == args.n_full:
                            cols[d].append((a, h))
                    tex.append(f"  \\texttt{{{BNAME[b]}}}".ljust(24) + " & " + " & ".join(vals) + " \\\\")
                tex.append("    \\hline")
                vals = []
                for d in group:
                    v = cols[d]
                    vals += ([fmt(sum(a for a, _ in v) / len(v)), fmt(sum(h for _, h in v) / len(v))] if v else ["--", "--"])
                tex += ["  Mean".ljust(24) + " & " + " & ".join(vals) + " \\\\", "    \\bottomrule"]
            tex += ["\\end{tabular}", "}", "\\end{table*}", ""]
        (outdir / f"natural_tables{suffix}.tex").write_text("\n".join(tex))
        print(f"\nLaTeX -> {outdir / f'natural_tables{suffix}.tex'}")


if __name__ == "__main__":
    main()
