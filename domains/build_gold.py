"""Build gold-label JSONL from domains/data_real/*_raw.txt (hand-written rows).

    python domains/build_gold.py

Writes domains/data_real/{leads,viral}_{train,eval}.jsonl. Fixed seed; eval is held out.
"""
import json, random, sys
from pathlib import Path
import laya

D = Path(__file__).parent / "data_real"
INTENT = dict(R="ready_to_buy", E="evaluating", S="researching", N="not_a_lead")
ACTION = dict(B="book_call", N="nurture", I="send_info", D="disqualify")
HOOK = dict(C="contrarian", N="numbers", S="story", H="how_to", W="news", X="none")
B = lambda s: {"0": False, "1": True}[s]


def parse(path, nfields):
    rows, seen = [], set()
    for ln, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        text, _, rest = line.rpartition(" | ")
        f = [x.strip() for x in rest.split("|")]
        if len(f) != nfields or not text:
            sys.exit(f"{path.name}:{ln}: expected {nfields} label fields: {line[:70]}")
        if text in seen:
            sys.exit(f"{path.name}:{ln}: duplicate text")
        seen.add(text)
        rows.append((text.strip(), f))
    return rows


def leads(r):
    t, (i, fit, bud, dm, act) = r
    return t, {"intent": INTENT[i], "fit": int(fit), "has_budget": B(bud), "decision_maker": B(dm), "next_action": ACTION[act]}


def viral(r):
    t, (h, v, cv, cta, acc) = r
    return t, {"hook_type": HOOK[h], "virality": int(v), "concrete_value": B(cv), "has_cta": B(cta), "is_accurate_ai_claim": B(acc)}


def emit(name, key, qs, rows, conv, n_eval, seed=0):
    items = [conv(r) for r in rows]
    random.Random(seed).shuffle(items)
    splits = {"eval": items[:n_eval], "train": items[n_eval:]}
    for sp, its in splits.items():
        with open(D / f"{name}_{sp}.jsonl", "w") as f:
            for text, exp in its:
                f.write(json.dumps({"state": {key: text}, "questions": qs, "expected": exp}, ensure_ascii=False) + "\n")
    print(name, {k: len(v) for k, v in splits.items()})


emit("leads", "lead", laya.lead_scoring_questions(), parse(D / "leads_raw.txt", 5), leads, 40)
emit("viral", "post", laya.viral_ai_content_questions(), parse(D / "viral_raw.txt", 5), viral, 40)
