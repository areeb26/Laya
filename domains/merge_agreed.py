"""Keep rows where the Sonnet generator's label and Opus's blind relabel agree on every question.

    python domains/merge_agreed.py   # writes domains/data/{leads,viral}_{train,eval}.jsonl + agreement report
"""
import glob
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from laya.presets import lead_scoring_questions, viral_ai_content_questions  # noqa: E402

TASKS = {"leads": (lead_scoring_questions, "lead"), "viral": (viral_ai_content_questions, "post")}
D = os.path.join(ROOT, "domains", "data_llm")


def load(p):
    return [json.loads(l) for l in open(p)]


def main(seed=0, eval_frac=0.1):
    rng = random.Random(seed)
    out_dir = os.path.join(ROOT, "domains", "data")
    os.makedirs(out_dir, exist_ok=True)
    report = {}
    for task, (qfn, key) in TASKS.items():
        kept, tot = [], 0
        field_hits, field_n = {}, {}
        for gen in sorted(glob.glob(os.path.join(D, task + "_gen_*.jsonl"))):
            opus_path = os.path.join(D, "blind", os.path.basename(gen).replace(".jsonl", ".opus.jsonl"))
            if not os.path.exists(opus_path):
                print("missing", opus_path)
                continue
            rows = load(gen)
            opus = {r["id"]: r["expected"] for r in load(opus_path)}
            for i, r in enumerate(rows):
                o = opus.get(i)
                if o is None:
                    continue
                tot += 1
                agree = True
                for k, v in r["expected"].items():
                    field_n[k] = field_n.get(k, 0) + 1
                    if o.get(k) == v:
                        field_hits[k] = field_hits.get(k, 0) + 1
                    else:
                        agree = False
                if agree:
                    kept.append({"state": {key: r["text"]}, "questions": qfn(), "expected": r["expected"]})
        rng.shuffle(kept)
        k = int(len(kept) * eval_frac)
        for name, part in (("train", kept[k:]), ("eval", kept[:k])):
            with open(os.path.join(out_dir, "%s_%s.jsonl" % (task, name)), "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in part)
        report[task] = {"rows": tot, "all_fields_agree": len(kept),
                        "per_field_agreement": {k: round(field_hits[k] / field_n[k], 3) for k in field_n}}
    print(json.dumps(report, indent=2))
    json.dump(report, open(os.path.join(D, "agreement_report.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
