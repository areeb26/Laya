"""Build a synthetic bootstrap dataset for the `leads` and `viral` presets.

Rows follow docs/finetune.md: {"state": {...}, "questions": {...}, "expected": {...}}.
The labels come from the templates below, so this data only teaches the shape of the task.
Replace or mix in real labelled rows (CRM outcomes, post analytics) before trusting a score.

    python domains/build_dataset.py --task leads --n 3000 --out domains/data
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from laya.presets import lead_scoring_questions, viral_ai_content_questions  # noqa: E402

ROLES = {"decision": ["Founder", "CEO", "VP of Sales", "Head of Growth", "CTO"],
         "other": ["Intern", "Student", "Analyst", "Freelance writer", "Recruiter"]}
INDUSTRIES = {"fit": ["B2B SaaS", "e-commerce", "fintech", "marketing agency"],
              "poor": ["a local bakery", "a university club", "a hobby blog"]}
SIZES = {"fit": ["50 people", "200 employees", "a Series A team of 30"],
         "poor": ["just me", "2 people", "10,000 employees with a locked vendor list"]}
ASKS = {"ready_to_buy": ["Can you send pricing and a contract? We want to start next month.",
                         "Please book a demo this week, we are ready to sign.",
                         "We need this live by the 1st. What is the annual price?"],
        "evaluating": ["We are comparing you with two other vendors. Does it integrate with Salesforce?",
                       "How does your tool differ from the alternatives on security and API access?"],
        "researching": ["Just exploring what is out there, no timeline yet.",
                        "Curious how this works in general. Maybe next year."],
        "not_a_lead": ["I am looking for a job on your team.", "We can offer you SEO services at a great rate.",
                       "My login is broken, please reset my password."]}
BUDGET = ["Budget is approved.", "We have funding set aside for this quarter.", ""]
NEXT = {"ready_to_buy": "book_call", "evaluating": "send_info", "researching": "nurture",
        "not_a_lead": "disqualify"}


def make_lead(rng):
    intent = rng.choice(list(ASKS))
    role_dec = rng.random() < 0.5
    fit_good = rng.random() < 0.6
    budget = rng.random() < 0.4
    role = rng.choice(ROLES["decision" if role_dec else "other"])
    ind = rng.choice(INDUSTRIES["fit" if fit_good else "poor"])
    size = rng.choice(SIZES["fit" if fit_good else "poor"])
    text = "From: %s at %s (%s). %s %s" % (role, ind, size, rng.choice(ASKS[intent]),
                                          rng.choice(BUDGET[:2]) if budget else "")
    fit = (2 if fit_good else 0) + (1 if fit_good and role_dec else 0) + (0 if fit_good else (1 if role_dec and rng.random() < .3 else 0))
    fit = min(fit, 3)
    nxt = NEXT[intent]
    if intent == "ready_to_buy" and not role_dec and not budget:
        nxt = "send_info"
    return {"state": {"lead": text.strip()}, "questions": lead_scoring_questions(),
            "expected": {"intent": intent, "fit": fit, "has_budget": budget,
                         "decision_maker": role_dec, "next_action": nxt}}


HOOKS = {"contrarian": ["Everyone says prompt engineering is dead. They are wrong.",
                        "Stop using AI agents for everything. Here is why."],
         "numbers": ["I cut my reporting time from 6 hours to 20 minutes with one AI workflow.",
                     "3 models, 1 task, a 41% gap in accuracy."],
         "story": ["Last year I was fired. This year an AI assistant runs my whole research desk.",
                   "My manager laughed at my AI side project. Then it shipped."],
         "how_to": ["How to turn any PDF into a weekly newsletter with Claude in 5 steps:",
                    "Here is the exact setup I use to summarize 50 papers a day:"],
         "news": ["A new open-weights model just dropped and it changes local inference.",
                  "Breaking: the latest model release beats the benchmark leaders on code."],
         "none": ["Thoughts on AI.", "AI is interesting I guess."]}
VALUE = ["Step 1: collect sources. Step 2: ask for a ranked summary. Step 3: paste into your tool.",
         "The prompt template is below, copy it.", "Result: 3x faster, here is the before and after."]
HYPE = ["AI will change everything.", "This is the future. Trust me.", "Mind-blowing, insane, game changer."]
TOOLS = ["Claude", "GPT", "Gemini", "Llama", "Mistral", "Cursor", "n8n", "LangChain", "Whisper", "Midjourney"]
CTA = ["Follow for more.", "Save this for later and share it with a teammate.", "Reply with your use case."]


def make_post(rng):
    hook = rng.choice(list(HOOKS))
    value = hook != "none" and rng.random() < 0.65
    cta = rng.random() < 0.5
    accurate = rng.random() < 0.8
    body = rng.choice(VALUE) if value else rng.choice(HYPE)
    if not accurate:
        body += " AI is now 100x smarter than any human expert."
    tool = rng.choice(TOOLS)
    n = rng.choice([3, 5, 7, 12, 20, 50])
    post = " ".join([rng.choice(HOOKS[hook]), "(%s, %d)" % (tool, n), body] + ([rng.choice(CTA)] if cta else []))
    v = (1 if hook != "none" else 0) + (1 if value else 0) + (1 if cta else 0)
    if not accurate:
        v = max(v - 1, 0)
    return {"state": {"post": post}, "questions": viral_ai_content_questions(),
            "expected": {"hook_type": hook, "virality": min(v, 3), "concrete_value": value,
                         "has_cta": cta, "is_accurate_ai_claim": accurate}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["leads", "viral", "both"], default="both")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--eval-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="domains/data")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    makers = {"leads": make_lead, "viral": make_post}
    os.makedirs(a.out, exist_ok=True)
    for task in (makers if a.task == "both" else [a.task]):
        seen, rows = set(), []
        while len(rows) < a.n:
            r = makers[task](rng)
            key = json.dumps(r["state"], sort_keys=True)
            if key not in seen:
                seen.add(key)
                rows.append(r)
            elif len(seen) < a.n // 2 and rng.random() < 0.01:  # tiny template space: allow repeats
                break
        k = int(len(rows) * a.eval_frac)
        for name, part in (("train", rows[k:]), ("eval", rows[:k])):
            path = os.path.join(a.out, "%s_%s.jsonl" % (task, name))
            with open(path, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in part)
            print(path, len(part))


if __name__ == "__main__":
    main()
