# Fine-tune Laya on this PC (instructions for Claude Code)

You are running on the user's own computer, in a clone of this repo. Fine-tune two Laya
checkpoints, one for **lead qualification** (`leads`) and one for **judging AI posts for viral
potential** (`viral`), then evaluate them honestly. Work through the steps in order. Run commands
yourself; ask the user only where this doc says to.

Read `domains/README.md` first for context. Key facts: Laya answers typed questions in one forward
pass and cannot write text. The data in `domains/data_agreed/` was written by Sonnet and kept only
where an independent Opus relabel agreed, per question. Some labels (`fit`, `hook_type`,
`virality`) are subjective, so **do not promise 99%**; the Sonnet–Opus agreement on them is
62–79%, which is the realistic ceiling.

## 0. Rules

* Do not edit `domains/data_agreed/*_eval.jsonl`, the criteria text in `laya/presets.py`, or the
  metric definitions to make numbers look better. Eval files are never trained on.
* Report results exactly as measured, including when they are worse than the baseline.
* Do not upload checkpoints anywhere, push checkpoints to git, or send data to any service
  unless the user asks. `checkpoints/` is git-ignored; keep it that way.
* Before finishing any change to `laya/`, run the CI gates in `AGENTS.md`.

## 1. Preflight

```bash
git pull origin main
python --version            # 3.10+
nvidia-smi || true          # NVIDIA GPU? note VRAM
python -c "import torch;print(torch.__version__, torch.cuda.is_available(), getattr(torch.backends,'mps',None) and torch.backends.mps.is_available())"
```

Install, using a virtualenv:

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .
# NVIDIA GPU: if torch.cuda.is_available() is False, install the CUDA build from pytorch.org first.
```

Check the Hugging Face access that downloading the base model needs:
`python -c "from huggingface_hub import hf_hub_download as d; print(d('convaiinnovations/laya','README.md'))"`.
If it fails, tell the user and stop.

Pick the device and batch settings:

| hardware | `--device` | `--micro-batch` | `--grad-accum` | base |
|---|---|---|---|---|
| NVIDIA >= 16 GB | `cuda` | 8 | 8 | `convaiinnovations/laya` (English) |
| NVIDIA 8-12 GB | `cuda` | 2 | 32 | same; on OOM use 1 / 64 |
| Apple Silicon | `mps` | 1 | 32 | same; or `notebooks/laya_finetune_typed_decisions_mps.py` |
| CPU only | `cpu` | 1 | 32 | do `--dry-run` only and tell the user a GPU is needed |

The leads and posts are mostly English, but the lead set includes some other languages. Train the
English base first; if the user wants non-English leads, repeat with
`--base convaiinnovations/laya-multilingual` (322M, about 2x faster) and compare.

## 2. Validate the data

```bash
python domains/merge_agreed.py              # rebuilds data_agreed/ and prints agreement; must match domains/README.md
laya-evals validate domains/data_agreed/leads_eval.jsonl
laya-evals validate domains/data_agreed/viral_eval.jsonl
laya-train --data domains/data_agreed/leads_train.jsonl --base convaiinnovations/laya \
           --dry-run --micro-batch 8 --grad-accum 8
```

The dry run prints how many items were built, what was skipped and why, and the optimizer-update
budget. The training sets are small (450 rows per task), so use more epochs than the default 4:
aim for roughly 300+ optimizer updates (`updates = epochs * ceil(ceil(items/micro_batch)/grad_accum)`).
With `--micro-batch 8 --grad-accum 8` that is about 7 updates per epoch, so use `--epochs 30`
for a first run, or lower `--grad-accum` to 2-4 for more updates per epoch. If the dry run skips
many items, fix the cause before training and tell the user.

## 3. Baseline (before training)

```bash
mkdir -p checkpoints reports
laya-evals run domains/data_agreed/leads_eval.jsonl --model english --json reports/leads_baseline.json
laya-evals run domains/data_agreed/viral_eval.jsonl --model english --json reports/viral_baseline.json
```

## 4. Train

One checkpoint per task. Use `--eval` so the run prints before/after on the held-out file.

```bash
laya-train --data domains/data_agreed/leads_train.jsonl --eval domains/data_agreed/leads_eval.jsonl \
           --base convaiinnovations/laya --out checkpoints/laya-leads \
           --epochs 30 --micro-batch 8 --grad-accum 4 --device auto --seed 0 2>&1 | tee reports/leads_train.log

laya-train --data domains/data_agreed/viral_train.jsonl --eval domains/data_agreed/viral_eval.jsonl \
           --base convaiinnovations/laya --out checkpoints/laya-viral \
           --epochs 30 --micro-batch 8 --grad-accum 4 --device auto --seed 0 2>&1 | tee reports/viral_train.log
```

Adjust batch settings per the table in step 1. Each run writes `checkpoint_latest/` after every
epoch, so a crash or reboot costs at most one epoch; to resume, rerun with the same `--out`.
Run long jobs in the background and poll the log.

Watch for: a run that stays at the class prior (see "Diagnose a run that stays at chance" in
`docs/finetune.md`), and the calibration warnings the run prints (a "not fitted" warning means
the confidences are uncalibrated; do not recommend confidence thresholds then).

## 5. Evaluate

```bash
laya-evals run domains/data_agreed/leads_eval.jsonl --model checkpoints/laya-leads \
    --baseline reports/leads_baseline.json --slice qid --json reports/leads_tuned.json --markdown reports/leads_tuned.md
laya-evals run domains/data_agreed/viral_eval.jsonl --model checkpoints/laya-viral \
    --baseline reports/viral_baseline.json --slice qid --json reports/viral_tuned.json --markdown reports/viral_tuned.md
```

If `--model` rejects a local path, load it in Python with `laya.load("checkpoints/laya-leads")`
and score the eval file with `laya.evals`; do not skip the step.

## 6. Report to the user

Give a table per task: per-question accuracy, baseline vs tuned, and the Sonnet-Opus agreement
from `domains/README.md` beside it. State plainly:

* which questions are at or near their agreement ceiling and which are well below it;
* that the eval rows are only labels both teachers agreed on, so real-world accuracy will be
  lower, and that only 50 eval rows per task means a few points of noise;
* whether the user's 99% target was met per question (expected: yes for `has_budget`,
  `decision_maker`, maybe `has_cta`/`is_accurate_ai_claim`; no for `fit`, `hook_type`, `virality`).

Then offer these follow-ups, and do none without asking: train the multilingual base and compare;
a second round with more or user-supplied labelled data (see below); serving the model with
`laya-serve` (Jev-compatible `POST /v1/systemone`).

## 7. Using the user's own data (optional, usually the biggest win)

If the user has real outcomes (CRM won/lost, meeting booked; post impressions/shares), convert
them to JSONL rows of `{"state": {"lead": "..."}, "questions": {...}, "expected": {...}}` in the
format of `domains/data_agreed/*.jsonl` (use `laya.lead_scoring_questions()` /
`laya.viral_ai_content_questions()` for the questions), put them in a new file
`domains/data_real/<task>_train.jsonl` plus a separate `_eval.jsonl`, and repeat steps 2-6,
mixing them with the agreed data. Real-outcome eval rows are the only trustworthy accuracy number.
