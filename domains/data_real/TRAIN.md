# Train Laya on the gold-label set (GPU laptop)

Instructions for whoever (or whichever Claude Code session) runs this on a GPU machine.
General procedure and rules are in `domains/FINETUNE_ON_PC.md`; this file only swaps in the gold data.

## What the data is

`domains/data_real/` holds hand-written, single-labeller examples (written by Claude Sonnet 5.5,
no second-model check). Sources are `leads_raw.txt` and `viral_raw.txt`;
`python domains/build_gold.py` rebuilds the JSONL (fixed seed).

| task | train | eval |
|---|---|---|
| leads | 187 | 40 |
| viral | 155 | 40 |

Rules that still apply: never train on any `*_eval.jsonl`, do not edit eval files or criteria
text to improve numbers, do not upload checkpoints, and report results as measured (40 eval rows =
a few points of noise; `fit`, `hook_type`, `virality` are subjective).

## 1. Setup

```bash
git checkout claude/laya-leads-viral && git pull
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e .          # NVIDIA: install the CUDA torch build from pytorch.org first
python -c "import torch;print(torch.cuda.is_available())"
python domains/build_gold.py          # must print leads {'eval': 40, 'train': 187}, viral {'eval': 40, 'train': 155}
laya-evals validate domains/data_real/leads_eval.jsonl
laya-evals validate domains/data_real/viral_eval.jsonl
```

Batch settings: >=16 GB VRAM `--micro-batch 8 --grad-accum 4`; 8-12 GB `--micro-batch 2 --grad-accum 16`
(on OOM: 1 / 32). Apple Silicon: `--device mps --micro-batch 1 --grad-accum 32`.

## 2. Baselines (before training)

```bash
mkdir -p checkpoints reports
for t in leads viral; do
  laya-evals run domains/data_real/${t}_eval.jsonl --model english --json reports/gold_${t}_baseline.json
done
```

## 3. Run A: gold data only

```bash
for t in leads viral; do
  laya-train --data domains/data_real/${t}_train.jsonl --eval domains/data_real/${t}_eval.jsonl \
    --base convaiinnovations/laya --out checkpoints/gold-${t} \
    --epochs 30 --micro-batch 8 --grad-accum 4 --device auto --seed 0 2>&1 | tee reports/gold_${t}_train.log
done
```

(Try `--dry-run` first. Aim for ~300+ optimizer updates; raise `--epochs` if the dry run shows fewer.)

## 4. Run B: gold + agreed data mixed

`laya-train` takes one `--data` file, so concatenate the train files (eval stays the gold eval):

```bash
mkdir -p checkpoints/mixed_data
for t in leads viral; do
  cat domains/data_agreed/${t}_train.jsonl domains/data_real/${t}_train.jsonl > checkpoints/mixed_data/${t}_train.jsonl
  laya-train --data checkpoints/mixed_data/${t}_train.jsonl --eval domains/data_real/${t}_eval.jsonl \
    --base convaiinnovations/laya --out checkpoints/mixed-${t} \
    --epochs 15 --micro-batch 8 --grad-accum 4 --device auto --seed 0 2>&1 | tee reports/mixed_${t}_train.log
done
```

## 5. Evaluate

```bash
for run in gold mixed; do for t in leads viral; do
  laya-evals run domains/data_real/${t}_eval.jsonl --model checkpoints/${run}-${t} \
    --baseline reports/gold_${t}_baseline.json --slice qid \
    --json reports/${run}_${t}_tuned.json --markdown reports/${run}_${t}_tuned.md
done; done
```

Also score each tuned checkpoint on `domains/data_agreed/${t}_eval.jsonl` to see transfer to the
Sonnet/Opus-agreed labels.

## 6. Report

Per task and per question: baseline vs gold-only vs mixed accuracy on the gold eval, plus the
agreed-eval number. State which questions stay near the class prior, and mention the calibration
warnings the runs print. Do not claim 99%. Do not commit `checkpoints/` or `reports/` unless asked.
