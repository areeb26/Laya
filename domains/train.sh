#!/usr/bin/env bash
# Fine-tune Laya on the lead-generation and viral-AI-content tasks. Needs a GPU for a 421M run
# (a 2xT4 or one 16 GB card; CPU works only for --dry-run) and access to huggingface.co.
#   bash domains/train.sh leads english     # or: viral multilingual
set -euo pipefail
TASK=${1:?leads|viral}; BASE=${2:-english}
python domains/build_dataset.py --task "$TASK" --n "${N:-3000}" --out domains/data
laya-train --data "domains/data/${TASK}_train.jsonl" --base "$BASE" --epochs "${EPOCHS:-6}" \
           --out "checkpoints/laya-${TASK}" ${DRY_RUN:+--dry-run}
laya-evals run --help >/dev/null 2>&1 && echo "evaluate with: laya-evals run (see docs) on domains/data/${TASK}_eval.jsonl"
