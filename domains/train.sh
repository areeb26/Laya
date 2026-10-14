#!/usr/bin/env bash
# Quick path; the full, checked procedure is domains/FINETUNE_ON_PC.md.
#   bash domains/train.sh leads english     # or: viral multilingual
set -euo pipefail
TASK=${1:?leads|viral}; BASE=${2:-english}
laya-train --data "domains/data_agreed/${TASK}_train.jsonl" --eval "domains/data_agreed/${TASK}_eval.jsonl" \
           --base "$BASE" --epochs "${EPOCHS:-30}" --micro-batch "${MICRO_BATCH:-8}" --grad-accum "${GRAD_ACCUM:-4}" \
           --out "checkpoints/laya-${TASK}" ${DRY_RUN:+--dry-run}
