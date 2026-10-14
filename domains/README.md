# Lead generation and viral-AI-content domain pack

Laya is a **non-autoregressive decision model**: one forward pass answers typed questions
(`choice`, `score`, `noul`). It cannot write text. This pack therefore trains it to *judge*:

| preset | state key | what it decides |
|---|---|---|
| `leads` (`laya.lead_scoring_questions`) | `lead` | buying intent, ICP fit, budget, decision maker, next sales action |
| `viral` (`laya.viral_ai_content_questions`) | `post` | hook type, share likelihood, concrete value, CTA, claim accuracy |

For viral content, pair it with a generator LLM: generate N drafts, score each with the `viral`
preset, keep the best. Laya is the filter and ranker.

## Jev compatibility

Answers use the same `choice`/`score`/`noul` payload as TypeSafe Jev, and `laya.serve`
exposes `POST /v1/systemone`, so a Jev client works against the fine-tuned model unchanged.
Differences: `confidence` is defined differently (see the main README) and a question is capped
by the checkpoint's option budget. Matching Jev's *accuracy* is a measurement, not something this
repo can promise: score both on the same held-out set of your own labelled data.

## Run

```bash
python domains/build_dataset.py --task both --n 3000 --out domains/data
bash domains/train.sh leads english        # GPU + huggingface.co access required
laya "From: CEO at B2B SaaS (200 employees). Send pricing, we start next month." --preset leads
```

## Read this before trusting the numbers

* `build_dataset.py` data is **synthetic and template-generated**. It teaches the schema; it will
  not teach real buyer or audience behaviour, and held-out accuracy on it is inflated because
  train and eval share templates. Replace it with real labelled rows: CRM outcomes
  (won/lost, reply, meeting booked) for leads, and impressions/shares for posts.
* Nothing here has been trained yet: the build environment had no GPU and could not reach
  huggingface.co. Run `train.sh` on a 16 GB GPU or Kaggle 2xT4.
