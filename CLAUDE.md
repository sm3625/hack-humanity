# Agent guide: job recruitment fraud detection

Read this before changing anything. `README.md` is the human-facing overview; this file is the
working context for an agent continuing the build: how the pieces connect, the rules that keep
the results honest, and what to do next.

## Goal and current state

Classify job postings as fraudulent or legitimate, with a focus on making it work for **India /
South Asia**, even though the only large labeled dataset (EMSCAD) is US/UK-centric.

Best results so far (held-out data, scam rules on):

| Model | US test (3,213 / 203 fraud) | India held-out (120 / 75 fake) |
|---|---|---|
| `model-testing1.py` TF-IDF, shared threshold (the **main** model) | recall 55%, 15 false alarms | acc 90.8%, 10 false alarms |
| `model-embeddings.py` TF-IDF + India threshold (the **saved** embed-script model) | recall 56%, 20 false alarms | acc 92.5%, 7 false alarms |
| `model-embeddings.py` hybrid (TF-IDF + MiniLM) + India threshold | recall 56%, 18 false alarms | acc 94.2%, 3 false alarms |

The hybrid was best on held-out data but lost in cross-validation (mean OOF AP 0.827 vs 0.839),
so it was deliberately **not** promoted. Promote it only on evidence from new data (see Next steps).

## File network

All code is in `trial-modeling-news/`. Paths below are relative to it.

```
                     ┌───────────────────────────┐
  Hugging Face  ───▶ │ convert_hf_india.py       │──writes──▶ india_synthetic_jobs.csv ─┐
  Aioshi/smolified-  │  fetch → convert → score  │                                       │
  fakejob (597 rows) └──────┬──────────▲─────────┘                                       │
                            │ imports  │ loads job_fraud_results/job_fraud_baseline.joblib
                            ▼          │ (skips its extra_train_ids, scores the held-out)│
                     ┌─────────────┐   │                                                 │
                     │scam_rules.py│◀──┼───────── imported by all three scripts          │
                     │ scam_flags  │   │                                                 │
                     │ any_flag    │   │                                                 │
                     └─────────────┘   │                                                 │
                            ▲          │                                                 │
fake_job_postings.csv ──┬───┼──────────┼─────────────────────────────────────────────────┤
 (EMSCAD, 17,880 rows)  │   │          │                                                 │
                        ▼   │          │                                                 ▼
               ┌────────────┴──────┐   │          ┌──────────────────────────────────────────┐
               │ model-testing1.py │───┘          │ model-embeddings.py                      │
               │ MAIN model        │              │ tfidf vs embed vs hybrid,                │
               │ TF-IDF + LR       │              │ per-region thresholds, detect_region()   │
               └────────┬──────────┘              └───────────────┬──────────────────────────┘
                        ▼                                         ▼
              job_fraud_results/                       job_fraud_results_embed/
               job_fraud_baseline.joblib  (bundle)      job_fraud_embed.joblib
               evaluation.json, metrics.csv             evaluation.json, comparison.csv
               *.png graphs, test_errors.csv            embedding_comparison.png
                                                        embedding_cache.joblib (gitignored)
```

Where to look for what:

| You want to… | Go to |
|---|---|
| Change cleaning or text fields | `clean_text` / `combine_fields`, **duplicated** in `model-testing1.py` (nested in `main`), `model-embeddings.py` (top level) and `convert_hf_india.py` (`clean_text` + `score`). Change all three or they diverge. |
| Change how alerts are decided | `alert()` in `model-testing1.py`; `evaluate()` + `score_posting()` in `model-embeddings.py`; `score()` in `convert_hf_india.py` |
| Add or adjust a scam red flag | `RULES` in `scam_rules.py`, then re-measure (see Invariants) |
| Change threshold selection | `model-testing1.py`: `TARGET_PRECISION` block (OOF over US trainval). `model-embeddings.py`: `us_threshold()` (precision target) and `india_threshold()` (max OOF accuracy) |
| Tune hyperparameters | `model-testing1.py`: `--c-grid` CV loop. `model-embeddings.py`: `CONFIGS` dict (C fixed per feature set) |
| Indian vs US routing | `detect_region()` + `INDIA_PATTERN` in `model-embeddings.py` |
| What a saved model contains | `bundle = {...}` in `model-testing1.py`; `joblib.dump({...})` in `model-embeddings.py` |
| Old, superseded code | `Job_Fraud_TFIDF_Colab.py`: the original baseline. Don't edit; kept for reference only. |
| Old result folders | `job_fraud_results_baseline/` (original C=1 model), `job_fraud_results_v2/` (quick run). Read-only history. |

## Data pipeline (both model scripts must match exactly)

1. EMSCAD → `combine_fields` over `title, company_profile, description, requirements, benefits`
   → drop empty → drop texts with conflicting labels → drop duplicate texts.
2. Group key = hash of company profile (or of the ad text if there is no profile). Splits are by
   group, so **no company appears in two splits**.
3. `GroupShuffleSplit(test_size=0.2, seed 42)` → trainval / **test**;
   `GroupShuffleSplit(test_size=0.25, seed 43)` on trainval → train / val.
4. India CSV: drop texts also in EMSCAD, dedupe, `train_test_split(test_size=0.2, seed 42,
   stratify)` → 477 train / **120 held-out**.
5. Final training rows = train + description-only copies of train (`desc_only`, so the model
   learns short single-paragraph ads) + India train rows with sample weight 5.
6. Final alert = `score >= threshold` **OR** any `scam_rules` hit.

## Invariants: don't break these

- **Never choose a model, threshold or hyperparameter by looking at the test sets** (US test or
  India held-out). Use out-of-fold CV on training data. The test sets have already been looked at
  many times, so any new claim needs fresh data (ideally real Indian postings).
- **Keep splits identical across scripts** (same cleaning, seeds and filters) so numbers are
  comparable. If you change the pipeline, change it everywhere and say that the old numbers no
  longer compare.
- **New models go in a new script with a new results folder.** Never overwrite
  `job_fraud_results/` or another model's outputs to show a new idea; the user wants old model
  performance preserved.
- **Saved bundles record `extra_train_ids`.** Anything that scores the India data must skip
  those IDs (see `convert_hf_india.py score()`), or the result is inflated by training rows.
- **Scam rules must stay near-zero on legitimate EMSCAD postings.** When written they hit 22 of
  17,014 legitimate postings (0.13%) and 147/373 Indian fakes. Re-measure both after any edit
  (the docstring in `scam_rules.py` records the numbers). Watch for salary lines ("Pay ₹20,000
  per week") and product mentions ("Twitter, WhatsApp") being misread as scam signals.
- **Unpaid internships are not fraud.** The synthetic India data labels many as fake; no rule
  was added for them on purpose.
- Model scores are **uncalibrated**: don't present them as probabilities.

## Environment and commands

- Python **3.12** venv at `trial-modeling-news/.venv` (gitignored). Rebuild:
  `python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt`
  (+ `requirements-embeddings.txt` for `model-embeddings.py`: torch, sentence-transformers).
- Always call `.venv/bin/python` explicitly. The user's shell has Conda `(base)` active, so a
  bare `python` may be the wrong interpreter.
- Run from `trial-modeling-news/`:
  - `.venv/bin/python model-testing1.py --quick --no-show` (~2 min; full run ~5 min)
  - `.venv/bin/python convert_hf_india.py --score` (needs internet)
  - `.venv/bin/python model-embeddings.py --no-show` (~10 min first run, ~3 min with cache;
    run in the background)
- All 6 "AUTOMATED CHECKS" printed by `model-testing1.py` must PASS.

## Gotchas already hit

- **The venv broke when it was moved:** `activate` hardcodes its absolute path. If the folder
  moves, rebuild the venv; don't patch it.
- **Python.org's macOS Python lacks root certificates** → `CERTIFICATE_VERIFY_FAILED`.
  `convert_hf_india.py get_json()` falls back to certifi, then to `curl`.
- **Virtual environments were once committed** (≈3M lines; still in git history, about 177 MB of
  the clone). `.gitignore` now blocks `.venv/`, `.venv-*/`, `__pycache__/` and
  `embedding_cache.joblib`. Check `git status` before committing; never commit environments or
  caches.
- **Embeddings alone are poor on US postings** (recall ~6%): MiniLM truncates at 256 tokens and
  EMSCAD postings are long. They help only alongside TF-IDF.
- **The India data is AI-generated** (`Aioshi/smolified-fakejob`). High accuracy on it partly
  reflects learning the generator's style.

## Git

- Remote: `github.com/sm3625/hack-humanity`, branch `main`. The user pushes; commit only when
  asked, and don't push or force-push without explicit permission.

## Next steps (in priority order)

1. **Real Indian evaluation data.** Collect 100–200 real labeled Indian postings
   (cybercrime.gov.in reports, Reddit r/developersIndia and r/india scam posts, Naukri /
   Internshala ads for the legitimate class) and save them in the same columns as
   `fake_job_postings.csv`. Score all models on it without retraining. This is the only
   trustworthy accuracy number and decides whether the hybrid gets promoted.
2. **Promote the hybrid if (1) supports it.** Set `selected = 'hybrid'` in `model-embeddings.py`
   (currently chosen by mean OOF AP), or better, add a selection rule that uses the real data.
3. **Fix long-text truncation for embeddings.** Split postings into chunks of at most 256
   tokens and average their embeddings, or embed title + description only, then re-run the
   comparison.
4. **Refactor shared code.** Move `clean_text`, `combine_fields`, the split logic and
   `detect_region` into a shared module (e.g. `pipeline.py`) so the three scripts can't drift.
   After the refactor, check that all numbers in the results table above stay the same.
5. **Calibrate scores** (e.g. `CalibratedClassifierCV`) if they will be shown to end users.
6. **More South/Southeast Asian scam patterns** (Bangladesh, Pakistan, Indonesia, Philippines):
   no public labeled data was found (searched Nov 2026). Building a collection pipeline is the
   path forward.
