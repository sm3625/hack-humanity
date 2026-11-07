# hack-humanity — job recruitment fraud detection

Flags likely-fraudulent job postings, with extra coverage for Indian/South Asian scam patterns.
All code lives in [`trial-modeling-news/`](trial-modeling-news/).

## Setup (Python 3.12)

```bash
cd trial-modeling-news
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Only needed for model-embeddings.py (adds PyTorch, ~1 GB):
pip install -r requirements-embeddings.txt
```

Run every command below from `trial-modeling-news/` with the environment active.

## Files

| File | What it is |
|---|---|
| `model-testing1.py` | **Main model.** TF-IDF + logistic regression, trained on the US/global dataset plus 80% of the Indian data. Writes `job_fraud_results/`. |
| `model-embeddings.py` | Comparison model: TF-IDF vs. sentence embeddings (all-MiniLM-L6-v2) vs. both, with separate thresholds for Indian postings. Writes `job_fraud_results_embed/`. |
| `scam_rules.py` | Hand-written India-specific red flags (upfront fees, WhatsApp/Telegram contact, Aadhaar/PAN/bank requests, guaranteed jobs). Any hit forces an alert. |
| `convert_hf_india.py` | Downloads the Indian dataset from Hugging Face, converts it to `india_synthetic_jobs.csv`, and with `--score` scores the held-out part with the saved model. |
| `fake_job_postings.csv` | EMSCAD: 17,880 real (mostly US/UK, 2012–14) postings, ~5% fraudulent. |
| `india_synthetic_jobs.csv` | 597 **AI-generated** Indian-context postings (373 fake) from `Aioshi/smolified-fakejob`. |
| `job_fraud_results/` | Current main model (`job_fraud_baseline.joblib`), metrics and graphs. |
| `job_fraud_results_embed/` | Embedding comparison results and saved model. |
| `job_fraud_results_baseline/`, `job_fraud_results_v2/` | Older runs, kept for comparison. |

## Common commands

```bash
python model-testing1.py                       # retrain main model (~5 min), opens graphs
python model-testing1.py --quick --no-show     # faster, no graph windows
python convert_hf_india.py --score             # score Indian held-out postings with saved model
python model-embeddings.py                     # embedding comparison (~10 min first run)
```

Useful flags for `model-testing1.py`: `--target-precision 0.7` (fewer false alarms, less fraud
caught), `--no-rules`, `--no-extra`, `--no-desc-aug`, `--extra-weight N`.

## Using the saved model

```python
import joblib
bundle = joblib.load('job_fraud_results/job_fraud_baseline.joblib')
score = bundle['pipeline'].predict_proba(['lowercased title + company profile + description ...'])[0, 1]
flag = score >= bundle['threshold']   # also flag if scam_rules.scam_flags(text) is non-empty
```

Text must be cleaned the same way as in training (`clean_text` / `combine_fields` in `model-testing1.py`).

## Current results

| Held-out set | Accuracy | Fraud caught | False alarms |
|---|---|---|---|
| US/global test (3,213 postings, 203 fraud) | 96.7% | 55% | 15 |
| Indian held-out (120 postings, 75 fake) — main model | 90.8% | 99% | 10 |
| Indian held-out — TF-IDF, India-specific threshold (`model-embeddings.py`) | 92.5% | 97% | 7 |
| Indian held-out — TF-IDF + embeddings, India threshold | 94.2% | 95% | 3 |

## Caveats

- The Indian data is AI-generated; real-world accuracy on Indian scams is unknown until tested on
  real labeled postings.
- The Indian held-out set is only 120 postings (±~5 points).
- Scores are uncalibrated and are not fraud probabilities. A posting below the threshold is not
  guaranteed safe.

## Next steps

1. Collect 100–200 real Indian postings with labels (cybercrime.gov.in reports, Reddit, Naukri)
   and evaluate on them.
2. If the TF-IDF + embeddings model also wins on real data, make it the main model.
