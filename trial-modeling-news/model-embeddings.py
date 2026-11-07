#!/usr/bin/env python3
"""Sentence-embedding fraud model with separate alert thresholds for US/global and Indian postings.

This is separate from model-testing1.py (the TF-IDF model), which is left unchanged along with
its results in job_fraud_results/. This script writes to job_fraud_results_embed/.

It compares three feature sets on exactly the same data splits as model-testing1.py:
  tfidf   - the existing word/bigram TF-IDF features (re-trained here for a fair comparison)
  embed   - all-MiniLM-L6-v2 sentence embeddings (understands meaning, not just exact words)
  hybrid  - both feature sets side by side
Each gets two thresholds: one for US/global postings and one for Indian postings, both chosen
from out-of-fold cross-validation scores (never the test sets). Postings are routed to a
threshold by detect_region() (location field or India-specific words such as ₹, lakh, Naukri).

Install: pip install -r requirements.txt sentence-transformers
Run:     python model-embeddings.py            (first run downloads the ~90 MB embedding model)
"""
import argparse
import hashlib
import html
import json
import re
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
SEED = 42
TEXT_COLUMNS = ['title', 'company_profile', 'description', 'requirements', 'benefits']
EMBED_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
INDIA_PATTERN = re.compile(
    r'₹|\brs\.?\s?\d|\binr\b|rupee|\blakhs?\b|\blpa\b|aadh?aa?r|naukri|internshala|\bindia\b'
    r'|bangalore|bengaluru|mumbai|delhi|pune|hyderabad|chennai|kolkata|noida|gurgaon|gurugram'
    r'|ahmedabad|jaipur|lucknow|kochi|indore|chandigarh', re.IGNORECASE)


def clean_text(value):
    import pandas as pd
    if pd.isna(value):
        return ''
    value = html.unescape(str(value))
    value = re.sub(r'<[^>]*>', ' ', value)
    return re.sub(r'\s+', ' ', value).strip().lower()


def combine_fields(frame, columns=TEXT_COLUMNS):
    import pandas as pd
    parts = [(frame[c] if c in frame else pd.Series('', index=frame.index)).map(clean_text)
             for c in columns]
    combined = parts[0]
    for part in parts[1:]:
        combined = combined + ' ' + part
    return combined.str.replace(r'\s+', ' ', regex=True).str.strip()


def detect_region(location, text):
    """'IN' for postings that look Indian, else 'US' (the default/global threshold)."""
    if str(location or '').strip().upper().startswith('IN'):
        return 'IN'
    return 'IN' if INDIA_PATTERN.search(text or '') else 'US'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', type=Path, default=HERE / 'fake_job_postings.csv')
    parser.add_argument('--extra-csv', type=Path, default=HERE / 'india_synthetic_jobs.csv')
    parser.add_argument('--extra-weight', type=float, default=5.0)
    parser.add_argument('--output', type=Path, default=HERE / 'job_fraud_results_embed')
    parser.add_argument('--target-precision', type=float, default=0.60,
                        help='US/global threshold: highest recall with at least this precision.')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import joblib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold,
                                         StratifiedKFold, train_test_split)
    from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                                 precision_recall_curve, precision_score, recall_score)
    from scam_rules import any_flag, scam_flags

    # ---------- Data: identical cleaning and splits to model-testing1.py ----------
    df = pd.read_csv(args.csv)
    df['fraudulent'] = df['fraudulent'].astype(int)
    df['text'] = combine_fields(df)
    df = df.loc[df['text'].str.len() > 0]
    conflicts = df.groupby('text')['fraudulent'].nunique()
    df = df.loc[~df['text'].isin(conflicts[conflicts > 1].index)]
    df = df.drop_duplicates('text').reset_index(drop=True)
    profiles = df['company_profile'].map(clean_text)
    keys = np.where(profiles.str.len() > 0, 'company:' + profiles, 'ad:' + df['text'])
    df['group'] = [hashlib.sha256(k.encode()).hexdigest() for k in keys]
    trainval_idx, test_idx = next(GroupShuffleSplit(1, test_size=0.20, random_state=SEED)
                                  .split(df, groups=df['group']))
    trainval, test = df.iloc[trainval_idx].copy(), df.iloc[test_idx].copy()
    train_idx, _ = next(GroupShuffleSplit(1, test_size=0.25, random_state=SEED + 1)
                        .split(trainval, groups=trainval['group']))
    train = trainval.iloc[train_idx].copy()

    extra = pd.read_csv(args.extra_csv)
    extra['fraudulent'] = pd.to_numeric(extra['fraudulent'], errors='coerce')
    extra = extra.dropna(subset=['fraudulent']).astype({'fraudulent': int})
    extra['text'] = combine_fields(extra)
    extra = extra.loc[(extra['text'].str.len() > 0) & ~extra['text'].isin(df['text'])]
    extra = extra.drop_duplicates('text').reset_index(drop=True)
    in_train, in_test = train_test_split(extra, test_size=0.20, random_state=SEED,
                                         stratify=extra['fraudulent'])
    print(f'US: trainval {len(trainval)}, train {len(train)}, test {len(test)} | '
          f'India: train {len(in_train)}, held-out {len(in_test)}')

    def desc_only(frame):
        text = combine_fields(frame, ['description'])
        return text[text.str.len() > 0], frame.loc[text.str.len() > 0, 'fraudulent']

    # ---------- Sentence embeddings (cached on disk) ----------
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(EMBED_MODEL)
    encoder.max_seq_length = 256
    cache_path = args.output / 'embedding_cache.joblib'
    try:
        cache = joblib.load(cache_path)
    except Exception:
        cache = {}

    def embed(texts):
        texts = list(texts)
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        missing = list(dict.fromkeys(t for t, k in zip(texts, keys) if k not in cache))
        if missing:
            print(f'Embedding {len(missing):,} new texts with {EMBED_MODEL} ...', flush=True)
            vectors = encoder.encode(missing, batch_size=64, normalize_embeddings=True,
                                     show_progress_bar=False, convert_to_numpy=True)
            for t, v in zip(missing, vectors):
                cache[hashlib.sha1(t.encode()).hexdigest()] = v.astype(np.float32)
            joblib.dump(cache, cache_path)
        return np.vstack([cache[k] for k in keys])

    # ---------- Feature sets ----------
    def tfidf_vec():
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98,
                               max_features=50000, sublinear_tf=True)

    CONFIGS = {'tfidf': {'C': 100.0}, 'embed': {'C': 10.0}, 'hybrid': {'C': 30.0}}

    def fit_model(kind, texts, labels, weights):
        vec = tfidf_vec().fit(texts) if kind in ('tfidf', 'hybrid') else None
        X = features(kind, vec, texts)
        clf = LogisticRegression(class_weight='balanced', C=CONFIGS[kind]['C'],
                                 solver='liblinear', max_iter=5000, random_state=SEED)
        clf.fit(X, labels, sample_weight=weights)
        return {'kind': kind, 'vectorizer': vec, 'classifier': clf}

    def features(kind, vec, texts):
        if kind == 'tfidf':
            return vec.transform(texts)
        if kind == 'embed':
            return embed(texts)
        return hstack([vec.transform(texts), csr_matrix(embed(texts))]).tocsr()

    def predict(fitted, texts):
        X = features(fitted['kind'], fitted['vectorizer'], texts)
        return fitted['classifier'].predict_proba(X)[:, 1]

    def training_set(us_rows, india_rows):
        d_text, d_y = desc_only(us_rows)
        texts = pd.concat([us_rows['text'], d_text, india_rows['text']], ignore_index=True)
        labels = pd.concat([us_rows['fraudulent'], d_y, india_rows['fraudulent']], ignore_index=True)
        weights = np.concatenate([np.ones(len(us_rows) + len(d_text)),
                                  np.full(len(india_rows), args.extra_weight)])
        return texts, labels, weights

    # Embed everything once up front (one progress message instead of many).
    embed(pd.concat([df['text'], desc_only(df)[0], extra['text']]))

    # ---------- Cross-validation: out-of-fold scores for both regions ----------
    us_folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
                    .split(trainval, trainval['fraudulent'], trainval['group']))
    in_folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED)
                    .split(in_train, in_train['fraudulent']))
    oof = {k: {'US': np.zeros(len(trainval)), 'IN': np.zeros(len(in_train))} for k in CONFIGS}
    for fold, ((us_tr, us_ts), (in_tr, in_ts)) in enumerate(zip(us_folds, in_folds), 1):
        texts, labels, weights = training_set(trainval.iloc[us_tr], in_train.iloc[in_tr])
        for kind in CONFIGS:
            fitted = fit_model(kind, texts, labels, weights)
            oof[kind]['US'][us_ts] = predict(fitted, trainval['text'].iloc[us_ts])
            oof[kind]['IN'][in_ts] = predict(fitted, in_train['text'].iloc[in_ts])
        print(f'CV fold {fold}/5 done', flush=True)

    def us_threshold(y, s):
        p, r, t = precision_recall_curve(y, s)
        p, r = p[:-1], r[:-1]
        ok = np.flatnonzero(p >= args.target_precision)
        return float(t[ok[np.argmax(r[ok])]] if len(ok) else t[np.argmax(2 * p * r / np.maximum(p + r, 1e-12))])

    def india_threshold(y, s):
        candidates = np.unique(s)
        acc = [accuracy_score(y, s >= c) for c in candidates]
        return float(candidates[int(np.argmax(acc))])     # best out-of-fold accuracy

    thresholds, cv_rows = {}, []
    for kind in CONFIGS:
        thresholds[kind] = {'US': us_threshold(trainval['fraudulent'], oof[kind]['US']),
                            'IN': india_threshold(in_train['fraudulent'], oof[kind]['IN'])}
        cv_rows.append({'model': kind,
                        'US OOF avg precision': average_precision_score(trainval['fraudulent'], oof[kind]['US']),
                        'India OOF avg precision': average_precision_score(in_train['fraudulent'], oof[kind]['IN']),
                        'US threshold': thresholds[kind]['US'], 'India threshold': thresholds[kind]['IN']})
    cv_table = pd.DataFrame(cv_rows).set_index('model')
    cv_table['mean OOF avg precision'] = cv_table[['US OOF avg precision', 'India OOF avg precision']].mean(axis=1)
    selected = cv_table['mean OOF avg precision'].idxmax()
    print('\nCROSS-VALIDATION (train data only)')
    print(cv_table.round(4).to_string())
    print(f'Selected by CV: {selected}')

    # ---------- Final models and held-out evaluation ----------
    texts, labels, weights = training_set(train, in_train)
    final = {kind: fit_model(kind, texts, labels, weights) for kind in CONFIGS}

    test_region = np.array([detect_region(l, t) for l, t in zip(test['location'], test['text'])])
    in_region = np.array([detect_region(l, t) for l, t in zip(in_test['location'], in_test['text'])])
    print(f'\nRegion detection: {np.mean(test_region == "IN"):.1%} of US test postings routed to the '
          f'India threshold; {np.mean(in_region == "IN"):.1%} of Indian held-out postings detected as India')
    rules_test, rules_in = np.array(any_flag(test['text'])), np.array(any_flag(in_test['text']))

    def evaluate(y, scores, regions, rules, th):
        cut = np.where(regions == 'IN', th['IN'], th['US'])
        pred = np.maximum((scores >= cut).astype(int), rules)
        return {'accuracy': accuracy_score(y, pred), 'precision': precision_score(y, pred, zero_division=0),
                'recall': recall_score(y, pred, zero_division=0), 'f1': f1_score(y, pred, zero_division=0),
                'false_alarms': int(((pred == 1) & (y == 0)).sum()), 'pred': pred}

    rows, held_scores = [], {}
    for kind in CONFIGS:
        s_test, s_in = predict(final[kind], test['text']), predict(final[kind], in_test['text'])
        held_scores[kind] = s_in
        for th_name, th in [('shared', {'US': thresholds[kind]['US'], 'IN': thresholds[kind]['US']}),
                            ('per-region', thresholds[kind])]:
            us = evaluate(test['fraudulent'].to_numpy(), s_test, test_region, rules_test, th)
            ind = evaluate(in_test['fraudulent'].to_numpy(), s_in, in_region, rules_in, th)
            rows.append({'model': kind, 'thresholds': th_name,
                         'US acc': us['accuracy'], 'US precision': us['precision'], 'US recall': us['recall'],
                         'US false alarms': us['false_alarms'],
                         'India acc': ind['accuracy'], 'India precision': ind['precision'],
                         'India recall': ind['recall'], 'India false alarms': ind['false_alarms']})
    results = pd.DataFrame(rows)
    print(f'\nHELD-OUT RESULTS (US test {len(test)} rows / {int(test.fraudulent.sum())} fraud; '
          f'India held-out {len(in_test)} rows / {int(in_test.fraudulent.sum())} fake; rules ON)')
    print(results.round(3).to_string(index=False))
    results.to_csv(args.output / 'comparison.csv', index=False)

    # ---------- Graph ----------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    labels_ = [f"{r.model}\n{r.thresholds}" for r in results.itertuples()]
    x = np.arange(len(results))
    for ax, metric, title in [(axes[0], 'US recall', 'US test: fraud caught (recall)'),
                              (axes[1], 'India acc', 'India held-out: accuracy')]:
        bars = ax.bar(x, results[metric], color=['#245c9c' if t == 'shared' else '#d57622'
                                                 for t in results['thresholds']])
        ax.bar_label(bars, fmt='%.2f', fontsize=8)
        ax.set_xticks(x, labels_, fontsize=8)
        ax.set(ylim=(0, 1.05), title=title)
    for kind, color in zip(CONFIGS, ['#245c9c', '#3a8a4f', '#d57622']):
        p, r, _ = precision_recall_curve(in_test['fraudulent'], held_scores[kind])
        axes[2].plot(r, p, label=f'{kind} (AP {average_precision_score(in_test["fraudulent"], held_scores[kind]):.3f})',
                     color=color, linewidth=2)
    axes[2].set(xlabel='Recall', ylabel='Precision', title='India held-out precision–recall', ylim=(0, 1.03))
    axes[2].legend()
    fig.suptitle('TF-IDF vs sentence embeddings, shared vs per-region thresholds (blue = shared, orange = per-region)')
    plt.tight_layout()
    graph = args.output / 'embedding_comparison.png'
    fig.savefig(graph, dpi=160, bbox_inches='tight')
    plt.close(fig)

    # ---------- Save the selected model ----------
    joblib.dump({'model': final[selected], 'kind': selected, 'thresholds': thresholds[selected],
                 'embedding_model': EMBED_MODEL, 'text_columns': TEXT_COLUMNS,
                 'region_rule': 'detect_region() in model-embeddings.py',
                 'rules': 'Alert also if scam_rules.scam_flags(text) is non-empty',
                 'extra_train_ids': in_train['job_id'].astype(str).tolist()},
                args.output / 'job_fraud_embed.joblib')
    (args.output / 'evaluation.json').write_text(json.dumps({
        'selected_model': selected, 'thresholds': thresholds,
        'cv': cv_table.reset_index().to_dict(orient='records'),
        'held_out': results.to_dict(orient='records')}, indent=2, default=float))

    def score_posting(posting):
        text = combine_fields(pd.DataFrame([posting])).iloc[0]
        if not text:
            return {'status': 'Insufficient information'}
        region = detect_region(posting.get('location'), text)
        score = float(predict(final[selected], [text])[0])
        flags = scam_flags(text)
        cut = thresholds[selected][region]
        return {'status': 'Review recommended' if score >= cut or flags else 'Below review threshold',
                'model_score': round(score, 4), 'region': region, 'threshold': round(cut, 4),
                'scam_flags': flags}

    print('\nExample:', json.dumps(score_posting({
        'title': 'Data entry operator', 'location': '',
        'description': 'Work from home, earn 25,000 rupees monthly. Pay 999 registration fee to start.'})))
    print(f'\nSaved to {args.output.resolve()}  (graph: {graph.name}, table: comparison.csv)')
    if not args.no_show:
        import subprocess, sys
        if sys.platform == 'darwin':
            subprocess.run(['open', str(graph)], check=False)


if __name__ == '__main__':
    main()
