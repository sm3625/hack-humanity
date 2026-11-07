#!/usr/bin/env python3
"""Convert the Aioshi/smolified-fakejob Hugging Face dataset into the
fake_job_postings.csv column format, and optionally score it with the saved model.

Source: https://huggingface.co/datasets/Aioshi/smolified-fakejob
  597 SYNTHETIC (LLM-generated) Indian-context job postings in chat format:
  user = "Job Posting: ...", assistant = "Classification: Fake/Real. Red Flags: ..."

Run:
  python convert_hf_india.py                 # download + convert -> india_synthetic_jobs.csv
  python convert_hf_india.py --score         # also score with job_fraud_results/job_fraud_baseline.joblib

Caveat: the data is generated, not scraped. Results on it show how the model handles
India-style scam wording, not real-world performance.
"""
import argparse
import html
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

DATASET = 'Aioshi/smolified-fakejob'
ROWS_API = ('https://datasets-server.huggingface.co/rows?dataset={ds}'
            '&config=default&split=train&offset={offset}&length={length}')
# Same columns, in the same order, as fake_job_postings.csv
COLUMNS = ['job_id', 'title', 'location', 'department', 'salary_range', 'company_profile',
           'description', 'requirements', 'benefits', 'telecommuting', 'has_company_logo',
           'has_questions', 'employment_type', 'required_experience', 'required_education',
           'industry', 'function', 'fraudulent']
TEXT_COLUMNS = ['title', 'company_profile', 'description', 'requirements', 'benefits']


def get_json(url):
    """Fetch JSON, working around python.org macOS builds that lack root certificates."""
    try:
        import ssl, certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = None
    try:
        with urllib.request.urlopen(url, timeout=60, context=context) as response:
            return json.load(response)
    except urllib.error.URLError as error:
        if 'CERTIFICATE_VERIFY_FAILED' not in str(error):
            raise
        import subprocess
        out = subprocess.run(['curl', '-sSfL', url], capture_output=True, text=True, check=True)
        return json.loads(out.stdout)


def fetch_rows():
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        url = ROWS_API.format(ds=DATASET, offset=offset, length=100)
        page = get_json(url)
        total = page['num_rows_total']
        rows.extend(item['row'] for item in page['rows'])
        offset += 100
        time.sleep(0.2)
    print(f'Downloaded {len(rows)} of {total} rows from {DATASET}')
    return rows


def parse_label(text):
    match = re.search(r'Classification:\s*(Fake|Real)', text or '', re.IGNORECASE)
    return None if match is None else int(match.group(1).lower() == 'fake')


def parse_red_flags(text):
    match = re.search(r'Red Flags:\s*(.*)', text or '', re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ''


def convert(rows):
    records = []
    for i, row in enumerate(rows):
        posting = re.sub(r'^\s*Job Posting:\s*', '', row.get('user') or '').strip()
        label = parse_label(row.get('assistant'))
        rec = {col: '' for col in COLUMNS}
        rec.update(job_id=f'hf-{i}', description=posting, fraudulent=label,
                   location='IN', telecommuting=int(bool(re.search(
                       r'work from home|remote|wfh', posting, re.IGNORECASE))))
        rec['red_flags'] = parse_red_flags(row.get('assistant'))
        records.append(rec)
    df = pd.DataFrame(records, columns=COLUMNS + ['red_flags'])
    n = len(df)
    df = df[df['description'].str.len() > 0]
    unlabeled = df['fraudulent'].isna().sum()
    df = df.dropna(subset=['fraudulent'])
    conflicts = df.groupby('description')['fraudulent'].nunique()
    df = df[~df['description'].isin(conflicts[conflicts > 1].index)]
    df = df.drop_duplicates('description').reset_index(drop=True)
    df['fraudulent'] = df['fraudulent'].astype(int)
    print(f'Kept {len(df)} of {n} rows (dropped {unlabeled} unlabeled, plus empty, '
          f'conflicting and duplicate postings)')
    print(df['fraudulent'].value_counts().rename(index={0: 'real', 1: 'fake'}).to_string())
    return df


def clean_text(value):
    if pd.isna(value):
        return ''
    value = html.unescape(str(value))
    value = re.sub(r'<[^>]*>', ' ', value)
    return re.sub(r'\s+', ' ', value).strip().lower()


def score(df, model_path, output_dir):
    import joblib
    from sklearn.metrics import (average_precision_score, classification_report,
                                 confusion_matrix)
    bundle = joblib.load(model_path)
    model, threshold = bundle['pipeline'], bundle['threshold']
    trained_on = set(bundle.get('extra_train_ids', []))
    if trained_on:
        before = len(df)
        df = df[~df['job_id'].astype(str).isin(trained_on)].reset_index(drop=True)
        print(f'\nSkipping {before - len(df)} postings the model was trained on; '
              f'scoring the {len(df)} held-out ones.')
    text = df[TEXT_COLUMNS].apply(lambda r: ' '.join(clean_text(v) for v in r), axis=1)
    text = text.str.replace(r'\s+', ' ', regex=True).str.strip()
    scores = model.predict_proba(text)[:, 1]
    model_pred = (scores >= threshold).astype(int)
    use_rules = bundle.get('use_rules', False)
    if use_rules:
        from scam_rules import scam_flags
        rule_flags = [', '.join(scam_flags(t)) for t in text]
        pred = (model_pred | pd.Series([int(bool(f)) for f in rule_flags]).to_numpy()).astype(int)
    else:
        rule_flags = [''] * len(text)
        pred = model_pred
    y = df['fraudulent']
    print(f'\nSCORING WITH {model_path} (threshold {threshold:.4f}, '
          f'red-flag rules {"ON" if use_rules else "OFF"})')
    if use_rules:
        tm, fm = ((model_pred == 1) & (y == 1)).sum(), ((model_pred == 1) & (y == 0)).sum()
        print(f'Model alone: {tm} fakes caught, {fm} real postings flagged')
    print(classification_report(y, pred, labels=[0, 1], target_names=['real', 'fake'],
                                digits=3, zero_division=0))
    if y.nunique() == 2:
        print('Average precision:', round(average_precision_score(y, scores), 3))
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    print(f'{tp} fakes caught, {fn} fakes missed, {fp} real postings flagged, {tn} real correctly passed')
    out = df.assign(model_score=scores.round(4), predicted=pred, rule_flags=rule_flags)
    out['result'] = [('caught' if p else 'MISSED fake') if t else ('FALSE ALARM' if p else 'ok')
                     for t, p in zip(y, pred)]
    path = output_dir / 'india_synthetic_scored.csv'
    out[['job_id', 'description', 'fraudulent', 'predicted', 'model_score', 'rule_flags',
         'result', 'red_flags']].to_csv(path, index=False)
    print('Per-posting scores:', path.resolve())


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=here / 'india_synthetic_jobs.csv')
    parser.add_argument('--score', action='store_true', help='Score with the saved model.')
    parser.add_argument('--model', type=Path,
                        default=here / 'job_fraud_results' / 'job_fraud_baseline.joblib')
    args = parser.parse_args()

    df = convert(fetch_rows())
    df.to_csv(args.output, index=False)
    print('Saved:', args.output.resolve())
    if args.score:
        if not args.model.is_file():
            parser.error(f'Model not found: {args.model}. Run model-testing1.py first.')
        score(df, args.model, args.output.parent)


if __name__ == '__main__':
    main()
