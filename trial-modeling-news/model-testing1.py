#!/usr/bin/env python3
"""Standalone TF-IDF + logistic regression training and testing.

Install: python -m pip install numpy pandas scikit-learn matplotlib joblib
Run: python Job_Fraud_TFIDF_Colab.py --csv fake_job_postings.csv

Outputs go to ./job_fraud_results (override with --output).
No Colab, IPython, or pre-existing variables are needed.
Scope: general recruitment fraud, not verified troll-farm recruitment.
Learning curves use training sample size, not epochs; loss is unweighted log loss.
The model's scores are uncalibrated. A below-threshold score is not a safety guarantee.
"""
import argparse
from pathlib import Path

def display(value):
    print(value.to_string() if hasattr(value, 'to_string') else value)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path,
                        default=Path(__file__).resolve().parent / 'fake_job_postings.csv')
    parser.add_argument('--output', type=Path, default=Path('job_fraud_results'))
    parser.add_argument('--quick', action='store_true',
                        help='Skip subset refits; learning graphs contain only the full-data point.')
    args = parser.parse_args()
    if not args.csv.is_file():
        parser.error(f'Dataset not found: {args.csv}. Supply --csv /path/to/fake_job_postings.csv')
    args.output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import re, html, json, hashlib

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import sklearn, joblib
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import (classification_report, confusion_matrix,
        ConfusionMatrixDisplay, precision_recall_curve, average_precision_score,
        PrecisionRecallDisplay, precision_score, recall_score, f1_score)
    SEED = 42
    TEXT_COLUMNS = ['title', 'company_profile', 'description', 'requirements', 'benefits']
    print('scikit-learn:', sklearn.__version__)



    CSV_PATH = args.csv
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {CSV_PATH}. Use --csv /path/to/fake_job_postings.csv")
    raw = pd.read_csv(CSV_PATH)
    print('Rows loaded:', len(raw))



    def clean_text(value):
        if pd.isna(value):
            return ''
        value = html.unescape(str(value))
        value = re.sub(r'<[^>]*>', ' ', value)
        return re.sub(r'\s+', ' ', value).strip().lower()

    def combine_fields(frame):
        parts = []
        for column in TEXT_COLUMNS:
            values = frame[column] if column in frame else pd.Series('', index=frame.index)
            parts.append(values.map(clean_text))
        combined = parts[0]
        for part in parts[1:]:
            combined = combined + ' ' + part
        return combined.str.replace(r'\s+', ' ', regex=True).str.strip()

    if 'fraudulent' not in raw:
        raise ValueError('Missing target column: fraudulent')
    if 'description' not in raw:
        raise ValueError('Missing text column: description')
    labels = pd.to_numeric(raw['fraudulent'], errors='coerce')
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError('Every fraudulent label must be 0 or 1.')
    df = raw.copy()
    df['fraudulent'] = labels.astype(int)
    df['text'] = combine_fields(df)
    df = df.loc[df['text'].str.len() > 0].copy()
    conflicting = df.groupby('text')['fraudulent'].nunique()
    conflicting = conflicting[conflicting > 1].index
    n_conflicts = df['text'].isin(conflicting).sum()
    df = df.loc[~df['text'].isin(conflicting)].copy()
    n_before = len(df)
    df = df.drop_duplicates('text').reset_index(drop=True)
    print('Excluded conflicting-label rows:', n_conflicts)
    print('Removed duplicate rows:', n_before - len(df))

    profiles = (df['company_profile'] if 'company_profile' in df
                else pd.Series('', index=df.index)).map(clean_text)
    group_keys = np.where(profiles.str.len() > 0, 'company:' + profiles, 'ad:' + df['text'])
    df['group'] = [hashlib.sha256(key.encode()).hexdigest() for key in group_keys]
    print(df['fraudulent'].value_counts().rename(index={0:'legitimate', 1:'fraudulent'}))



    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    trainval_idx, test_idx = next(outer.split(df, groups=df['group']))
    trainval = df.iloc[trainval_idx].copy()
    test = df.iloc[test_idx].copy()
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED + 1)
    train_idx, val_idx = next(inner.split(trainval, groups=trainval['group']))
    train = trainval.iloc[train_idx].copy()
    val = trainval.iloc[val_idx].copy()
    for name, split in [('train', train), ('validation', val), ('test', test)]:
        if split['fraudulent'].nunique() != 2:
            raise ValueError(f'{name} lacks one class. More independent labeled groups are needed.')
        print(f'{name:12s}: {len(split):5d} rows, fraud fraction {split.fraudulent.mean():.3f}')
    assert set(train.group).isdisjoint(val.group)
    assert set(train.group).isdisjoint(test.group)
    assert set(val.group).isdisjoint(test.group)



    model = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2,
             max_df=0.98, max_features=50000, sublinear_tf=True, dtype=np.float64)),
        ('classifier', LogisticRegression(class_weight='balanced', C=1.0,
             solver='liblinear', max_iter=2000, random_state=SEED))
    ])
    model.fit(train['text'], train['fraudulent'])
    print('Training complete. Vocabulary size:', len(model['tfidf'].vocabulary_))



    TARGET_PRECISION = 0.80
    val_scores = model.predict_proba(val['text'])[:, 1]
    p, r, thresholds = precision_recall_curve(val['fraudulent'], val_scores)
    p, r = p[:-1], r[:-1]
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    valid = np.flatnonzero(p >= TARGET_PRECISION)
    if len(valid):
        chosen = valid[np.argmax(r[valid])]
        selection_note = 'Highest validation recall meeting the precision target'
    else:
        chosen = int(np.argmax(f1))
        selection_note = 'Precision target unmet; fallback to best validation F1'
    threshold = float(thresholds[chosen])
    print(selection_note)
    print(f'Threshold: {threshold:.4f}; precision: {p[chosen]:.3f}; recall: {r[chosen]:.3f}')
    print('Validation alerts:', int((val_scores >= threshold).sum()))



    test_scores = model.predict_proba(test['text'])[:, 1]
    test_pred = (test_scores >= threshold).astype(int)
    print('Always-legitimate accuracy:', round(1 - test['fraudulent'].mean(), 3))
    print(classification_report(test['fraudulent'], test_pred, labels=[0, 1],
          target_names=['legitimate', 'fraudulent'], digits=3, zero_division=0))
    metrics = {
        'precision': float(precision_score(test['fraudulent'], test_pred, zero_division=0)),
        'recall': float(recall_score(test['fraudulent'], test_pred, zero_division=0)),
        'f1': float(f1_score(test['fraudulent'], test_pred, zero_division=0)),
        'average_precision': float(average_precision_score(test['fraudulent'], test_scores)),
        'test_rows': len(test), 'test_fraud_rows': int(test['fraudulent'].sum())
    }
    print('Average precision:', round(metrics['average_precision'], 3))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ConfusionMatrixDisplay(confusion_matrix(test['fraudulent'], test_pred, labels=[0, 1]),
        display_labels=['legitimate', 'fraudulent']).plot(ax=axes[0], colorbar=False)
    PrecisionRecallDisplay.from_predictions(test['fraudulent'], test_scores, ax=axes[1])
    axes[1].set_title('Held-out test precision–recall')
    plt.tight_layout()
    fig.savefig(args.output / 'test_evaluation.png', dpi=180, bbox_inches='tight')
    plt.close(fig)



    def score_posting(posting, fitted_model=model, alert_threshold=threshold):
        text = combine_fields(pd.DataFrame([posting])).iloc[0]
        vector = fitted_model['tfidf'].transform([text])
        if not text or vector.nnz == 0:
            return {'status': 'Insufficient information', 'model_score': None}
        score = float(fitted_model['classifier'].predict_proba(vector)[0, 1])
        return {
            'status': 'Review recommended' if score >= alert_threshold else 'Below review threshold',
            'model_score': round(score, 4),
            'threshold': round(float(alert_threshold), 4),
            'note': 'Uncalibrated model score; not a fraud probability or a safety guarantee.'
        }

    example = {
        'title': 'Remote customer service assistant',
        'company_profile': '',
        'description': 'Earn money processing customer payments from home. Contact our recruiter to apply.',
        'requirements': 'No experience required.',
        'benefits': 'Flexible hours.'
    }
    print(json.dumps(score_posting(example), indent=2))



    def contributing_terms(posting, top_n=10):
        text = combine_fields(pd.DataFrame([posting])).iloc[0]
        row = model['tfidf'].transform([text])
        names = model['tfidf'].get_feature_names_out()
        contribution = row.data * model['classifier'].coef_[0, row.indices]
        signals = pd.DataFrame({'term': names[row.indices], 'contribution': contribution})
        return signals.sort_values('contribution', ascending=False).head(top_n)

    display(contributing_terms(example))



    bundle = {
        'pipeline': model, 'threshold': threshold, 'text_columns': TEXT_COLUMNS,
        'sklearn_version': sklearn.__version__, 'seed': SEED,
        'scope': 'General recruitment-fraud benchmark; not a troll-farm detector',
        'preprocessing': 'Reuse clean_text and combine_fields from this notebook'
    }
    joblib.dump(bundle, args.output / 'job_fraud_baseline.joblib')
    (args.output / 'evaluation.json').write_text(json.dumps({
        'test_metrics': metrics, 'threshold': threshold,
        'threshold_selection': selection_note, 'sklearn_version': sklearn.__version__,
        'split_sizes': {'train': len(train), 'validation': len(val), 'test': len(test)}
    }, indent=2))
    restored = joblib.load(args.output / 'job_fraud_baseline.joblib')
    assert np.allclose(restored['pipeline'].predict_proba(test['text'].iloc[:5]),
                       model.predict_proba(test['text'].iloc[:5]))
    print('Saved model and evaluation report.')
    from sklearn.base import clone
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
    from sklearn.model_selection import StratifiedShuffleSplit

    summary_rows = []
    for split_name, part in [('Train', train), ('Validation', val), ('Test', test)]:
        scores = model.predict_proba(part['text'])[:, 1]
        predictions = (scores >= threshold).astype(int)
        summary_rows.append({
            'Split': split_name, 'Rows': len(part), 'Fraud examples': int(part.fraudulent.sum()),
            'Accuracy': accuracy_score(part.fraudulent, predictions),
            'Balanced accuracy': balanced_accuracy_score(part.fraudulent, predictions),
            'Fraud precision': precision_score(part.fraudulent, predictions, zero_division=0),
            'Fraud recall': recall_score(part.fraudulent, predictions, zero_division=0),
            'Fraud F1': f1_score(part.fraudulent, predictions, zero_division=0),
            'Log loss': log_loss(part.fraudulent, scores, labels=[0, 1])
        })
    summary = pd.DataFrame(summary_rows).set_index('Split')
    print(f'PERFORMANCE AT SELECTED ALERT THRESHOLD: {threshold:.4f}')
    display(summary.round(4))

    learning_rows = []
    for fraction in ([1.00] if args.quick else [0.20, 0.50, 1.00]):
        if fraction == 1.0:
            subset, candidate = train, model
        else:
            sampler = StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=SEED)
            subset_idx, _ = next(sampler.split(train['text'], train['fraudulent']))
            subset = train.iloc[subset_idx]
            candidate = clone(model)
            candidate.fit(subset['text'], subset['fraudulent'])
        for split_name, part in [('Train', subset), ('Validation', val)]:
            scores = candidate.predict_proba(part['text'])[:, 1]
            learning_rows.append({
                'Training examples': len(subset), 'Split': split_name,
                'Log loss': log_loss(part.fraudulent, scores, labels=[0, 1]),
                'Accuracy': accuracy_score(part.fraudulent, scores >= 0.5),
                'Balanced accuracy': balanced_accuracy_score(part.fraudulent, scores >= 0.5)
            })
        print(f'Learning-curve point finished: {len(subset):,} training examples')
    learning = pd.DataFrame(learning_rows)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    colors = {'Train': '#245c9c', 'Validation': '#d57622'}
    for axis, metric in zip(axes[0], ['Log loss', 'Accuracy', 'Balanced accuracy']):
        for split_name in ['Train', 'Validation']:
            part = learning[learning['Split'] == split_name]
            axis.plot(part['Training examples'], part[metric], marker='o', linewidth=2,
                      label=split_name, color=colors[split_name])
        axis.set_xlabel('Training examples')
        axis.set_ylabel(metric)
        axis.set_title(metric + (' (lower is better)' if metric == 'Log loss' else ' (threshold 0.5)'))
        if metric != 'Log loss': axis.set_ylim(0, 1.03)
        axis.legend()
    majority_label = int(train.fraudulent.mode().iloc[0])
    axes[0, 1].axhline((val.fraudulent == majority_label).mean(), linestyle=':',
                       color='gray', label='Validation majority baseline')
    axes[0, 1].legend(fontsize=8)
    axes[0, 2].axhline(0.5, linestyle=':', color='gray')

    fixed_test_scores = model.predict_proba(test['text'])[:, 1]
    fixed_test_pred = (fixed_test_scores >= threshold).astype(int)
    cm = confusion_matrix(test.fraudulent, fixed_test_pred, labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=['Legitimate', 'Fraud']).plot(
        ax=axes[1, 0], cmap='Blues', colorbar=False)
    axes[1, 0].set_title(f'Test confusion matrix (threshold {threshold:.3f})')
    axes[1, 0].grid(False)
    PrecisionRecallDisplay.from_predictions(test.fraudulent, fixed_test_scores, ax=axes[1, 1])
    axes[1, 1].axhline(test.fraudulent.mean(), color='gray', linestyle=':', label='Fraud prevalence')
    axes[1, 1].scatter([recall_score(test.fraudulent, fixed_test_pred)],
        [precision_score(test.fraudulent, fixed_test_pred, zero_division=0)],
        s=65, color='#d57622', label='Selected threshold', zorder=5)
    axes[1, 1].set_title('Test precision vs. recall')
    axes[1, 1].legend(fontsize=8)
    for label, name, color in [(0, 'Legitimate', '#245c9c'), (1, 'Fraud', '#d57622')]:
        axes[1, 2].hist(fixed_test_scores[test.fraudulent.to_numpy() == label],
            bins=np.linspace(0, 1, 21), density=True, alpha=0.55, label=name, color=color)
    axes[1, 2].axvline(threshold, color='black', linestyle='--', label='Alert threshold')
    axes[1, 2].set(xlabel='Uncalibrated model score', ylabel='Density', title='Test score distributions')
    axes[1, 2].legend(fontsize=8)
    fig.suptitle('Job recruitment fraud baseline | learning curves and held-out evaluation', fontsize=17)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.output / 'performance_dashboard.png', dpi=180, bbox_inches='tight')
    plt.close(fig)

    checks = {
        'No shared employer-profile groups across splits': (
            set(train.group).isdisjoint(val.group) and set(train.group).isdisjoint(test.group)
            and set(val.group).isdisjoint(test.group)),
        'No exact duplicate text across splits': (
            set(train.text).isdisjoint(val.text) and set(train.text).isdisjoint(test.text)
            and set(val.text).isdisjoint(test.text)),
        'Both label classes present in every split': all(
            set(part.fraudulent.unique()) == {0, 1} for part in [train, val, test]),
        'Finite scores in [0, 1]': bool(np.isfinite(fixed_test_scores).all()
            and ((fixed_test_scores >= 0) & (fixed_test_scores <= 1)).all()),
        'Blank input returns insufficient information': score_posting({})['status'] == 'Insufficient information',
        'Save/reload preserves scores': bool(np.allclose(
            restored['pipeline'].predict_proba(test['text'].iloc[:10]),
            model.predict_proba(test['text'].iloc[:10])))
    }
    print('\nAUTOMATED CHECKS')
    for name, passed in checks.items(): print(('PASS' if passed else 'FAIL') + ': ' + name)
    assert all(checks.values()), 'A testing-suite check failed.'
    tn, fp, fn, tp = cm.ravel()
    print(f'\nTEST RESULTS: {tp} fraud postings caught; {fn} fraud postings missed; '
          f'{fp} legitimate postings flagged; {tn} legitimate postings correctly unflagged.')
    print('Low log loss is better. A large train/validation gap can indicate overfitting or distribution shift.')
    print('High accuracy can hide low fraud recall. The test set has already been inspected: '
          'future tuning needs a new final holdout for an unbiased final claim.')
    summary.to_csv(args.output / 'metrics.csv')
    learning.to_csv(args.output / 'learning_curves.csv', index=False)
    (args.output / 'checks.json').write_text(json.dumps(checks, indent=2))
    print(f"\nResults saved to: {args.output.resolve()}")


if __name__ == '__main__':
    main()
