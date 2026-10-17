"""Ridge regression of guided-minus-actor success from cached two-view embeddings."""
import argparse
import json
import math
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedGroupKFold

from train_small_gain_router import digest, evaluate, select_threshold, read_jsonl

CONTRACT = 'question-context_and_first-answer_v1_utf8chunks2800'
AUX_KEYS = {'domain=code', 'domain=qa', 'log_question_length', 'log_answer_length',
            'confidence_mean_logprob', 'confidence_p10_logprob', 'confidence_min_logprob',
            'confidence_available', 'confidence_token_count', 'online_output_nonempty', 'online_truncated'}


def auxiliary(row):
    values = row.get('auxiliary', {})
    result = {}
    for key in AUX_KEYS:
        if key in values:
            value = values[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Nonfinite auxiliary feature')
            result[key] = float(value)
    return result


def validate(rows, question, answer, ids, *, labeled):
    if not rows or question.ndim != 2 or answer.shape != question.shape or len(rows) != len(question):
        raise ValueError('Embedding shape mismatch')
    if question.shape[1] != 1024:
        raise ValueError('Expected embedding-3 1024 dimensions per view')
    if list(ids) != [r['sample_id'] for r in rows] or len(set(ids)) != len(ids):
        raise ValueError('Embedding identity mismatch or duplicate IDs')
    if not np.isfinite(question).all() or not np.isfinite(answer).all():
        raise ValueError('Nonfinite embeddings')
    for matrix in (question, answer):
        if not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4):
            raise ValueError('Expected L2 normalized embeddings')
    for r in rows:
        if r['domain'] not in ('code', 'qa'):
            raise ValueError('Unsupported domain')
        auxiliary(r)
        if labeled:
            if r['split'] not in ('train', 'val') or r['a'] not in (0, 1) or r['b'] not in (0, 1):
                raise ValueError('Invalid split or outcomes')
            if r['delta'] != r['b'] - r['a'] or r['label'] != int(r['delta'] > 0):
                raise ValueError('Target inconsistent with counterfactual outcomes')
    if labeled:
        for split in ('train', 'val'):
            if not any(r['split'] == split for r in rows):
                raise ValueError('Missing split')
        for key in ('sample_id', 'group_id', 'prompt_sha256'):
            a = {r[key] for r in rows if r['split'] == 'train'}
            b = {r[key] for r in rows if r['split'] == 'val'}
            if a & b:
                raise ValueError('Train/val overlap: ' + key)
        if len({r['group_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate groups')


def load_vectors(folder, run=None):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    for name, key in [('vectors.npz', 'vectors_sha256'), ('metadata.json', 'metadata_sha256')]:
        if digest(folder / name) != manifest[key]:
            raise ValueError('Input checksum mismatch: ' + name)
    meta = json.loads((folder / 'metadata.json').read_text())
    if meta['contract'] != CONTRACT or meta['model'] != 'embedding-3':
        raise ValueError('Embedding contract/model mismatch')
    with np.load(folder / 'vectors.npz', allow_pickle=False) as data:
        question, answer, ids = data['question'], data['answer'], data['sample_ids']
    rows = meta['rows']
    validate(rows, question, answer, ids, labeled=run is not None)
    if run is not None:
        required = {'labels-train/router_dataset.jsonl', 'labels-train/labels.jsonl',
                    'labels-val/router_dataset.jsonl', 'labels-val/labels.jsonl'}
        if set(meta['input_sha256']) != required:
            raise ValueError('Missing source checksums')
        for rel, h in meta['input_sha256'].items():
            if rel not in {'labels-train/router_dataset.jsonl', 'labels-train/labels.jsonl',
                           'labels-val/router_dataset.jsonl', 'labels-val/labels.jsonl'}:
                raise ValueError('Unexpected source path')
            if digest(Path(run) / rel) != h:
                raise ValueError('Server source differs from embedding source: ' + rel)
        original = {}
        for split in ('train', 'val'):
            for record in read_jsonl(Path(run) / ('labels-' + split) / 'labels.jsonl'):
                if record['label'] is not None and record['domain'] in ('code', 'qa'):
                    original[record['sample_id']] = record
        if set(original) != {r['sample_id'] for r in rows}:
            raise ValueError('Cached sample set differs from source labels')
        for row in rows:
            record = original[row['sample_id']]
            if any(record[k] != row[k] for k in ('group_id', 'domain', 'split', 'label')):
                raise ValueError('Cached label identity mismatch')
            a = int(record['round_scores']['actor_only'][-1]['metrics']['success'])
            b = int(record['round_scores']['guided'][-1]['metrics']['success'])
            if row['a'] != a or row['b'] != b or row['delta'] != record['delta_success']:
                raise ValueError('Cached outcomes differ from original labels')
    return rows, np.hstack([question, answer]).astype(np.float64), meta, manifest


def matrix(embeddings, rows, dv, *, fit):
    # Offline targets/outcomes never enter this feature matrix.
    aux = [auxiliary(r) for r in rows]
    values = dv.fit_transform(aux) if fit else dv.transform(aux)
    return np.hstack([embeddings, values])


def summary(rows, outcomes, predictions, threshold, penalty):
    metrics = evaluate(rows, outcomes, predictions, threshold, penalty)
    target = np.array([r['delta'] for r in rows])
    metrics['delta_mse'] = float(mean_squared_error(target, predictions))
    return metrics


def train(args):
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError('Output exists; choose a new directory')
    rows, embeddings, meta, manifest = load_vectors(args.vectors, args.run_dir)
    train_i = np.array([i for i, r in enumerate(rows) if r['split'] == 'train'])
    val_i = np.array([i for i, r in enumerate(rows) if r['split'] == 'val'])
    tr, va = [rows[i] for i in train_i], [rows[i] for i in val_i]
    target = np.array([r['delta'] for r in tr], dtype=np.float64)
    strata = np.array([r['domain'] + ':' + str(r['label']) for r in tr])
    if min(np.sum(strata == k) for k in set(strata)) < 3:
        raise ValueError('Too few samples per domain/class for 3-fold CV')
    groups = np.array([r['group_id'] for r in tr])
    candidates = {alpha: np.empty(len(tr)) for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0)}
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=args.seed)
    for fold, (ti, vi) in enumerate(cv.split(np.zeros(len(tr)), strata, groups)):
        dv = DictVectorizer(sparse=False)
        xtr = matrix(embeddings[train_i[ti]], [tr[i] for i in ti], dv, fit=True)
        xva = matrix(embeddings[train_i[vi]], [tr[i] for i in vi], dv, fit=False)
        for alpha, scores in candidates.items():
            model = Ridge(alpha=alpha, solver='lsqr', tol=1e-8, max_iter=3000)
            model.fit(xtr, target[ti])
            scores[vi] = np.clip(model.predict(xva), -1, 1)
        print('RIDGE_CV_FOLD', fold + 1, flush=True)
    selection = [{'alpha': alpha, 'oof_delta_mse': float(mean_squared_error(target, p))}
                 for alpha, p in candidates.items()]
    best = min(selection, key=lambda v: v['oof_delta_mse'])
    oof = candidates[best['alpha']]
    threshold = select_threshold(oof, target, args.call_penalty)
    dv = DictVectorizer(sparse=False)
    xtr = matrix(embeddings[train_i], tr, dv, fit=True)
    xva = matrix(embeddings[val_i], va, dv, fit=False)
    model = Ridge(alpha=best['alpha'], solver='lsqr', tol=1e-8, max_iter=3000).fit(xtr, target)
    predictions = np.clip(model.predict(xva), -1, 1)
    outcomes = {r['sample_id']: {k: r[k] for k in ('a', 'b', 'extra_tokens')} for r in rows}
    output.mkdir(parents=True)
    artifact = {'format': 'embedding3_gain_ridge_v1', 'model': model, 'metadata_vectorizer': dv,
                'threshold': threshold, 'call_penalty': args.call_penalty, 'embedding_contract': CONTRACT,
                'embedding_model': 'embedding-3', 'dimensions_per_view': 1024,
                'domains': ['code', 'qa'], 'target': 'guided_success_minus_actor_success'}
    joblib.dump(artifact, output / 'router.joblib', compress=3)
    report = {'model': 'embedding-3 + Ridge', 'target': artifact['target'], 'seed': args.seed,
              'train_n': len(tr), 'val_n': len(va), 'head_parameters': int(model.coef_.size + 1),
              'train_delta_counts': {str(d): int(np.sum(target == d)) for d in (-1, 0, 1)},
              'selected_alpha': best['alpha'], 'candidates': selection,
              'threshold': threshold, 'call_penalty_success_units': args.call_penalty,
              'validation': summary(va, outcomes, predictions, threshold, args.call_penalty),
              'validation_fixed_cost_threshold': summary(va, outcomes, predictions, args.call_penalty, args.call_penalty),
              'training_oof_selection_diagnostic': summary(tr, outcomes, oof, threshold, args.call_penalty),
              'input_manifest': manifest, 'source_sha256': meta['input_sha256'],
              'embedding_contract': {k: meta[k] for k in ('contract', 'model', 'dimensions_per_view', 'chunk_bytes', 'aggregation')},
              'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__},
              'command': sys.argv,
              'notes': ['Alpha chosen by training OOF MSE; threshold by training OOF penalized net gain only.',
                        'Same OOF is used for selection; diagnostics are optimistic, not independent test metrics.',
                        'Validation fixed-cost threshold 0.02 is an additional prespecified diagnostic, not selected on validation.',
                        'Scores are uncalibrated delta estimates clipped to [-1,1], not probabilities.',
                        'Only Code/QA; original Math and excluded-review pools are omitted.',
                        'Only 8 validation positives; questions were used in expert DPO.',
                        'API chunked full-context input differs from the MiniLM 256-token baseline; comparison does not isolate encoder quality.']}
    for name, data, scores in [('train_oof_predictions.jsonl', tr, oof), ('val_predictions.jsonl', va, predictions)]:
        with (output / name).open('w') as f:
            for row, score in zip(data, scores):
                value = {k: row[k] for k in ('sample_id', 'group_id', 'domain', 'split', 'label', 'delta', 'a', 'b', 'extra_tokens')}
                value.update(predicted_delta=float(score), call_guidance=bool(score >= threshold))
                f.write(json.dumps(value) + '\n')
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    # Read source checksums again after training.
    load_vectors(args.vectors, args.run_dir)
    print(json.dumps({k: report[k] for k in ('train_n', 'val_n', 'train_delta_counts', 'head_parameters', 'selected_alpha', 'threshold', 'validation')}, indent=2), flush=True)


def predict(args):
    artifact = joblib.load(args.model)
    if artifact['format'] != 'embedding3_gain_ridge_v1':
        raise ValueError('Unexpected model format')
    rows, embeddings, _, _ = load_vectors(args.vectors)
    scores = np.clip(artifact['model'].predict(matrix(embeddings, rows, artifact['metadata_vectorizer'], fit=False)), -1, 1)
    with Path(args.output).open('x') as f:
        for row, score in zip(rows, scores):
            f.write(json.dumps({'sample_id': row['sample_id'], 'predicted_delta': float(score),
                                'call_guidance': bool(score >= artifact['threshold'])}) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train'); t.add_argument('--vectors', required=True); t.add_argument('--run-dir', required=True)
    t.add_argument('--output-dir', required=True); t.add_argument('--seed', type=int, default=42)
    t.add_argument('--call-penalty', type=float, default=0.02)
    p = sub.add_parser('predict'); p.add_argument('--vectors', required=True); p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'train':
        if not math.isfinite(args.call_penalty) or args.call_penalty < 0:
            parser.error('call penalty must be finite and nonnegative')
        train(args)
    else:
        predict(args)


if __name__ == '__main__':
    main()
