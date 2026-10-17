"""Small binary gain router. Offline branch outcomes are evaluation-only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import joblib
import numpy as np
from scipy import sparse
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

TEXT_KEYS = ('problem', 'context', 'initial_answer', 'initial_reasoning')
CONF_KEYS = ('mean_logprob', 'p10_logprob', 'min_logprob', 'available', 'token_count')


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def visible(row):
    f = row['features']
    values = {k: f.get(k, '') for k in TEXT_KEYS}
    if any(not isinstance(v, str) for v in values.values()):
        raise ValueError('Text features must be strings')
    if not values['problem'].strip():
        raise ValueError('Missing problem')
    text = '\n'.join(k + ': ' + values[k][:12000] for k in TEXT_KEYS)
    aux = {'domain=' + row['domain']: 1.0}
    for k, v in values.items():
        aux['log_length_' + k] = math.log1p(len(v)) / 10
    for k in CONF_KEYS:
        v = (f.get('confidence') or {}).get(k)
        if isinstance(v, (int, float)) and math.isfinite(v):
            aux['confidence_' + k] = (math.log1p(max(v, 0)) / 10 if k == 'token_count'
                                      else float(np.clip(v, -30, 30)) / 10)
    for k in ('output_nonempty', 'truncated'):
        v = (f.get('online_feedback') or {}).get(k)
        if isinstance(v, bool):
            aux['online_' + k] = float(v)
    return text, aux


def validate_splits(train, val):
    for name, rows in [('train', train), ('val', val)]:
        if not rows:
            raise ValueError('Empty ' + name)
        for key in ('sample_id', 'group_id'):
            if any(not r.get(key) for r in rows) or len({r[key] for r in rows}) != len(rows):
                raise ValueError('Missing or duplicate ' + key + ' in ' + name)
        if any(r.get('split') != name or type(r.get('label')) is not int or r['label'] not in (0, 1) for r in rows):
            raise ValueError('Invalid labels or split tags in ' + name)
    for key in ('sample_id', 'group_id'):
        if {r[key] for r in train} & {r[key] for r in val}:
            raise ValueError('Train/val overlap: ' + key)
    def prompt_key(r):
        f = r['features']
        return ' '.join((f['problem'] + '\n' + f.get('context', '')).split()).casefold()
    if {prompt_key(r) for r in train} & {prompt_key(r) for r in val}:
        raise ValueError('Train/val duplicate prompts')
    if set(r['label'] for r in train) != {0, 1}:
        raise ValueError('Training requires both classes')


def load_data(run, domains):
    sets, outcomes = [], {}
    for split in ('train', 'val'):
        folder = run / ('labels-' + split)
        rows = [r for r in read_jsonl(folder / 'router_dataset.jsonl') if r['domain'] in domains]
        wanted = {r['sample_id']: r for r in rows}
        for r in read_jsonl(folder / 'labels.jsonl'):
            if r['sample_id'] not in wanted:
                continue
            row = wanted[r['sample_id']]
            if any(r[k] != row[k] for k in ('group_id', 'domain', 'split', 'label')):
                raise ValueError('Label identity mismatch')
            a = int(r['round_scores']['actor_only'][-1]['metrics']['success'])
            b = int(r['round_scores']['guided'][-1]['metrics']['success'])
            if r['exclusion_reasons'] or r['delta_success'] != b - a or row['label'] != int(b > a):
                raise ValueError('Inconsistent or excluded label')
            outcomes[r['sample_id']] = {'a': a, 'b': b,
                'extra_tokens': r.get('extra_guided_cost', {}).get('total_tokens')}
        if any(r['sample_id'] not in outcomes for r in rows):
            raise ValueError('Missing evaluation outcomes')
        sets.append(rows)
    validate_splits(*sets)
    return *sets, outcomes


def fit_features(rows, backend, embeddings=None):
    texts, aux = zip(*(visible(r) for r in rows))
    dv = DictVectorizer()
    meta = dv.fit_transform(aux)
    vectorizer = None
    if backend == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=16000, ngram_range=(1, 2),
                                    min_df=2, sublinear_tf=True, dtype=np.float64)
        x = vectorizer.fit_transform(texts)
    else:
        x = sparse.csr_matrix(embeddings)
    return sparse.hstack([x, meta], format='csr'), vectorizer, dv


def transform(rows, vectorizer, dv, embeddings=None):
    texts, aux = zip(*(visible(r) for r in rows))
    x = vectorizer.transform(texts) if vectorizer is not None else sparse.csr_matrix(embeddings)
    return sparse.hstack([x, dv.transform(aux)], format='csr')


def encode(rows, model_path, device='cpu', batch_size=32):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    encoder = AutoModel.from_pretrained(model_path, local_files_only=True, trust_remote_code=False).to(device)
    encoder.eval()
    arrays = []
    # Separate question/context and first answer so the answer survives token truncation.
    for keys in [('problem', 'context'), ('initial_answer', 'initial_reasoning')]:
        texts = ['\n'.join(r['features'].get(k, '')[:12000] for k in keys) for r in rows]
        batches = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                tokens = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                                   max_length=256, return_tensors='pt').to(device)
                hidden = encoder(**tokens).last_hidden_state
                mask = tokens['attention_mask'].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                batches.append(torch.nn.functional.normalize(pooled, dim=1).cpu().numpy())
        arrays.append(np.concatenate(batches))
    del encoder
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return np.concatenate(arrays, axis=1)


def select_threshold(probabilities, delta, penalty):
    candidates = np.r_[np.unique(probabilities), np.nextafter(float(max(probabilities)), np.inf)]
    threshold = max(candidates, key=lambda t: (
        float(np.sum((delta - penalty) * (probabilities >= t))),
        -int(np.sum(probabilities >= t))))
    return float(threshold)


def policy_metrics(rows, outcomes, selected, penalty):
    a = np.array([outcomes[r['sample_id']]['a'] for r in rows])
    b = np.array([outcomes[r['sample_id']]['b'] for r in rows])
    delta = b - a
    known_costs = [outcomes[r['sample_id']]['extra_tokens'] for r, s in zip(rows, selected) if s]
    return {'n': len(rows), 'calls': int(np.sum(selected)), 'call_rate': float(np.mean(selected)),
            'improvements': int(np.sum((delta > 0) & selected)),
            'regressions': int(np.sum((delta < 0) & selected)),
            'net_success_gain': int(np.sum(delta * selected)),
            'task_success_rate': float(np.mean(np.where(selected, b, a))),
            'penalized_gain': float(np.sum((delta - penalty) * selected)),
            'extra_tokens': sum(known_costs) if all(c is not None for c in known_costs) else None}


def evaluate(rows, outcomes, probabilities, threshold, penalty):
    y = np.array([r['label'] for r in rows]); selected = probabilities >= threshold
    result = {'positive': int(y.sum()), 'positive_rate': float(y.mean()),
              'average_precision': float(average_precision_score(y, probabilities)) if y.any() else None,
              'roc_auc': float(roc_auc_score(y, probabilities)) if len(set(y)) == 2 else None,
              'precision': float(precision_score(y, selected, zero_division=0)),
              'recall': float(recall_score(y, selected, zero_division=0)),
              'f1': float(f1_score(y, selected, zero_division=0)),
              'chosen_policy': policy_metrics(rows, outcomes, selected, penalty),
              'never': policy_metrics(rows, outcomes, np.zeros(len(rows), dtype=bool), penalty),
              'always': policy_metrics(rows, outcomes, np.ones(len(rows), dtype=bool), penalty)}
    delta = np.array([outcomes[r['sample_id']]['b'] - outcomes[r['sample_id']]['a'] for r in rows])
    curve = []
    for rate in (0.01, 0.05, 0.1, 0.2, 0.5, 1.0):
        k = min(len(rows), max(1, int(math.ceil(rate * len(rows)))))
        mask = np.zeros(len(rows), dtype=bool)
        mask[np.argsort(-probabilities, kind='stable')[:k]] = True
        item = policy_metrics(rows, outcomes, mask, penalty)
        item['target_rate'] = rate
        item['random_expected_net_gain_at_same_calls'] = float(k * delta.mean())
        curve.append(item)
    result['budget_curve_exploratory'] = curve
    return result


def train(args):
    run = Path(args.run_dir).resolve(); out = Path(args.output_dir)
    if out.exists():
        raise ValueError('Output already exists; choose a new output directory')
    if 'math' in args.domains:
        raise ValueError('Math labels require a repaired scorer before training')
    rows, val, outcomes = load_data(run, args.domains)
    hashes = {str(p): digest(p) for s in ('train', 'val') for p in
              [run / ('labels-' + s) / 'router_dataset.jsonl', run / ('labels-' + s) / 'labels.jsonl']}
    y = np.array([r['label'] for r in rows]); groups = np.array([r['group_id'] for r in rows])
    strata = np.array([r['domain'] + ':' + str(r['label']) for r in rows])
    if min(np.sum(strata == k) for k in set(strata)) < 3:
        raise ValueError('At least three samples per domain/class required for grouped CV')
    embeddings = val_embeddings = None
    if args.backend == 'embedding':
        if not args.encoder_path:
            raise ValueError('--encoder-path required')
        all_embeddings = encode(rows + val, args.encoder_path, args.device, args.batch_size)
        embeddings, val_embeddings = all_embeddings[:len(rows)], all_embeddings[len(rows):]
    candidates = {(c, w): np.empty(len(rows)) for c in (0.1, 1.0, 10.0) for w in (None, 'balanced')}
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=args.seed)
    for fold, (ti, vi) in enumerate(cv.split(np.zeros(len(rows)), strata, groups)):
        tr, va = [rows[i] for i in ti], [rows[i] for i in vi]
        xtr, vec, dv = fit_features(tr, args.backend, None if embeddings is None else embeddings[ti])
        xva = transform(va, vec, dv, None if embeddings is None else embeddings[vi])
        for (c, weight), predictions in candidates.items():
            model = LogisticRegression(C=c, class_weight=weight, solver='liblinear', max_iter=2000, random_state=args.seed)
            model.fit(xtr, y[ti]); predictions[vi] = model.predict_proba(xva)[:, 1]
        print('CV fold complete', fold + 1, flush=True)
    scores = [{'C': c, 'class_weight': w, 'oof_average_precision': float(average_precision_score(y, p))}
              for (c, w), p in candidates.items()]
    best = max(scores, key=lambda s: s['oof_average_precision'])
    oof = candidates[(best['C'], best['class_weight'])]
    delta = np.array([outcomes[r['sample_id']]['b'] - outcomes[r['sample_id']]['a'] for r in rows])
    threshold = select_threshold(oof, delta, args.call_penalty)
    x, vectorizer, dv = fit_features(rows, args.backend, embeddings)
    model = LogisticRegression(C=best['C'], class_weight=best['class_weight'], solver='liblinear',
                               max_iter=2000, random_state=args.seed).fit(x, y)
    val_prob = model.predict_proba(transform(val, vectorizer, dv, val_embeddings))[:, 1]
    out.mkdir(parents=True)
    artifact = {'format': 'small_gain_router_v1', 'backend': args.backend, 'model': model,
                'vectorizer': vectorizer, 'metadata_vectorizer': dv, 'threshold': threshold,
                'encoder_path': str(Path(args.encoder_path).resolve()) if args.encoder_path else None,
                'domains': args.domains, 'feature_contract': 'problem/context/initial_answer/initial_reasoning + domain + initial confidence/flags',
                'seed': args.seed}
    joblib.dump(artifact, out / 'router.joblib', compress=3)
    report = {'train_n': len(rows), 'train_positive': int(y.sum()), 'val_n': len(val),
              'backend': args.backend, 'head_parameters': int(model.coef_.size + model.intercept_.size),
              'threshold': threshold, 'selection': best, 'candidates': scores,
              'call_penalty_success_units': args.call_penalty,
              'validation': evaluate(val, outcomes, val_prob, threshold, args.call_penalty),
              'training_oof_selection_diagnostic': evaluate(rows, outcomes, oof, threshold, args.call_penalty),
              'notes': ['Hyperparameters and threshold selected using training OOF only.',
                        'OOF selection metrics are optimistic selection diagnostics, not an unbiased test.',
                        'Validation has only a few positives; budget curves are exploratory.',
                        'Math excluded due to known label bias; review pools not merged.',
                        'These questions were used in expert DPO; unseen-question generalization is unproven.',
                        'Decision is before the five-round branch; not a per-round policy.'],
              'input_sha256': hashes,
              'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__},
              'command': sys.argv}
    for name, data, probabilities in [('val_predictions.jsonl', val, val_prob), ('train_oof_predictions.jsonl', rows, oof)]:
        with (out / name).open('w') as f:
            for row, p in zip(data, probabilities):
                record = {k: row[k] for k in ('sample_id', 'group_id', 'domain', 'split', 'label')}
                record.update(score=float(p), call_guidance=bool(p >= threshold), **outcomes[row['sample_id']])
                f.write(json.dumps(record) + '\n')
    for path, expected in hashes.items():
        if digest(path) != expected:
            raise ValueError('Input changed during training: ' + path)
    (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: report[k] for k in ('train_n', 'train_positive', 'val_n', 'backend', 'head_parameters', 'threshold', 'selection', 'validation')}, indent=2), flush=True)


def predict(args):
    artifact = joblib.load(args.model)
    rows = list(read_jsonl(args.input))
    if not rows:
        raise ValueError('Empty prediction input')
    outside = {r['domain'] for r in rows} - set(artifact['domains'])
    if outside:
        raise ValueError('Unsupported domains: ' + str(outside))
    embeddings = None
    if artifact['backend'] == 'embedding':
        embeddings = encode(rows, artifact['encoder_path'], args.device, args.batch_size)
    matrix = transform(rows, artifact['vectorizer'], artifact['metadata_vectorizer'], embeddings)
    probabilities = artifact['model'].predict_proba(matrix)[:, 1]
    with Path(args.output).open('x') as f:
        for row, p in zip(rows, probabilities):
            f.write(json.dumps({'sample_id': row.get('sample_id'), 'score': float(p),
                                'call_guidance': bool(p >= artifact['threshold'])}) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('--run-dir', required=True); t.add_argument('--output-dir', required=True)
    t.add_argument('--domains', nargs='+', choices=['code', 'qa', 'math'], default=['code', 'qa'])
    t.add_argument('--backend', choices=['tfidf', 'embedding'], default='tfidf')
    t.add_argument('--encoder-path'); t.add_argument('--seed', type=int, default=42)
    t.add_argument('--call-penalty', type=float, default=0.02,
                   help='Cost per guided branch, measured in task success units; fixed before evaluation')
    t.add_argument('--device', default='cpu'); t.add_argument('--batch-size', type=int, default=32)
    p = sub.add_parser('predict')
    p.add_argument('--model', required=True); p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu'); p.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.command == 'train':
        if not math.isfinite(args.call_penalty) or args.call_penalty < 0:
            parser.error('--call-penalty must be finite and nonnegative')
        train(args)
    else:
        predict(args)


if __name__ == '__main__':
    main()
