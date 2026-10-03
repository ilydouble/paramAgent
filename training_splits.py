"""Frozen train/validation membership shared by Qwen SFT, DPO and collection."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from split_protocol import content_group_id, file_sha256, load_manifest


def add_split_arguments(parser):
    parser.add_argument('--split_manifest', required=True,
                        help='Frozen problem-level manifest; never randomly repartition examples.')
    parser.add_argument('--exclude_unmatched', action='store_true',
                        help='Explicitly exclude examples outside the manifest and audit their count.')


def example_identity(row, text):
    # Code prompts are cleaned later; fingerprint the original manifest input.
    gid = content_group_id(text)
    if row.get('group_id') not in (None, gid):
        raise ValueError('Declared group_id disagrees with original input text')
    return {'group_id': gid, **{key: row[key] for key in ('split', 'split_seed') if key in row}}


def partition_examples(examples, manifest, *, exclude_unmatched=False):
    lookup = {row['group_id']: row['split'] for row in manifest['assignments']}
    partitions = {'train': [], 'val': []}
    counts = Counter()
    for row in examples:
        split = lookup.get(row['group_id'])
        if split is None:
            if not exclude_unmatched:
                raise ValueError('Example has no manifest match; explicitly exclude unmatched sources first')
            counts['unmatched_excluded'] += 1
            continue
        if row.get('split') not in (None, split) or row.get('split_seed') not in (None, manifest['seed']):
            raise ValueError('Example split metadata disagrees with manifest')
        if split == 'test':
            counts['test_excluded'] += 1
            continue
        partitions[split].append(row)
        counts[split] += 1
    if not partitions['train'] or not partitions['val']:
        raise ValueError('Both frozen train and val must contain examples; no random fallback')
    return partitions, dict(counts)


def validate_training_artifact(path, manifest, *, stage=None, domain=None):
    root = Path(path)
    report_path = root / 'training_split.json'
    if not report_path.is_file():
        raise ValueError(f'Missing frozen training provenance: {report_path}; historical models require retraining')
    report = json.loads(report_path.read_text())
    if (report.get('status') != 'complete' or report.get('assignment_sha256') != manifest['assignment_sha256']
            or report.get('split_seed') != manifest['seed']
            or (stage is not None and report.get('stage') != stage)
            or (domain is not None and report.get('domain') != domain)):
        raise ValueError('Model training provenance does not match the requested frozen split/stage/domain')
    lookup = {row['group_id']: row['split'] for row in manifest['assignments']}
    groups = report.get('group_ids', {})
    if set(groups) != {'train', 'val'} or any(lookup.get(gid) != split
            for split, ids in groups.items() for gid in ids):
        raise ValueError('Model provenance includes held-out or unmatched training groups')
    artifacts = report.get('artifacts', {})
    if not artifacts:
        raise ValueError('Model provenance has no verified artifact hashes')
    for name, digest in artifacts.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid artifact path')
        if file_sha256(root / relative) != digest:
            raise ValueError(f'Model artifact checksum mismatch: {name}')
    return report


def frozen_datasets(examples, args, *, stage, domain):
    # Run this before any CUDA allocation so invalid data/base models fail early.
    manifest = load_manifest(args.split_manifest)
    base_report = None
    if stage == 'dpo':
        base_report = validate_training_artifact(args.base_model, manifest, stage='sft', domain=domain)
    partitions, counts = partition_examples(examples, manifest, exclude_unmatched=args.exclude_unmatched)
    output = Path(args.output_dir)
    report = {'status': 'prepared', 'stage': stage, 'domain': domain,
              'split_seed': manifest['seed'], 'model_seed': args.seed,
              'assignment_sha256': manifest['assignment_sha256'],
              'manifest_sha256': file_sha256(args.split_manifest),
              'dataset_sha256': file_sha256(args.dataset_path), 'base_model': str(args.base_model), 'counts': counts,
              'group_ids': {key: sorted({row['group_id'] for row in values}) for key, values in partitions.items()}}
    if base_report is not None:
        report['base_artifacts'] = base_report['artifacts']
    provenance = output / 'training_split.json'
    if args.__dict__.get('resume_from_checkpoint'):
        checkpoint = Path(args.resume_from_checkpoint).resolve()
        if checkpoint.parent != output.resolve() or not checkpoint.name.startswith('checkpoint-'):
            raise ValueError('Resume checkpoint must belong to this audited output directory')
        previous = json.loads(provenance.read_text())
        if any(previous.get(key) != report[key] for key in report if key != 'status'):
            raise ValueError('Resume provenance differs from current frozen training inputs')
    elif output.exists() and any(output.iterdir()):
        raise ValueError('Use a fresh output directory for each manifest/stage run')
    output.mkdir(parents=True, exist_ok=True)
    provenance.write_text(json.dumps(report, indent=2))
    print('Frozen split audit:', counts, 'split seed:', manifest['seed'], 'model seed:', args.seed)
    from datasets import Dataset
    return {key: Dataset.from_list(rows).shuffle(seed=args.seed) for key, rows in partitions.items()}


def record_training_completion(args):
    root = Path(args.output_dir)
    report = json.loads((root / 'training_split.json').read_text())
    report['artifacts'] = {name: file_sha256(root / name)
                           for name in ('adapter_model.safetensors', 'adapter_config.json')}
    report['status'] = 'complete'
    (root / 'training_split.json').write_text(json.dumps(report, indent=2))
