#!/usr/bin/env python3
"""Run the prepared full train/val paired pool; preview unless --execute is given."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/root/autodl-tmp/lrr/router-paired-full-20261004'))
    parser.add_argument('--execute', action='store_true', help='Collect train, then val, then score both')
    parser.add_argument('--min-free-gb', type=float, default=2.0)
    args = parser.parse_args()
    if args.min_free_gb < 1:
        parser.error('--min-free-gb must be at least 1')
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from gain_router.paired import collect, load_settings
    from gain_router.inputs import read_objects
    from gain_router.traces import write_json_atomic

    output = args.output.resolve()
    tasks = output / 'pool/tasks.jsonl'
    supervision = output / 'pool/supervision.jsonl'
    rows = list(read_objects(tasks))
    if not supervision.is_file():
        raise ValueError('Missing offline supervision')
    settings_paths = [output / f'settings-{split}.json' for split in ('train', 'val')]
    for split, path in zip(('train', 'val'), settings_paths):
        settings, actor, preferences = load_settings(path)
        counts = {d: sum(t['split'] == split and t['domain'] == d for t in rows)
                  for d in ('code', 'math', 'qa')}
        if settings['rounds'] != 5 or settings['split'] != split:
            raise ValueError(f'Expected five rounds and split={split}: {path}')
        if settings['max_per_domain'] < max(counts.values()):
            raise ValueError(f'Configuration would truncate the full pool: {path}')
        for domain, model in preferences.items():
            weights = Path(model.weights_path)
            adapter = Path(model.adapter_path) / 'adapter_model.safetensors'
            if not weights.is_dir():
                raise ValueError(f'Missing SFT weights: {weights}')
            with adapter.open('rb') as handle:
                digest = hashlib.file_digest(handle, 'sha256').hexdigest()
            if model.revision != 'sha256:' + digest:
                raise ValueError(f'Adapter hash mismatch: {domain}')
        preview = collect(tasks, path, output / f'run-{split}', local_preference=True)
        print(split, counts, preview, flush=True)

    endpoint = actor.endpoint.rstrip('/')
    health = endpoint.removesuffix('/v1') + '/health'
    with urllib.request.urlopen(health, timeout=10) as response:
        if response.status != 200:
            raise ValueError('Actor health check failed')
    with urllib.request.urlopen(endpoint + '/models', timeout=10) as response:
        models = json.load(response)['data']
    if not any(m['id'] == actor.model_id and m.get('max_model_len', 0) >= 16384 for m in models):
        raise ValueError('Actor model or context limit differs from verified service')
    print('Free disk GB:', round(shutil.disk_usage(output).free / 2**30, 2), flush=True)
    if not args.execute:
        print('Preview passed. Add --execute to start; no generation was performed.')
        return

    # The outer lock covers both splits and prevents two launchers racing for the GPU.
    with (output / '.launcher.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        prefix = [sys.executable, '-u', '-m', 'gain_router.paired']

        def run_phase(phase, command):
            status_path = output / 'launcher-status.json'
            def status(state, **extra):
                write_json_atomic(status_path, {'phase': phase, 'state': state,
                    'updated_at_unix': time.time(), **extra})
            if shutil.disk_usage(output).free < args.min_free_gb * 2**30:
                status('disk_low')
                raise RuntimeError('Disk reserve reached; free space and rerun the same command')
            with (output / f'{phase}.log').open('a') as log:
                child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
                status('running', pid=child.pid, command=command)
                print('Started', phase, 'PID', child.pid, flush=True)
                try:
                    while True:
                        try:
                            code = child.wait(timeout=30)
                            break
                        except subprocess.TimeoutExpired:
                            if shutil.disk_usage(output).free < args.min_free_gb * 2**30:
                                status('disk_low', pid=child.pid)
                                raise RuntimeError('Disk reserve reached; preserve files and add storage before resuming')
                except BaseException:
                    child.send_signal(signal.SIGINT)
                    try:
                        child.wait(timeout=60)
                    except subprocess.TimeoutExpired:
                        child.terminate()
                        child.wait()
                    raise
            status('complete' if code == 0 else 'failed', exit_code=code)
            if code:
                raise RuntimeError(f'{phase} exited {code}; inspect {output / (phase + ".log")}')

        for split, path in zip(('train', 'val'), settings_paths):
            run_phase('collect-' + split, prefix + ['collect', '--tasks', str(tasks),
                '--settings', str(path), '--output', str(output / f'run-{split}'),
                '--local-preference', '--execute'])
        for split in ('train', 'val'):
            labels = output / f'labels-{split}'
            if (labels / 'report.json').is_file():
                print('Existing label report:', labels, flush=True)
                continue
            if labels.exists():
                raise RuntimeError(f'Incomplete labels at {labels}; preserve them before rescoring')
            run_phase('label-' + split, prefix + ['label', '--run', str(output / f'run-{split}'),
                '--supervision', str(supervision), '--output', str(labels), '--allow-code-execution'])
        write_json_atomic(output / 'launcher-status.json', {'state': 'complete', 'updated_at_unix': time.time()})
        print('Full collection and offline scoring complete.', flush=True)


if __name__ == '__main__':
    main()
