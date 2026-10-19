"""Build two-view API embeddings locally; credentials never enter exported data."""
import argparse
import concurrent.futures
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
import urllib.error
import urllib.request
import zipfile

import numpy as np

URL = 'https://open.bigmodel.cn/api/paas/v4/embeddings'
MODEL = 'embedding-3'
DIM = 1024
CHUNK_BYTES = 2800
CONTRACT = 'question-context_and_first-answer_v1_utf8chunks2800'


def sha(value):
    return hashlib.sha256(value).hexdigest()


def chunks(text):
    result, current, size = [], [], 0
    for char in text:
        n = len(char.encode('utf-8'))
        if current and size + n > CHUNK_BYTES:
            result.append(''.join(current)); current, size = [], 0
        current.append(char); size += n
    if current:
        result.append(''.join(current))
    return result or ['[empty first answer]']


def cache_key(text):
    return sha((URL + '\n' + MODEL + '\n' + str(DIM) + '\n' + text).encode())


def load_source(source):
    rows, texts, hashes = [], [], {}
    with zipfile.ZipFile(source) as z:
        expected = json.loads(z.read('router-dataset/SHA256.json'))
        for split in ('train', 'val'):
            def read(name):
                rel = 'labels-' + split + '/' + name
                raw = z.read('router-dataset/' + rel)
                assert sha(raw) == expected[rel], rel
                hashes[rel] = sha(raw)
                return [json.loads(line) for line in raw.decode().split('\n') if line.strip()]
            data = [r for r in read('router_dataset.jsonl') if r['domain'] in ('code', 'qa')]
            outcomes = {r['sample_id']: r for r in read('labels.jsonl') if r['label'] is not None}
            for r in data:
                detail, f = outcomes[r['sample_id']], r['features']
                assert all(detail[k] == r[k] for k in ('sample_id', 'group_id', 'domain', 'split', 'label'))
                a = int(detail['round_scores']['actor_only'][-1]['metrics']['success'])
                b = int(detail['round_scores']['guided'][-1]['metrics']['success'])
                assert not detail['exclusion_reasons']
                assert detail['delta_success'] == b-a and r['label'] == int(b > a)
                question = 'QUESTION:\n' + f['problem']
                if f.get('context', '').strip():
                    question += '\nCONTEXT:\n' + f['context']
                answer = f['initial_answer']
                aux = {'domain=' + r['domain']: 1.0,
                       'log_question_length': math.log1p(len(question))/10,
                       'log_answer_length': math.log1p(len(answer))/10}
                for k in ('mean_logprob', 'p10_logprob', 'min_logprob', 'available', 'token_count'):
                    value = (f.get('confidence') or {}).get(k)
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        aux['confidence_' + k] = (math.log1p(max(value, 0))/10 if k == 'token_count'
                                                  else float(np.clip(value, -30, 30))/10)
                for k in ('output_nonempty', 'truncated'):
                    value = (f.get('online_feedback') or {}).get(k)
                    if isinstance(value, bool): aux['online_' + k] = float(value)
                row = {k: r[k] for k in ('sample_id', 'group_id', 'domain', 'split', 'label')}
                row.update(delta=b-a, a=a, b=b, extra_tokens=detail['extra_guided_cost']['total_tokens'],
                           auxiliary=aux, question_sha256=sha(question.encode()), answer_sha256=sha(answer.encode()),
                           prompt_sha256=sha(' '.join((f['problem']+'\n'+f.get('context','')).split()).casefold().encode()))
                rows.append(row); texts.append((question, answer))
    assert len(rows) == 3372 and sum(r['label'] for r in rows) == 82
    for k in ('sample_id', 'group_id', 'prompt_sha256'):
        assert not {r[k] for r in rows if r['split']=='train'} & {r[k] for r in rows if r['split']=='val'}, k
    return rows, texts, hashes


def api_batch(batch, key):
    payload = json.dumps({'model': MODEL, 'dimensions': DIM, 'input': [text for _, text in batch]}).encode()
    for attempt in range(6):
        req = urllib.request.Request(URL, data=payload, headers={
            'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=90) as response:
                value = json.load(response)
            items = sorted(value['data'], key=lambda item: item['index'])
            assert [item['index'] for item in items] == list(range(len(batch)))
            vectors = np.asarray([item['embedding'] for item in items], dtype=np.float32)
            assert vectors.shape == (len(batch), DIM) and np.isfinite(vectors).all()
            assert (np.linalg.norm(vectors, axis=1) > 0).all()
            return batch, vectors, value.get('usage', {}), value.get('model', MODEL)
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504):
                raise RuntimeError('Embedding HTTP status ' + str(error.code)) from None
        except (urllib.error.URLError, TimeoutError):
            pass
        if attempt == 5: raise RuntimeError('Embedding API retries exhausted')
        time.sleep(min(20, 2 ** attempt))


def get_api_key(env_file=None):
    key = os.environ.get('ZHIPUAI_API_KEY', '').strip()
    if key:
        return key
    if env_file is not None:
        for line in env_file.read_text().splitlines():
            match = re.match(r'\s*(?:export\s+)?ZHIPUAI_API_KEY\s*=\s*(.*?)\s*$', line)
            if match:
                key = match[1].strip().strip("\"'")
                if key:
                    return key
    raise RuntimeError('Set ZHIPUAI_API_KEY or pass --env-file')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='Input router-dataset ZIP with SHA256.json')
    parser.add_argument('--work-dir', type=Path, required=True,
                        help='Directory for the cache, exported vectors and ZIP')
    parser.add_argument('--env-file', type=Path,
                        help='Optional local credentials file; environment takes precedence')
    args = parser.parse_args(argv)
    root = args.work_dir.resolve()
    output = root/'embedding3-ridge-input'
    if (output/'manifest.json').exists():
        print('Vectors already complete:', output, flush=True)
        return
    key = get_api_key(args.env_file)
    rows, texts, hashes = load_source(args.source)
    root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(root/'embedding3-cache.sqlite')
    connection.execute('CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB NOT NULL)')
    connection.execute('CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY, timestamp REAL, usage TEXT, model TEXT)')
    cached = {key: np.frombuffer(blob, dtype=np.float32).copy() for key, blob in connection.execute('SELECT key, vector FROM vectors')}
    views = [(chunks(q), chunks(a)) for q, a in texts]
    unique = {cache_key(text): text for pair in views for view in pair for text in view}
    pending = [(key, text) for key, text in unique.items() if key not in cached]
    batches = [pending[i:i+16] for i in range(0, len(pending), 16)]
    print(json.dumps({'samples':len(rows),'unique_chunks':len(unique),'cached_chunks':len(cached),'pending_batches':len(batches)}), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        # Keep only two requests in flight; failed runs resume from the content cache.
        queue = {}; next_batch = 0; completed = 0
        while next_batch < len(batches) or queue:
            while next_batch < len(batches) and len(queue) < 2:
                future = pool.submit(api_batch, batches[next_batch], key)
                queue[future] = next_batch; next_batch += 1
            done, _ = concurrent.futures.wait(queue, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                del queue[future]
                batch, vectors, usage, model = future.result()
                with connection:
                    for (cache_id, _), vector in zip(batch, vectors):
                        cached[cache_id] = vector
                        connection.execute('INSERT OR REPLACE INTO vectors VALUES (?,?)', (cache_id, vector.tobytes()))
                    connection.execute('INSERT INTO requests(timestamp,usage,model) VALUES (?,?,?)',
                                       (time.time(), json.dumps(usage), model))
                completed += 1
                if completed % 10 == 0 or completed == len(batches):
                    print('EMBEDDING_BATCHES', completed, '/', len(batches), flush=True)
    def aggregate(parts):
        weights = np.asarray([len(p.encode()) for p in parts], dtype=np.float64)
        vector = np.average(np.stack([cached[cache_key(p)] for p in parts]), weights=weights, axis=0)
        return (vector / np.linalg.norm(vector)).astype(np.float32)
    q = np.stack([aggregate(pair[0]) for pair in views])
    a = np.stack([aggregate(pair[1]) for pair in views])
    output.mkdir(exist_ok=True)
    vector_file = output/'vectors.npz'
    np.savez_compressed(vector_file, question=q, answer=a, sample_ids=np.asarray([r['sample_id'] for r in rows]))
    metadata = {'contract':CONTRACT,'model':MODEL,'dimensions_per_view':DIM,'chunk_bytes':CHUNK_BYTES,
                'aggregation':'UTF-8-byte weighted chunk mean followed by L2 normalization',
                'input_sha256':hashes,'rows':rows}
    (output/'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False))
    usage_total = sum(json.loads(v).get('total_tokens',0) for (v,) in connection.execute('SELECT usage FROM requests'))
    manifest = {'vectors_sha256':sha(vector_file.read_bytes()),
                'metadata_sha256':sha((output/'metadata.json').read_bytes()),'model':MODEL,
                'samples':len(rows),'api_total_tokens_in_cache':usage_total,'created_unix':time.time()}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    archive = root/'embedding3-ridge-input.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in output.iterdir():z.write(p,output.name+'/'+p.name)
    print('API_VECTORS_VERIFIED', archive, archive.stat().st_size, 'tokens', usage_total, flush=True)


if __name__ == '__main__':
    main()
