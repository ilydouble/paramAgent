import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts import build_api_vectors as builder


class ApiVectorBuilderTests(unittest.TestCase):
    def test_complete_export_skips_credentials_source_and_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'embedding3-ridge-input'
            output.mkdir()
            manifest = output / 'manifest.json'
            manifest.write_text('{}')
            before = manifest.read_bytes()
            with patch.object(builder, 'get_api_key') as key, \
                 patch.object(builder, 'load_source') as source, \
                 patch.object(builder, 'api_batch') as api, \
                 contextlib.redirect_stdout(io.StringIO()):
                builder.main(['--source', str(root / 'absent.zip'), '--work-dir', str(root)])
            key.assert_not_called()
            source.assert_not_called()
            api.assert_not_called()
            self.assertEqual(manifest.read_bytes(), before)

    def test_environment_precedes_explicit_credentials_file(self):
        with patch.dict(os.environ, {'ZHIPUAI_API_KEY': 'test-key-only'}, clear=True):
            self.assertEqual(builder.get_api_key(Path('/nonexistent/test.env')), 'test-key-only')
        with tempfile.TemporaryDirectory() as directory:
            env = Path(directory) / 'test.env'
            env.write_text('export ZHIPUAI_API_KEY="fixture-key"\n')
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(builder.get_api_key(env), 'fixture-key')
                with self.assertRaises(RuntimeError):
                    builder.get_api_key()

    def test_relocated_cache_exports_offline_into_explicit_work_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'experiment'
            root.mkdir()
            question, answer = 'test question', 'test answer'
            q = np.arange(1, builder.DIM + 1, dtype=np.float32)
            a = np.ones(builder.DIM, dtype=np.float32)
            with sqlite3.connect(root / 'embedding3-cache.sqlite') as connection:
                connection.execute('CREATE TABLE vectors (key TEXT PRIMARY KEY, vector BLOB NOT NULL)')
                for text, vector in [(question, q), (answer, a)]:
                    connection.execute('INSERT INTO vectors VALUES (?,?)',
                                       (builder.cache_key(text), vector.tobytes()))
            source = Path(directory) / 'input.zip'
            rows = [{'sample_id': 'sample-1'}]
            with patch.dict(os.environ, {'ZHIPUAI_API_KEY': 'fixture-only'}, clear=True), \
                 patch.object(builder, 'load_source', return_value=(rows, [(question, answer)], {})) as load, \
                 patch.object(builder, 'api_batch', side_effect=AssertionError('network forbidden')) as api, \
                 contextlib.redirect_stdout(io.StringIO()):
                builder.main(['--source', str(source), '--work-dir', str(root)])
            load.assert_called_once_with(source)
            api.assert_not_called()
            output = root / 'embedding3-ridge-input'
            with np.load(output / 'vectors.npz') as vectors:
                np.testing.assert_allclose(vectors['question'][0], q / np.linalg.norm(q), rtol=1e-6)
                np.testing.assert_allclose(vectors['answer'][0], a / np.linalg.norm(a), rtol=1e-6)
            manifest = json.loads((output / 'manifest.json').read_text())
            self.assertEqual(manifest['vectors_sha256'],
                             hashlib.sha256((output / 'vectors.npz').read_bytes()).hexdigest())
            self.assertTrue((root / 'embedding3-ridge-input.zip').is_file())
