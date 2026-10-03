import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from scripts.run_full_paired_router import main


class FullPairedLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        (self.output / 'pool').mkdir()
        rows = [{'sample_id': str(i), 'domain': 'qa', 'split': split}
                for i, split in enumerate(('train', 'train', 'val'))]
        (self.output / 'pool/tasks.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in rows))
        (self.output / 'pool/supervision.jsonl').touch()
        weights = self.output / 'weights'
        weights.mkdir()
        (weights / 'adapter_model.safetensors').write_bytes(b'fixture')
        model = {'endpoint': 'http://localhost:8000/v1', 'model_id': 'actor',
                 'revision': 'sha256:' + hashlib.sha256(b'fixture').hexdigest(),
                 'temperature': .2, 'max_tokens': 4096,
                 'weights_path': str(weights), 'adapter_path': str(weights)}
        for split in ('train', 'val'):
            settings = {'rounds': 5, 'max_per_domain': 3, 'seed': 42, 'split': split,
                        'actor': model, 'preference_models': {'qa': model}}
            (self.output / f'settings-{split}.json').write_text(json.dumps(settings))

    def invoke(self, execute=False, free=10 * 2**30):
        health = MagicMock()
        health.__enter__.return_value.status = 200
        models = io.BytesIO(json.dumps({'data': [{'id': 'actor', 'max_model_len': 16384}]}).encode())
        argv = ['launcher', '--output', str(self.output)] + (['--execute'] if execute else [])
        with patch('sys.argv', argv), patch('gain_router.paired.collect') as collect, \
                patch('scripts.run_full_paired_router.urllib.request.urlopen', side_effect=[health, models]), \
                patch('scripts.run_full_paired_router.shutil.disk_usage', return_value=SimpleNamespace(free=free)), \
                patch('scripts.run_full_paired_router.subprocess.Popen') as process:
            main()
            return collect, process

    def change_train(self, **changes):
        path = self.output / 'settings-train.json'
        settings = json.loads(path.read_text())
        settings.update(changes)
        path.write_text(json.dumps(settings))

    def test_preview_never_launches_generation(self):
        collect, process = self.invoke()
        self.assertEqual(collect.call_count, 2)
        self.assertTrue(all('execute' not in call.kwargs for call in collect.call_args_list))
        process.assert_not_called()

    def test_refuses_partial_pool(self):
        self.change_train(max_per_domain=1)
        with self.assertRaisesRegex(ValueError, 'truncate'):
            self.invoke()

    def test_refuses_smoke_rounds(self):
        self.change_train(rounds=2)
        with self.assertRaisesRegex(ValueError, 'five rounds'):
            self.invoke()

    def test_disk_reserve_blocks_child_start(self):
        with self.assertRaisesRegex(RuntimeError, 'Disk reserve'):
            self.invoke(execute=True, free=2**30)
        status = json.loads((self.output / 'launcher-status.json').read_text())
        self.assertEqual(status['state'], 'disk_low')


if __name__ == '__main__':
    unittest.main()
