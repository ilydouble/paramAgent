import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from shared.entrypoints import run_script


class MethodEntrypointTests(unittest.TestCase):
    def test_compatibility_sources_and_expert_share_identity(self):
        from gain_router.local_preference import LocalPreference
        from score.experts import LocalPreference as expert
        self.assertIs(LocalPreference, expert)
        module = importlib.import_module('gain_router.paired')
        self.assertEqual(Path(module.__file__).parent.name, 'data')
        self.assertEqual(Path(module.__file__).parent.parent.name, 'score')

    def test_paramagent_dispatch_preserves_arguments(self):
        from paramagent.runner import main
        with patch.object(sys, 'argv', ['runner', 'qa', '--max_iters', '3']), \
             patch('paramagent.runner.run_script') as run:
            main()
        run.assert_called_once_with('qa/mainQA_parametric.py', ['--max_iters', '3'])

    def test_training_dispatch_all_domains_and_stages(self):
        from training.runner import main
        for stage in ('sft', 'dpo'):
            for domain, name in (('code', 'Code'), ('math', 'Math'), ('qa', 'QA')):
                with patch.object(sys, 'argv', ['runner', domain, '--lr', '1e-6']), \
                     patch('training.runner.run_script') as run:
                    main(stage)
                suffix = '_DPO' if stage == 'dpo' else ''
                run.assert_called_once_with(f'{domain}/LoRA_Qwen35_{name}{suffix}_4090.py',
                                            ['--lr', '1e-6'])

    def test_script_context_restored_on_failure(self):
        argv, paths = sys.argv, sys.path[:]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'domain').mkdir()
            (root / 'domain' / 'entry.py').write_text(
                'import sys\nfrom pathlib import Path\n'
                'assert sys.argv[1:] == ["argument"]\n'
                'assert sys.path[0] == str(Path(__file__).parent)\n'
                'raise RuntimeError("expected")\n')
            with patch('shared.entrypoints.ROOT', root):
                with self.assertRaisesRegex(RuntimeError, 'expected'):
                    run_script('domain/entry.py', ['argument'])
        self.assertIs(sys.argv, argv)
        self.assertEqual(sys.path, paths)
