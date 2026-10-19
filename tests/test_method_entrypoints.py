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
        from paramagent.experts import LocalPreference as expert
        from score.experts import LocalPreference as legacy_expert
        self.assertIs(expert, legacy_expert)
        self.assertIs(LocalPreference, expert)
        module = importlib.import_module('gain_router.paired')
        self.assertEqual(Path(module.__file__).parent.name, 'data')
        self.assertEqual(Path(module.__file__).parent.parent.name, 'paramagent')

    def test_paramagent_dispatch_preserves_arguments(self):
        from paramagent.runner import main
        with patch.object(sys, 'argv', ['runner', 'qa', '--max_iters', '3']), \
             patch('paramagent.runner.run_script') as run, \
             patch('sys.stderr') as stderr:
            main()
        run.assert_called_once_with('qa/mainQA_parametric.py', ['--max_iters', '3'])
        self.assertTrue(stderr.write.called)

    def test_paramagent_dispatches_offline_commands_without_baseline_fallback(self):
        from paramagent.runner import main
        with patch.object(sys, 'argv', ['runner', 'collect', '--help']), \
             patch('gain_router.paired.main') as data_main, \
             patch('paramagent.runner.run_script') as legacy:
            main()
        data_main.assert_called_once_with()
        legacy.assert_not_called()

    def test_parammem_variants_use_same_domain_pipeline_and_separate_results(self):
        from parammem.runner import main, SCRIPTS, MEMORY_ARGS
        with tempfile.TemporaryDirectory() as directory:
            memory = Path(directory) / 'memory.jsonl'
            memory.write_text('{}\n')
            for domain in SCRIPTS:
                for method in ('parammem', 'parammem-dpo'):
                    with self.subTest(domain=domain, method=method):
                        with patch.object(sys, 'argv', ['runner', domain, '--method', method,
                                          '--memory-file', str(memory), '--model', 'vllm:Qwen3.5-4B']), \
                             patch('parammem.runner.run_script') as run:
                            main()
                        enable, memory_arg = MEMORY_ARGS[domain]
                        run.assert_called_once_with(SCRIPTS[domain], [
                            '--model', 'vllm:Qwen3.5-4B', '--strategy', 'dot', enable,
                            memory_arg, str(memory), '--run_name', f'{method}_{domain}',
                            '--root_dir', f'results/{domain}/{method}'])

    def test_parammem_rejects_missing_or_conflicting_memories(self):
        from parammem.runner import main
        with tempfile.TemporaryDirectory() as directory:
            memory = Path(directory) / 'memory.jsonl'
            memory.write_text('{}\n')
            for path, extra in [(memory.with_name('missing.jsonl'), []),
                                (memory, ['--insight_json_path=another.jsonl'])]:
                with self.subTest(path=path, extra=extra):
                    with patch.object(sys, 'argv', ['runner', 'qa', '--method', 'parammem',
                                      '--memory-file', str(path), *extra]), \
                         patch('parammem.runner.run_script') as run, patch('sys.stderr'):
                        with self.assertRaises(SystemExit):
                            main()
                        run.assert_not_called()

    def test_parammem_preserves_explicit_run_and_budget(self):
        from parammem.runner import main
        with tempfile.TemporaryDirectory() as directory:
            memory = Path(directory) / 'memory.jsonl'
            memory.write_text('{}\n')
            extra = ['--run_name=custom', '--root_dir', directory,
                     '--strategy=reflexion', '--max_iters', '6']
            with patch.object(sys, 'argv', ['runner', 'math', '--method', 'parammem-dpo',
                              '--memory-file', str(memory), *extra]), \
                 patch('parammem.runner.run_script') as run:
                main()
            self.assertEqual(run.call_args.args[1], extra + [
                '--use_expert_module', '--insight_json_path', str(memory)])

    def test_generation_variants_forward_identical_sampling_arguments(self):
        from parammem.generate import main, SCRIPTS
        for domain in SCRIPTS:
            for method in ('parammem', 'parammem-dpo'):
                with self.subTest(domain=domain, method=method):
                    extra = ['--input_jsonl', 'input.jsonl', '--output_jsonl', 'out.jsonl',
                             '--temperature', '0.2', '--num_versions', '1', '--max_new_tokens', '1024']
                    with patch.object(sys, 'argv', ['generate', domain, '--method', method,
                                      '--base_model', 'sft-merged', '--lora_path', 'adapter', *extra]), \
                         patch('parammem.generate.run_script') as run:
                        main()
                    run.assert_called_once_with(SCRIPTS[domain], [
                        '--base_model', 'sft-merged', *extra, '--lora_path', 'adapter'])

    def test_generation_dpo_requires_adapter(self):
        from parammem.generate import main
        with patch.object(sys, 'argv', ['generate', 'code', '--method', 'parammem-dpo',
                          '--base_model', 'sft-merged']), \
             patch('parammem.generate.run_script') as run, patch('sys.stderr'):
            with self.assertRaises(SystemExit):
                main()
            run.assert_not_called()

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
