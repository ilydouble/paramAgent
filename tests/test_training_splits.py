import ast
import argparse
import json
import re
import tempfile
import unittest
from pathlib import Path

from split_protocol import assign_exact_groups, content_group_id, file_sha256
from training_splits import (add_split_arguments, example_identity, partition_examples,
                             validate_training_artifact)

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    examples = [{**example_identity({}, f'problem {i}'), 'text': f'problem {i}'} for i in range(10)]
    records = [{**row, 'sample_id': f'sample:{i}', 'stratum': 'code'} for i, row in enumerate(examples)]
    _, manifest = assign_exact_groups(records, seed=42, targets={'code': {'train': 6, 'val': 2, 'test': 2}})
    return examples, manifest


class TrainingSplitsTests(unittest.TestCase):
    def test_sft_and_dpo_share_membership_and_never_include_test(self):
        rows, manifest = fixture()
        sft, counts = partition_examples(rows, manifest)
        dpo, _ = partition_examples(list(reversed(rows)), manifest)
        self.assertEqual(counts, {'train': 6, 'val': 2, 'test_excluded': 2})
        for split in ('train', 'val'):
            self.assertEqual({r['group_id'] for r in sft[split]}, {r['group_id'] for r in dpo[split]})
        self.assertTrue(set(r['group_id'] for r in sft['train']).isdisjoint(r['group_id'] for r in sft['val']))

    def test_unmatched_requires_explicit_exclusion_and_is_audited(self):
        rows, manifest = fixture()
        rows.append(example_identity({}, 'outside frozen pool'))
        with self.assertRaisesRegex(ValueError, 'no manifest match'):
            partition_examples(rows, manifest)
        _, counts = partition_examples(rows, manifest, exclude_unmatched=True)
        self.assertEqual(counts['unmatched_excluded'], 1)

    def test_stale_split_and_forged_identity_fail(self):
        rows, manifest = fixture()
        rows[0]['split_seed'] = 123
        with self.assertRaisesRegex(ValueError, 'metadata'):
            partition_examples(rows, manifest)
        with self.assertRaisesRegex(ValueError, 'group_id'):
            example_identity({'group_id': 'forged'}, 'original')

    def test_empty_val_never_falls_back_to_random_split(self):
        rows, manifest = fixture()
        selected = [r for r in rows if next(a['split'] for a in manifest['assignments'] if a['group_id'] == r['group_id']) == 'train']
        with self.assertRaisesRegex(ValueError, 'Both frozen'):
            partition_examples(selected, manifest)

    def test_manifest_argument_is_required(self):
        parser = argparse.ArgumentParser()
        add_split_arguments(parser)
        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_model_provenance_rejects_history_wrong_seed_test_and_tampering(self):
        rows, manifest = fixture()
        partitions, _ = partition_examples(rows, manifest)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'Missing frozen'):
                validate_training_artifact(root, manifest)
            artifact = root / 'adapter_model.safetensors'
            artifact.write_bytes(b'weights')
            report = {'status': 'complete', 'stage': 'sft', 'domain': 'code', 'split_seed': 42,
                      'assignment_sha256': manifest['assignment_sha256'],
                      'group_ids': {k: [r['group_id'] for r in v] for k, v in partitions.items()},
                      'artifacts': {artifact.name: file_sha256(artifact)}}
            path = root / 'training_split.json'
            path.write_text(json.dumps(report))
            validate_training_artifact(root, manifest, stage='sft', domain='code')
            for change in ({'split_seed': 123}, {'status': 'prepared'}, {'stage': 'dpo'}, {'domain': 'qa'}):
                path.write_text(json.dumps({**report, **change}))
                with self.subTest(change=change), self.assertRaises(ValueError):
                    validate_training_artifact(root, manifest, stage='sft', domain='code')
            path.write_text(json.dumps({**report, 'group_ids': {'train': [a['group_id'] for a in manifest['assignments'] if a['split'] == 'test'], 'val': []}}))
            with self.assertRaisesRegex(ValueError, 'held-out'):
                validate_training_artifact(root, manifest)
            path.write_text(json.dumps(report))
            artifact.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                validate_training_artifact(root, manifest)

    def test_all_six_loaders_preserve_raw_identity_before_prompt_cleanup(self):
        for domain, title in [('code', 'Code'), ('math', 'Math'), ('qa', 'QA')]:
            for dpo in (False, True):
                path = ROOT / domain / f'LoRA_Qwen35_{title}_{"DPO_" if dpo else ""}4090.py'
                tree = ast.parse(path.read_text())
                names = {'load_preferences'} if dpo else {'load_code_examples', 'load_examples', 'load_qa_examples', '_normalize_func_sign'}
                funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
                namespace = {'json': json, 're': re, 'example_identity': example_identity}
                exec(compile(ast.Module(body=funcs, type_ignores=[]), str(path), 'exec'), namespace)
                field = {'code': 'func_sign', 'math': 'problem', 'qa': 'question'}[domain]
                text = '```python\ndef f(x):\n    pass\n```' if domain == 'code' else 'Problem with   spacing'
                row = {field: text, 'pitfalls': 'valid pitfalls', 'decomposition': 'valid reasoning',
                       'chosen_pitfalls': 'chosen', 'rejected_pitfalls': 'rejected',
                       'chosen_decomposition': 'chosen', 'rejected_decomposition': 'rejected'}
                with tempfile.TemporaryDirectory() as tmp:
                    dataset = Path(tmp) / 'data.json'
                    dataset.write_text(json.dumps([row, row]) if not dpo and domain != 'qa' else '\n'.join(json.dumps(row) for _ in range(2)))
                    loader = namespace['load_preferences' if dpo else {'code': 'load_code_examples', 'math': 'load_examples', 'qa': 'load_qa_examples'}[domain]]
                    # Code DPO's normalization helper comes from its SFT module.
                    if dpo and domain == 'code':
                        sft_tree = ast.parse((ROOT / 'code/LoRA_Qwen35_Code_4090.py').read_text())
                        fn = next(n for n in sft_tree.body if isinstance(n, ast.FunctionDef) and n.name == '_normalize_func_sign')
                        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
                    loaded = loader(str(dataset))
                    self.assertTrue(all(r['group_id'] == content_group_id(text) for r in loaded), path)


if __name__ == '__main__':
    unittest.main()
