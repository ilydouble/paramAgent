import argparse
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import joblib

spec = importlib.util.spec_from_file_location('small_router', Path(__file__).with_name('train_small_gain_router.py'))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def row(i, split='train', label=0):
    return {'sample_id': split + str(i), 'group_id': split + str(i), 'domain': 'code',
            'split': split, 'label': label, 'features': {'problem': f'unique problem {split} {i}',
            'context': 'public evidence', 'initial_answer': 'uncertain buggy' if label else 'certain valid',
            'confidence': {'mean_logprob': -2.0, 'gold': 999}}}


class SmallRouterTests(unittest.TestCase):
    def test_offline_fields_cannot_change_features(self):
        a = row(1); b = copy.deepcopy(a)
        b.update(label=1, delta_success=1, gold='SECRET', guided='SECRET')
        b['features'].update(gold='SECRET', final_score=1, guided_output='SECRET')
        b['features']['confidence']['gold'] = 123
        self.assertEqual(r.visible(a), r.visible(b))
        self.assertNotIn('SECRET', str(r.visible(b)))

    def test_split_overlap_rejected(self):
        a = [row(1, label=0), row(2, label=1)]; b = [row(3, 'val')]
        r.validate_splits(a, b)
        b[0]['group_id'] = a[0]['group_id']
        with self.assertRaises(ValueError): r.validate_splits(a, b)
        b = [row(3, 'val')]; b[0]['features']['problem'] = a[0]['features']['problem']
        with self.assertRaises(ValueError): r.validate_splits(a, b)

    def test_validation_vocabulary_not_fitted(self):
        train = [row(1), row(2, label=1)]
        x, vec, dv = r.fit_features(train, 'tfidf')
        test = row(3, 'val'); test['features']['problem'] = 'unseenvalidationsecret'
        r.transform([test], vec, dv)
        self.assertNotIn('unseenvalidationsecret', vec.vocabulary_)

    def test_threshold_can_choose_no_calls_and_penalizes_harm(self):
        p = np.array([0.1, 0.9])
        t = r.select_threshold(p, np.array([-1, 0]), 0.02)
        self.assertFalse(np.any(p >= t))
        t = r.select_threshold(p, np.array([-1, 1]), 0.02)
        self.assertEqual(list(p >= t), [False, True])

    def test_counterfactual_gain_counts_regressions(self):
        rows = [row(1), row(2)]
        outcomes = {'train1': {'a': 0, 'b': 1, 'extra_tokens': 5},
                    'train2': {'a': 1, 'b': 0, 'extra_tokens': 7}}
        m = r.policy_metrics(rows, outcomes, np.array([True, True]), 0.02)
        self.assertEqual((m['improvements'], m['regressions'], m['net_success_gain']), (1, 1, 0))
        self.assertEqual(m['extra_tokens'], 12)

    def test_training_save_reload_predict_and_inputs_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root/'run'; out = root/'out'
            for split, n in [('train', 36), ('val', 12)]:
                folder = run/('labels-' + split); folder.mkdir(parents=True)
                rows = [row(i, split, int(i % 4 == 0)) for i in range(n)]
                labels = []
                for x in rows:
                    detail = {k: x[k] for k in ('sample_id', 'group_id', 'domain', 'split', 'label')}
                    detail.update(exclusion_reasons=[], delta_success=x['label'],
                        round_scores={'actor_only': [{'metrics': {'success': False}}],
                                      'guided': [{'metrics': {'success': bool(x['label'])}}]})
                    labels.append(detail)
                for name, data in [('router_dataset.jsonl', rows), ('labels.jsonl', labels)]:
                    (folder/name).write_text(''.join(json.dumps(x)+'\n' for x in data))
            args = argparse.Namespace(run_dir=str(run), output_dir=str(out), domains=['code'],
                backend='tfidf', encoder_path=None, seed=42, call_penalty=0.02, device='cpu', batch_size=8)
            r.train(args)
            report = json.loads((out/'report.json').read_text())
            self.assertEqual(report['train_n'], 36)
            self.assertEqual(len(list(r.read_jsonl(out/'val_predictions.jsonl'))), 12)
            pred = argparse.Namespace(model=str(out/'router.joblib'),
                input=str(run/'labels-val/router_dataset.jsonl'), output=str(root/'pred.jsonl'), device='cpu', batch_size=8)
            r.predict(pred)
            self.assertEqual(len(list(r.read_jsonl(root/'pred.jsonl'))), 12)
            with self.assertRaises(ValueError): r.train(args)


if __name__ == '__main__':
    unittest.main()
