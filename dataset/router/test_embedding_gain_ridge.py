import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer

import train_embedding_gain_ridge as r


def fixture(root, flip_val=False):
    root=Path(root); run=root/'run'; vectors=root/'vectors'; vectors.mkdir(parents=True)
    rows=[]; source_hashes={}
    for split,n in [('train',36),('val',12)]:
        folder=run/('labels-'+split);folder.mkdir(parents=True)
        originals=[]
        for i in range(n):
            delta=(-1,0,1)[i%3]
            if split=='val' and flip_val:delta=-delta
            a=int(delta<=0);b=int(delta>=0)
            if delta==0:a=b=0
            row={'sample_id':split+str(i),'group_id':split+str(i),'prompt_sha256':split+str(i),
                 'domain':'code','split':split,'label':int(delta>0),'delta':delta,'a':a,'b':b,
                 'extra_tokens':5,'auxiliary':{'domain=code':1.,'log_answer_length':0.2}}
            rows.append(row)
            record={k:row[k] for k in ('sample_id','group_id','domain','split','label')}
            record.update(delta_success=delta,exclusion_reasons=[],round_scores={
                'actor_only':[{'metrics':{'success':bool(a)}}], 'guided':[{'metrics':{'success':bool(b)}}]})
            originals.append(record)
        for name in ['router_dataset.jsonl','labels.jsonl']:
            p=folder/name;p.write_text(''.join(json.dumps(v)+'\n' for v in originals));source_hashes[str(p.relative_to(run))]=r.digest(p)
    rng=np.random.default_rng(7)
    q=rng.normal(size=(len(rows),1024));q=(q/np.linalg.norm(q,axis=1,keepdims=True)).astype(np.float32)
    a=rng.normal(size=(len(rows),1024));a=(a/np.linalg.norm(a,axis=1,keepdims=True)).astype(np.float32)
    np.savez_compressed(vectors/'vectors.npz',question=q,answer=a,sample_ids=np.array([v['sample_id'] for v in rows]))
    meta={'contract':r.CONTRACT,'model':'embedding-3','dimensions_per_view':1024,'chunk_bytes':2800,
          'aggregation':'byte weighted mean and L2','input_sha256':source_hashes,'rows':rows}
    (vectors/'metadata.json').write_text(json.dumps(meta))
    manifest={'vectors_sha256':r.digest(vectors/'vectors.npz'),'metadata_sha256':r.digest(vectors/'metadata.json')}
    (vectors/'manifest.json').write_text(json.dumps(manifest))
    return run,vectors


class RidgeTests(unittest.TestCase):
    def test_auxiliary_never_uses_offline_values(self):
        x={'auxiliary':{'domain=code':1.,'gold':999.,'delta':1.},'label':1,'delta':1}
        self.assertEqual(r.auxiliary(x),{'domain=code':1.})
        x['auxiliary']['log_answer_length']=float('nan')
        with self.assertRaises(ValueError):r.auxiliary(x)

    def test_matrix_ignores_target_and_future_results(self):
        rows=[{'auxiliary':{'domain=code':1.},'delta':0}]
        dv=DictVectorizer(sparse=False);e=np.ones((1,2048));a=r.matrix(e,rows,dv,fit=True)
        rows[0].update(delta=1,gold='secret',guided_output='secret')
        np.testing.assert_array_equal(a,r.matrix(e,rows,dv,fit=False))

    def test_cache_checksum_and_id_alignment(self):
        with tempfile.TemporaryDirectory() as t:
            run,v=fixture(t);rows,x,_,_=r.load_vectors(v,run)
            with np.load(v/'vectors.npz',allow_pickle=False) as d:q,a,ids=d['question'],d['answer'],d['sample_ids']
            with self.assertRaises(ValueError):r.validate(rows,q,a,ids[::-1],labeled=True)
            with (v/'metadata.json').open('a') as f:f.write(' ')
            with self.assertRaises(ValueError):r.load_vectors(v,run)

    def test_original_target_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            run,v=fixture(t);p=v/'metadata.json';meta=json.loads(p.read_text());x=meta['rows'][0]
            x.update(delta=1,a=0,b=1,label=1);p.write_text(json.dumps(meta))
            m=json.loads((v/'manifest.json').read_text());m['metadata_sha256']=r.digest(p);(v/'manifest.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError):r.load_vectors(v,run)

    def test_training_independent_of_validation_targets_and_reload(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);models=[];reports=[]
            for name,flip in [('first',False),('flipped',True)]:
                run,v=fixture(root/name,flip_val=flip);out=root/(name+'-out')
                args=argparse.Namespace(vectors=str(v),run_dir=str(run),output_dir=str(out),seed=42,call_penalty=0.02)
                r.train(args);model=joblib.load(out/'router.joblib');models.append(model)
                reports.append(json.loads((out/'report.json').read_text()))
                pred=argparse.Namespace(vectors=str(v),model=str(out/'router.joblib'),output=str(root/(name+'-predict.jsonl')))
                r.predict(pred)
                values=[json.loads(s) for s in Path(pred.output).read_text().split('\n') if s.strip()]
                self.assertEqual(len(values),48)
                saved=[json.loads(s) for s in (out/'val_predictions.jsonl').read_text().split('\n') if s.strip()]
                np.testing.assert_allclose([x['predicted_delta'] for x in values[-12:]], [x['predicted_delta'] for x in saved],atol=1e-12)
                with self.assertRaises(ValueError):r.train(args)
            np.testing.assert_array_equal(models[0]['model'].coef_,models[1]['model'].coef_)
            self.assertEqual(models[0]['threshold'],models[1]['threshold'])
            self.assertEqual(reports[0]['selected_alpha'],reports[1]['selected_alpha'])


if __name__=='__main__':
    unittest.main()
