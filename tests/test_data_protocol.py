import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from dsfn.data import load_dataset

spec=importlib.util.spec_from_file_location('prepare_paper_data',Path(__file__).resolve().parents[1]/'prepare_paper_data.py')
prepare_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare_module)


class GroupedDataTests(unittest.TestCase):
    def test_released_uavfd_manifest_has_disjoint_groups(self):
        path=Path(__file__).resolve().parents[1]/'data/manifests/uavfd_flights.csv'
        rows=prepare_module.read_manifest(path)
        self.assertEqual(len(rows),18)
        self.assertEqual({s:sum(r['split']==s for r in rows) for s in ('train','val','test')},dict(train=12,val=3,test=3))

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.flights=self.root/'flights';self.flights.mkdir()

    def tearDown(self):self.temp.cleanup()

    def fixture(self):
        rows=[]
        for r,role in enumerate(('train','val','test')):
            for label in range(3):
                log='%s_%s'%(role,label)
                signal=np.random.RandomState(10+r*3+label).normal(r*100,1,(2048,9)).astype(np.float32)
                np.savez(self.flights/(log+'.npz'),signals=signal,label=np.int64(label))
                rows.append(dict(log_id=log,group_id=log,split=role))
        manifest=self.root/'manifest.csv'
        with manifest.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['log_id','group_id','split']);w.writeheader();w.writerows(rows)
        return manifest

    def test_strides_and_train_only_statistics(self):
        manifest=self.fixture();out=self.root/'prepared'
        prepare_module.prepare(self.flights,manifest,out,'uavfd');ctx=load_dataset(out)
        self.assertEqual({k:len(v) for k,v in ctx['ids'].items()},dict(train=9,val=9,test=6))
        expected=np.asarray(ctx['raw'][ctx['ids']['train']],dtype=np.float64)
        np.testing.assert_allclose(ctx['mean'],expected.mean(axis=(0,2)),rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(ctx['scale'],expected.std(axis=(0,2)),rtol=1e-6,atol=1e-6)
        self.assertLess(float(ctx['mean'].max()),1.)

    def test_group_leakage_rejected_before_windowing(self):
        manifest=self.fixture()
        with manifest.open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
        rows[-1]['group_id']=rows[0]['group_id']
        with manifest.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['log_id','group_id','split']);w.writeheader();w.writerows(rows)
        with self.assertRaisesRegex(ValueError,'group'):
            prepare_module.prepare(self.flights,manifest,self.root/'bad','uavfd')

    def test_duplicate_window_assignment_rejected(self):
        manifest=self.fixture();out=self.root/'prepared'
        prepare_module.prepare(self.flights,manifest,out,'uavfd')
        split=json.loads((out/'splits.json').read_text());split['test'][0]=split['train'][0]
        (out/'splits.json').write_text(json.dumps(split))
        with self.assertRaisesRegex(ValueError,'without overlap'):load_dataset(out)


if __name__=='__main__':unittest.main()
