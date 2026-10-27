"""Small synthetic integration checks, never manuscript experiments."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from dsfn.protocol import CONDITIONS,EvaluationBank,grouped_metrics
from dsfn.training import train_one,evaluate_one


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self):
        cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/uavfd.json').read_text())
        cfg.update(base_epochs=1,gtd_epochs=1,baseline_epochs=1,
                   centaur_autoencoder_epochs=1,centaur_classifier_epochs=1)
        raw=np.random.RandomState(55).normal(size=(12,9,1024)).astype(np.float32)
        y=np.tile(np.arange(3),4).astype(np.int64)
        ctx=dict(raw=raw,y=y,groups=np.array(['flight_%s'%i for i in range(12)]),
                 ids={'train':np.arange(6),'val':np.arange(6,9),'test':np.arange(9,12)},
                 mean=raw[:6].mean((0,2)),scale=raw[:6].std((0,2)),weights=np.ones(3,np.float32),
                 spec={'dataset_id':'UAV-FD','num_classes':3,'class_names':cfg['class_names']})
        return ctx,cfg

    def test_noise_bank_and_regime_counts(self):
        ctx,cfg=self.fixture();bank=EvaluationBank(ctx,'val',123)
        noisy=bank.batch('noise:accel:0',0,3)
        np.testing.assert_array_equal(noisy[:,:3],bank.clean[:,:3])
        np.testing.assert_array_equal(noisy[:,6:],bank.clean[:,6:])
        masked=bank.batch('mask:gyro',0,3)
        self.assertTrue((masked[:,:3]==0).all())
        rows=[dict(condition=c,accuracy=i,macro_f1=i) for i,c in enumerate(CONDITIONS)]
        groups=grouped_metrics(rows)
        self.assertEqual(groups['noise_mean']['accuracy'],np.mean(range(1,19)))
        self.assertEqual(groups['masking_mean']['accuracy'],np.mean(range(19,22)))
        self.assertEqual(groups['low_snr_mean']['accuracy'],np.mean([4,5,6,10,11,12,16,17,18]))

    def round_trip(self,family):
        ctx,cfg=self.fixture()
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            run=train_one(ctx,cfg,family,42,tmp,'cpu')
            self.assertTrue((run/'training_complete.json').exists())
            result=evaluate_one(ctx,cfg,family,42,tmp,'cpu')
            self.assertEqual(set(result),{'clean','noise_mean','low_snr_mean','masking_mean'})
            with np.load(run/'test/predictions.npz') as predictions:
                self.assertEqual(predictions['logits'].shape,(22,3,3))
            if family=='dsfn':
                self.assertTrue((run/'test/branch_quality.csv').exists())
            with self.assertRaises(FileExistsError):
                train_one(ctx,cfg,family,42,tmp,'cpu')

    def test_uavfd_original_noise_subset_replay(self):
        ctx,cfg=self.fixture()
        ctx['spec'].update(test_noise_original_count=6,test_noise_positions=[0,2,4])
        bank=EvaluationBank(ctx,'test',20261021)
        expected=np.random.RandomState(20261021).standard_normal((6,9,1024)).astype(np.float32)[[0,2,4]]
        np.testing.assert_array_equal(bank.noise,expected)
        self.assertTrue(bank.replayed_subset)

    def test_dsfn_train_load_evaluate_round_trip(self):
        self.round_trip('dsfn')

    def test_centaur_train_load_evaluate_round_trip(self):
        self.round_trip('centaur')


if __name__=='__main__':unittest.main()
