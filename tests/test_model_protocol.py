"""CPU-only checks of architecture, masking, and the frozen-base GTD stage.

Only synthetic tensors and fresh random initializations are used. These tests
do not load experimental checkpoints, access datasets, or run an optimizer.
"""
import unittest

import torch

from dsfn.models import (
    BASELINES,
    build_base_model,
    build_model,
    extract_logits,
    gtd_loss,
    wrap_gtd,
)
from dsfn.centaur import Centaur, build_centaur


class ModelProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def inputs(self, batch=2):
        return torch.randn(batch, 9, 1024,
                           generator=torch.Generator().manual_seed(731))

    def test_dsfn_architecture_matches_paper_parameter_count(self):
        model = build_model('dsfn', 6, seed=42)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 526361)
        score = model.base.rcd_fusion.score_mlp
        self.assertEqual((score[0].in_features, score[0].out_features), (130, 64))
        self.assertEqual((score[3].in_features, score[3].out_features), (64, 1))
        self.assertFalse(any(isinstance(m, (torch.nn.MultiheadAttention,
                                           torch.nn.TransformerEncoderLayer))
                             for m in model.modules()))

    def test_gtd_initial_identity_and_zero_scale_skips_module(self):
        model = build_model('dsfn', 6, seed=42).eval()
        tokens = model.features(self.inputs())
        with torch.no_grad():
            for adapter, token in zip(model.adapters, tokens):
                self.assertTrue(torch.equal(adapter(token), token))

        calls = [0, 0, 0]
        def counted(index):
            def hook(_module, _inputs, _output):
                calls[index] += 1
            return hook
        handles = [adapter.register_forward_hook(counted(i))
                   for i, adapter in enumerate(model.adapters)]
        try:
            with torch.no_grad():
                _, refined, _, _, _ = model.classify_detail(tokens)
            self.assertEqual(calls, [0, 1, 1])
            for before, after in zip(tokens, refined):
                self.assertTrue(torch.equal(before, after))
        finally:
            for handle in handles:
                handle.remove()

    def test_each_missing_modality_is_excluded_and_bypasses_all_gtd(self):
        model = build_model('dsfn', 6, seed=42).eval()
        for missing in range(3):
            with self.subTest(missing=missing):
                x = self.inputs()
                x[:, missing * 3:missing * 3 + 3] = 0
                calls = []
                handles = [adapter.register_forward_hook(
                    lambda module, _inputs, _output: calls.append(module))
                    for adapter in model.adapters]
                try:
                    logits, evidence = model.infer_detail(x)
                finally:
                    for handle in handles:
                        handle.remove()
                self.assertEqual(calls, [])
                self.assertEqual(tuple(logits.shape), (2, 6))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue(torch.equal(evidence['alpha'][:, missing],
                                            torch.zeros(2)))
                self.assertTrue(torch.allclose(evidence['alpha'].sum(1),
                                               torch.ones(2), atol=1e-6))
                self.assertFalse(evidence['availability'][:, missing].any())
                with torch.no_grad():
                    expected, *_ = model.base.available_detail(
                        x, evidence['availability'][0])
                self.assertTrue(torch.equal(logits, expected))

    def test_no_available_modality_is_rejected(self):
        model = build_model('dsfn', 6, seed=42).eval()
        x = self.inputs()
        x[0] = 0
        with self.assertRaisesRegex(ValueError, 'No available modality'):
            model.infer_detail(x)

    def test_gtd_backward_leaves_base_parameters_and_bn_frozen(self):
        base = build_base_model('dsfn', 6, seed=42)
        model = wrap_gtd(base, seed=42).train()
        self.assertFalse(model.base.training)
        self.assertTrue(all(not p.requires_grad for p in model.base.parameters()))
        before_buffers = {k: v.clone() for k, v in model.base.named_buffers()}
        clean = self.inputs()
        noisy = clean.clone()
        noisy[0, 3:6] += .25
        loss, _ = gtd_loss(model, clean, noisy, torch.tensor([0, 1]),
                           torch.ones(6),
                           (torch.tensor([True, False]), torch.tensor([1, 0])))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None for p in model.base.parameters()))
        self.assertTrue(all(torch.equal(before_buffers[k], v)
                            for k, v in model.base.named_buffers()))
        grads = [p.grad for p in model.adapters.parameters() if p.grad is not None]
        self.assertTrue(grads)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(torch.count_nonzero(g) > 0 for g in grads))

    def test_seven_comparison_models_emit_class_logits(self):
        x = self.inputs()
        for family in (*BASELINES, 'cnn_dbf'):
            with self.subTest(family=family):
                model = build_base_model(family, 3, seed=42).eval()
                with torch.no_grad():
                    logits = extract_logits(model(x))
                self.assertEqual(tuple(logits.shape), (2, 3))
                self.assertTrue(torch.isfinite(logits).all())

    def test_centaur_two_stage_output_shapes(self):
        for classes, dataset in ((6, 'rfly'), (3, 'uavfd')):
            with self.subTest(dataset=dataset):
                dae, classifier = build_centaur(classes, 42, dataset)
                model = Centaur(dae, classifier).eval()
                native = torch.rand(1, 9, 1024,
                                    generator=torch.Generator().manual_seed(19))
                with torch.no_grad():
                    reconstructed = dae(native)
                    logits = model(native)
                self.assertEqual(tuple(reconstructed.shape), (1, 9, 1024))
                self.assertEqual(tuple(logits.shape), (1, classes))
                self.assertTrue(torch.isfinite(logits).all())


if __name__ == '__main__':
    unittest.main()
