"""Centaur's separate DAE and classifier stages used in the revised comparison.

The classifier trains on complete native-coordinate training windows, not DAE
reconstructions. Validation and inference always pass through the frozen DAE.
"""
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .centaur import affine_native, build_centaur, corrupt_mode4, stage_seed
from .evaluate import save_csv, save_json
from .protocol import CONDITIONS, metrics, set_seed


def _dataset_kind(ctx, config):
    name = str(config.get('dataset', ctx['spec']['dataset_id'])).lower().replace('-', '')
    if name in ('rfly', 'rflymad'):
        return 'rfly'
    if name == 'uavfd':
        return 'uavfd'
    raise ValueError('Centaur stage seeds are defined for RflyMAD and UAV-FD')


def _cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _windows(ctx, ids):
    values = (ctx['raw'][ids] - ctx['mean'][None, :, None]) / ctx['scale'][None, :, None]
    if values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError('Centaur requires finite standardized float32 windows')
    return values


def _statistics(ctx):
    minimum = np.full(9, np.inf, dtype=np.float32)
    maximum = np.full(9, -np.inf, dtype=np.float32)
    for start in range(0, len(ctx['ids']['train']), 64):
        values = _windows(ctx, ctx['ids']['train'][start:start + 64])
        minimum = np.minimum(minimum, values.min((0, 2)))
        maximum = np.maximum(maximum, values.max((0, 2)))
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all() or not (maximum > minimum).all():
        raise ValueError('Centaur min-max statistics require nonconstant training channels')
    return dict(minimum=minimum, maximum=maximum)


def _native(values, stats):
    x = torch.from_numpy(np.ascontiguousarray(values))
    return affine_native(x, stats['minimum'], stats['maximum'])[0]


@torch.no_grad()
def _reconstruct(dae, values, stats, device):
    dae.eval()
    chunks = []
    for start in range(0, len(values), 16):
        native = _native(values[start:start + 16], stats)
        chunks.append(dae(native.to(device)).cpu().numpy())
    result = np.concatenate(chunks)
    if result.shape != values.shape or result.dtype != np.float32 or not np.isfinite(result).all():
        raise ValueError('Invalid Centaur reconstruction')
    return result


@torch.no_grad()
def _classifier_predictions(classifier, values, device):
    classifier.eval()
    chunks = []
    for start in range(0, len(values), 16):
        x = torch.from_numpy(np.ascontiguousarray(values[start:start + 16])).to(device)
        chunks.append(classifier(x).cpu().numpy())
    result = np.concatenate(chunks)
    if not np.isfinite(result).all():
        raise ValueError('Invalid Centaur classifier logits')
    return result


def train_centaur(ctx, config, seed, out, device, val_bank):
    """Train DAE100 and classifier120 independently and return stage metadata.

    Root orchestration owns config.json and the training-complete marker. This
    function writes the selected classifier, fixed DAE and their histories.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dataset = _dataset_kind(ctx, config)
    classes = int(ctx['spec']['num_classes'])
    dae_epochs = int(config.get('centaur_autoencoder_epochs', 100))
    classifier_epochs = int(config.get('centaur_classifier_epochs', 120))
    if dae_epochs < 1 or classifier_epochs < 1:
        raise ValueError('Both Centaur training stages require positive epochs')
    stats = _statistics(ctx)
    np.savez(out / 'internal_normalizer.npz', **stats)
    dae, classifier = build_centaur(classes, seed=seed, dataset=dataset)
    initial_classifier = _cpu_state(classifier)
    torch.save(_cpu_state(dae), out / 'dae_initial.pth')
    torch.save(initial_classifier, out / 'initial_model.pth')
    del classifier
    started = time.monotonic()

    # Stage 1: Gaussian noise plus independent-channel exponential missing gaps.
    dae_seed = stage_seed(dataset, seed, 'dae')
    set_seed(dae_seed)
    dae = dae.to(device)
    optimizer = torch.optim.RMSprop(dae.parameters(), lr=1e-4, momentum=.1)
    rng = np.random.RandomState(dae_seed)
    generator = torch.Generator(device='cpu').manual_seed(dae_seed)
    history = []
    for epoch in range(1, dae_epochs + 1):
        dae.train()
        permutation = torch.randperm(len(ctx['ids']['train']), generator=generator).numpy()
        loss_sum, samples = 0., 0
        for start in range(0, len(permutation), 64):
            ids = ctx['ids']['train'][permutation[start:start + 64]]
            target = _native(_windows(ctx, ids), stats)
            corrupted, _ = corrupt_mode4(target, rng)
            optimizer.zero_grad(set_to_none=True)
            for micro in range(0, len(ids), 16):
                target_micro = target[micro:micro + 16].to(device)
                loss = F.mse_loss(dae(corrupted[micro:micro + 16].to(device)), target_micro)
                loss = loss * len(target_micro) / len(ids)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite Centaur DAE loss')
                loss.backward()
                loss_sum += float(loss.detach()) * len(ids)
            if any(not torch.isfinite(p.grad).all() for p in dae.parameters() if p.grad is not None):
                raise FloatingPointError('Non-finite Centaur DAE gradient')
            optimizer.step()
            samples += len(ids)
        row = dict(epoch=epoch, mse=loss_sum / samples, elapsed_seconds=time.monotonic() - started)
        history.append(row)
        save_csv(out / 'dae_history.csv', history)
        save_json(out / 'status.json', dict(phase='dae', epoch=epoch, target=dae_epochs))
        print('CENTAUR_DAE', seed, row, flush=True)
    # Keep the manuscript filename for the default registered 100-epoch stage.
    dae_path = out / ('dae_fixed%s.pth' % dae_epochs)
    torch.save(_cpu_state(dae), dae_path)
    dae.eval()
    for parameter in dae.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    del optimizer

    # Cache validation reconstructions once. They never become training inputs.
    cached = {}
    for condition in CONDITIONS[:19]:
        values = np.concatenate([val_bank.batch(condition, start, start + 64)
                                 for start in range(0, len(val_bank.y), 64)])
        cached[condition] = _reconstruct(dae, values, stats, device)
    save_json(out / 'validation_bank.json', dict(noise_seed=val_bank.noise_seed,
              sample_ids=val_bank.ids.tolist(), dae_checkpoint=dae_path.name,
              classifier_selection_conditions=list(CONDITIONS[1:19])))
    del dae

    # Stage 2: fresh stage-specific classifier, original complete train windows.
    unused_dae, classifier = build_centaur(classes, seed=seed, dataset=dataset)
    del unused_dae
    if any(not torch.equal(value, initial_classifier[name])
           for name, value in classifier.state_dict().items()):
        raise AssertionError('Classifier stage initialization changed')
    del initial_classifier
    classifier_seed = stage_seed(dataset, seed, 'classifier')
    set_seed(classifier_seed)
    classifier = classifier.to(device)
    weights = torch.as_tensor(ctx['weights'], device=device)
    optimizer = torch.optim.SGD(classifier.parameters(), lr=.01, momentum=.9, weight_decay=1e-4)
    generator = torch.Generator(device='cpu').manual_seed(classifier_seed)
    history, best, best_epoch = [], -1., 0
    for epoch in range(1, classifier_epochs + 1):
        classifier.train()
        permutation = torch.randperm(len(ctx['ids']['train']), generator=generator).numpy()
        loss_sum, samples = 0., 0
        for start in range(0, len(permutation), 64):
            ids = ctx['ids']['train'][permutation[start:start + 64]]
            x = _native(_windows(ctx, ids), stats)
            y = torch.from_numpy(ctx['y'][ids]).to(device)
            denominator = weights[y].sum()
            optimizer.zero_grad(set_to_none=True)
            for micro in range(0, len(ids), 16):
                logits = classifier(x[micro:micro + 16].to(device))
                loss = F.cross_entropy(logits, y[micro:micro + 16], weight=weights,
                                       reduction='sum') / denominator
                if not torch.isfinite(loss):
                    raise FloatingPointError('Non-finite Centaur classifier loss')
                loss.backward()
                loss_sum += float(loss.detach()) * len(ids)
            if any(not torch.isfinite(p.grad).all() for p in classifier.parameters() if p.grad is not None):
                raise FloatingPointError('Non-finite Centaur classifier gradient')
            optimizer.step()
            samples += len(ids)
        scores = [metrics(val_bank.y, _classifier_predictions(classifier, cached[c], device), classes)[0]
                  for c in CONDITIONS[:19]]
        score = float(np.mean([value['macro_f1'] for value in scores[1:]]))
        if score > best:
            best, best_epoch = score, epoch
            torch.save(_cpu_state(classifier), out / 'best_model.pth')
        row = dict(epoch=epoch, train_loss=loss_sum / samples,
                   clean_accuracy=scores[0]['accuracy'], clean_macro_f1=scores[0]['macro_f1'],
                   noise_accuracy=float(np.mean([value['accuracy'] for value in scores[1:]])),
                   noise_macro_f1=score, learning_rate=.01, elapsed_seconds=time.monotonic() - started)
        history.append(row)
        save_csv(out / 'history.csv', history)
        save_json(out / 'status.json', dict(phase='classifier', epoch=epoch,
                  target=classifier_epochs, best_epoch=best_epoch, best_score=best))
        print('CENTAUR_CLASSIFIER', seed, row, flush=True)
    torch.save(_cpu_state(classifier), out / ('fixed%s.pth' % classifier_epochs))
    metadata = dict(best_epoch=best_epoch, best_validation_noise_macro_f1=best,
                    base_epochs=classifier_epochs, dae_epochs=dae_epochs,
                    classifier_epochs=classifier_epochs, dae_checkpoint=dae_path.name,
                    dae_stage_seed=dae_seed, classifier_stage_seed=classifier_seed,
                    effective_batch_size=64, micro_batch_size=16, ema=False,
                    dae_optimizer='RMSprop', dae_lr=1e-4, dae_momentum=.1,
                    dae_gaussian_std=.1, missing_interval_normal_mean=480,
                    missing_interval_corrupted_mean=300,
                    classifier_optimizer='SGD', classifier_lr=.01,
                    classifier_momentum=.9, classifier_weight_decay=1e-4,
                    classifier_training_inputs='original complete train windows, not DAE reconstructions',
                    test_used_for_selection=False, elapsed_seconds=time.monotonic() - started)
    save_json(out / 'centaur_result.json', metadata)
    return metadata


class CentaurInference(nn.Module):
    """Canonical standardized windows -> missing sentinel -> min-max -> DAE -> classifier."""
    def __init__(self, dae, classifier, minimum, maximum, micro_batch_size=16):
        super().__init__()
        self.dae, self.classifier = dae, classifier
        self.register_buffer('minimum', torch.as_tensor(minimum, dtype=torch.float32))
        self.register_buffer('maximum', torch.as_tensor(maximum, dtype=torch.float32))
        self.micro_batch_size = int(micro_batch_size)
        if self.minimum.shape != (9,) or self.maximum.shape != (9,) or self.micro_batch_size < 1:
            raise ValueError('Invalid Centaur inference normalization/batch size')
        if not torch.isfinite(self.minimum).all() or not torch.isfinite(self.maximum).all() or not (self.maximum > self.minimum).all():
            raise ValueError('Invalid Centaur inference min-max statistics')

    def forward(self, canonical):
        # The official adaptation uses micro-batches of 16 for its large
        # temporal-attention classifier, even when the outer evaluator uses 64.
        chunks = []
        for start in range(0, len(canonical), self.micro_batch_size):
            native, _ = affine_native(canonical[start:start + self.micro_batch_size],
                                      self.minimum, self.maximum)
            chunks.append(self.classifier(self.dae(native)))
        if not chunks:
            raise ValueError('Centaur input batch must be nonempty')
        return torch.cat(chunks)


def load_centaur(ctx, config, seed, out, device):
    """Load the independently selected classifier and fixed DAE for evaluation."""
    out = Path(out)
    dataset = _dataset_kind(ctx, config)
    dae_epochs = int(config.get('centaur_autoencoder_epochs', 100))
    with np.load(out / 'internal_normalizer.npz', allow_pickle=False) as values:
        minimum, maximum = values['minimum'].copy(), values['maximum'].copy()
    dae, classifier = build_centaur(int(ctx['spec']['num_classes']), seed=seed, dataset=dataset)
    dae.load_state_dict(torch.load(out / ('dae_fixed%s.pth' % dae_epochs),
                                  map_location='cpu', weights_only=True), strict=True)
    classifier.load_state_dict(torch.load(out / 'best_model.pth', map_location='cpu', weights_only=True), strict=True)
    return CentaurInference(dae, classifier, minimum, maximum).to(device).eval()
