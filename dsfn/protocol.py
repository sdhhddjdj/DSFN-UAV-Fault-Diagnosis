"""Shared evaluation conditions for the revised manuscript."""
import random
import numpy as np
import torch

SEEDS = (42, 2025, 3407, 1234, 5678)
MODALITIES = ('gyro', 'accel', 'vel')
SNR_DB = (20, 10, 5, 0, -5, -10)
FAMILIES = ('dsfn', 'resnet1d', 'tcn', 'inceptiontime', 'lstm_fcn',
            'drsn', 'mra_cnn', 'cnn_dbf', 'centaur')
CONDITIONS = ('clean',) + tuple('noise:%s:%s' % (m, s) for m in MODALITIES for s in SNR_DB) + tuple('mask:' + m for m in MODALITIES)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EvaluationBank:
    """One common Gaussian bank for every model and seed in a dataset split.

    An archived standard-Gaussian .npy bank can be supplied for exact replay.
    The generated bank is deterministic but does not replace missing flight splits.
    """
    def __init__(self, ctx, split, seed, noise_path=None):
        if split not in ('val', 'test'):
            raise ValueError('Evaluation split must be val or test')
        self.ids = ctx['ids'][split]
        self.y = ctx['y'][self.ids]
        raw = np.asarray(ctx['raw'][self.ids], dtype=np.float32)
        self.clean = (raw - ctx['mean'][None, :, None]) / ctx['scale'][None, :, None]
        if not np.isfinite(self.clean).all():
            raise ValueError('Evaluation windows must be finite after normalization')
        self.noise_seed = int(seed)
        self.replayed_subset = False
        if noise_path:
            self.noise = np.load(noise_path, allow_pickle=False)
        elif split == 'test' and 'test_noise_positions' in ctx['spec']:
            positions = np.asarray(ctx['spec']['test_noise_positions'], dtype=np.int64)
            count = int(ctx['spec']['test_noise_original_count'])
            if positions.shape != (len(self.ids),) or len(np.unique(positions)) != len(positions) or positions.min()<0 or positions.max()>=count:
                raise ValueError('Invalid original-test noise positions')
            original_shape = (count,) + self.clean.shape[1:]
            self.noise = np.random.RandomState(seed).standard_normal(original_shape).astype(np.float32)[positions]
            self.replayed_subset = True
        else:
            self.noise = np.random.RandomState(seed).standard_normal(self.clean.shape).astype(np.float32)
        if self.noise.shape != self.clean.shape or not np.isfinite(self.noise).all():
            raise ValueError('Noise bank must be finite and match the split [N,9,1024]')

    def batch(self, condition, start, stop):
        if condition not in CONDITIONS:
            raise ValueError(condition)
        x = self.clean[start:stop].copy()
        if condition == 'clean':
            return x
        parts = condition.split(':')
        m = MODALITIES.index(parts[1]); sl = slice(3*m, 3*m+3)
        if parts[0] == 'mask':
            x[:, sl] = 0.
        else:
            power = np.mean(self.clean[start:stop, sl] ** 2, axis=(1, 2), keepdims=True).clip(1e-12)
            x[:, sl] += self.noise[start:stop, sl] * np.sqrt(power / np.float32(10 ** (float(parts[2]) / 10)))
        return x


def metrics(y, logits, num_classes):
    pred = logits.argmax(1)
    cm = np.bincount(y * num_classes + pred, minlength=num_classes**2).reshape(num_classes, num_classes)
    den = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * cm.diagonal(), den, out=np.zeros(num_classes), where=den > 0)
    return {'accuracy': float(cm.trace() / len(y)), 'macro_f1': float(f1.mean())}, cm


def condition_group(condition):
    if condition == 'clean':
        return ('clean',)
    if condition.startswith('mask:'):
        return ('masking_mean',)
    return ('noise_mean', 'low_snr_mean') if float(condition.split(':')[2]) <= 0 else ('noise_mean',)


def grouped_metrics(rows):
    output = {}
    for group in ('clean', 'noise_mean', 'low_snr_mean', 'masking_mean'):
        selected = [r for r in rows if group in condition_group(r['condition'])]
        output[group] = {m: float(np.mean([r[m] for r in selected])) for m in ('accuracy', 'macro_f1')}
    return output
