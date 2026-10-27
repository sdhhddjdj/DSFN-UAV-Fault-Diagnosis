"""Training augmentations ported from the archived experiment recipe."""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TrainingWindows(Dataset):
    def __init__(self, ctx, weak=True):
        self.ctx, self.weak = ctx, weak
        self.indices = ctx['ids']['train']

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = int(self.indices[i]); raw = self.ctx['raw'][j].copy()
        if self.weak:
            raw += np.random.normal(0, .005, raw.shape)
            if np.random.random() < .5:
                for channel in (6, 7, 8):
                    raw[channel] += np.linspace(0, np.random.uniform(-.5, .5), 1024)
            raw *= np.random.uniform(.98, 1.02)
        x = (raw - self.ctx['mean'][:, None]) / self.ctx['scale'][:, None]
        return torch.from_numpy(x.astype(np.float32)), torch.tensor(self.ctx['y'][j]).long()


def loader(ctx, seed, weak=True, batch_size=64):
    return DataLoader(TrainingWindows(ctx, weak), batch_size=batch_size, shuffle=True,
                      num_workers=0, generator=torch.Generator().manual_seed(seed))


class ModalityNoise:
    def __init__(self, seed):
        self.generator = torch.Generator(device='cpu').manual_seed(100000 + seed)

    def __call__(self, x, return_meta=False):
        n, g = len(x), self.generator
        apply = torch.rand(n, generator=g) < .5
        mod = torch.randint(0, 3, (n,), generator=g)
        snr = -10 + 30 * torch.rand(n, generator=g)
        white = torch.randn(n, 3, 1024, generator=g)
        out = x.clone()
        for m in range(3):
            keep = apply & (mod == m); sl = slice(m*3, m*3+3)
            a = x[keep, sl]
            power = a.square().mean((1, 2), keepdim=True).clamp(min=1e-12)
            sigma = (power / (10**(snr[keep, None, None]/10))).sqrt()
            out[keep, sl] = a + white[keep] * sigma
        return (out, apply, mod, snr) if return_meta else out
