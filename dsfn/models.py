"""Residual CNN, GTD and 130-input CRCF used in the revised DSFN.

Definitions are extracted from the frozen revision model sources. Module and
parameter names preserve the six-/three-class checkpoint layout. Initialization
retains the historical constant-column-to-bias mapping, but the final scoring
network has exactly 130 inputs and receives only pooled features, confidence,
and cross-modal consistency. No Transformer or dynamic stability term is used.

Source: basic_models_20261020.py and basic_no_scalar_models_20261020.py;
final 0.1/0.1 objective: uavfd_full_train.py (same manuscript setting).
"""
from collections.abc import Mapping
from contextlib import contextmanager
import copy
import torch
from torch import nn
from torch.nn import functional as F
from . import baseline_models, cnn_dbf

MODS = ('g', 'a', 'v')
BASELINES = ('resnet1d', 'tcn', 'inceptiontime', 'lstm_fcn', 'drsn', 'mra_cnn')
FAMILIES = ('dsfn',) + BASELINES + ('cnn_dbf', 'centaur')
INPUT_CHANNELS, WINDOW_LENGTH, FEATURE_DIM = 9, 1024, 128
TRAINING_SCALES, INFERENCE_SCALES = (1., 1., 1.), (0., .25, 1.)



@contextmanager
def cpu_seed(seed):
    """Isolate CPU initialization without seeding/querying other GPUs."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(int(seed))
        yield

def validate_classes(num_classes):
    if isinstance(num_classes, bool) or int(num_classes) != num_classes or int(num_classes) < 2:
        raise ValueError('num_classes must come from the audited label map and be >=2')
    return int(num_classes)

def validate_input(x):
    if x.ndim != 3 or tuple(x.shape[1:]) != (INPUT_CHANNELS, WINDOW_LENGTH):
        raise ValueError('Expected floating input [B,9,1024] in audited G/A/V order')
    if not torch.is_floating_point(x) or not torch.isfinite(x).all() or len(x) == 0:
        raise ValueError('Inputs must be finite floating point and nonempty')

def infer_availability(*parts):
    if len(parts) != 3 or any(z.ndim != 3 or z.shape[1] != 3 for z in parts):
        raise ValueError('Each modality must be [B,3,time]')
    if any(z.shape != parts[0].shape for z in parts) or any(not torch.isfinite(z).all() for z in parts):
        raise ValueError('Modalities must be aligned and finite')
    return torch.stack([(z != 0).flatten(1).any(1) for z in parts], dim=1)

def assert_no_transformer(model):
    forbidden = (nn.MultiheadAttention, nn.Transformer, nn.TransformerEncoder,
        nn.TransformerDecoder, nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)
    if any(isinstance(module, forbidden) for module in model.modules()):
        raise AssertionError('Transformer/attention is forbidden in this model protocol')

class ResidualDilatedBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2*dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(width), nn.ReLU(),
            nn.Conv1d(width, width, 5, padding=2*dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(width))
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.body(x))

class ResidualDilatedCNN(nn.Module):
    def __init__(self, in_ch=3, width=32):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_ch, width, 3, padding=1, bias=False),
                                  nn.BatchNorm1d(width), nn.ReLU())
        self.blocks = nn.Sequential(*[ResidualDilatedBlock(width, d) for d in (1, 2, 4, 8)])

    def forward(self, x):
        return self.blocks(self.stem(x))

class PatchEmbedding(nn.Module):
    def __init__(self, in_ch=32, embed_dim=128, patch=16):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, embed_dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        return self.conv(x).permute(0, 2, 1)

class ZeroScore(nn.Module):
    def forward(self, x):
        return x.new_zeros((len(x), 1))

class _InitializationFusion(nn.Module):
    def __init__(self, dim=128, num_classes=6, hidden=64, temperature=1.):
        super().__init__()
        self.num_classes = validate_classes(num_classes)
        for name in MODS:
            setattr(self, 'branch_head_' + name, nn.Linear(dim, self.num_classes))
        self.score_mlp = nn.Sequential(nn.Linear(dim+3, hidden), nn.ReLU(inplace=True),
                                       nn.Dropout(.1), nn.Linear(hidden, 1))
        self.temperature = float(temperature)

    def forward(self, g, a, v, probs_tuple=None):
        if probs_tuple is not None and any(z is not None for z in probs_tuple):
            raise ValueError('There is no selector/certainty input in the pure CNN protocol')
        pooled = [z.mean(dim=1) for z in (g, a, v)]
        return self.forward_pooled(pooled)


class CRCFusion(_InitializationFusion):
    def __init__(self, dim=128, num_classes=6, hidden=64, temperature=1.):
        nn.Module.__init__(self)
        self.num_classes = validate_classes(num_classes)
        for name in MODS:
            setattr(self, 'branch_head_' + name, nn.Linear(dim, self.num_classes))
        self.score_mlp = nn.Sequential(nn.Linear(dim + 2, hidden), nn.ReLU(inplace=True),
                                       nn.Dropout(.1), nn.Linear(hidden, 1))
        self.temperature = float(temperature)

    def forward_pooled(self,pooled):
        active=[i for i,value in enumerate(pooled) if value is not None]
        if not active:raise ValueError('No available modality')
        first=pooled[active[0]];n=len(first)
        branch={i:getattr(self,'branch_head_'+MODS[i])(pooled[i]) for i in active}
        confidence={i:F.softmax(branch[i],dim=1).max(dim=1,keepdim=True)[0] for i in active}
        if len(active)==3:
            ga=F.cosine_similarity(pooled[0],pooled[1],dim=1,eps=1e-8).unsqueeze(1)
            gv=F.cosine_similarity(pooled[0],pooled[2],dim=1,eps=1e-8).unsqueeze(1)
            av=F.cosine_similarity(pooled[1],pooled[2],dim=1,eps=1e-8).unsqueeze(1)
            consistency={0:.5*(.5*(ga+1.)+.5*(gv+1.)),
                1:.5*(.5*(ga+1.)+.5*(av+1.)),2:.5*(.5*(gv+1.)+.5*(av+1.))}
        else:
            consistency={}
            for i in active:
                others=[j for j in active if j!=i]
                consistency[i]=sum((F.cosine_similarity(pooled[i],pooled[j],dim=1,eps=1e-8)[:,None]+1)*.5
                    for j in others)/len(others) if others else torch.ones_like(confidence[i])
        scores=[self.score_mlp(torch.cat([pooled[i],confidence[i],consistency[i]],dim=1)) for i in active]
        aa=F.softmax(torch.cat(scores,dim=1)/self.temperature,dim=1)
        if len(active)==3:
            fused=aa[:,0:1]*pooled[0]+aa[:,1:2]*pooled[1]+aa[:,2:3]*pooled[2]
        else:
            fused=sum(aa[:,j:j+1]*pooled[i] for j,i in enumerate(active))
        alpha,confout,consout=[first.new_zeros((n,3)) for _ in range(3)]
        branch_logits={name:first.new_zeros((n,self.num_classes)) for name in MODS}
        for j,i in enumerate(active):
            alpha[:,i]=aa[:,j]
            confout[:,i],consout[:,i]=confidence[i][:,0],consistency[i][:,0]
            branch_logits[MODS[i]]=branch[i]
        return fused,alpha,branch_logits,dict(confidence=confout,consistency=consout,fusion_weights=alpha)

def _add_frontends(model, seed):
    with cpu_seed(500000 + int(seed)):
        for name in MODS:
            setattr(model, 'pre_' + name, ResidualDilatedCNN(3, 32))
            setattr(model, 'embed_' + name, PatchEmbedding(32, 128, 32 if name == 'v' else 16))

class Tokens(list):
    def __init__(self, values, probs=None):
        super().__init__(values)
        self.probs = probs

class _InitializationCNN(nn.Module):
    def __init__(self, family, seed, num_classes=6):
        super().__init__()
        if family not in ('full', 'uniform'):
            raise ValueError(family)
        self.family, self.num_classes = family, validate_classes(num_classes)
        _add_frontends(self, seed)
        for name in MODS:
            setattr(self, 'amcf_' + name, nn.Identity())
            setattr(self, 'denoise_' + name, nn.Identity())
        with cpu_seed(620000 + int(seed)):
            self.rcd_fusion = _InitializationFusion(128, self.num_classes, 64, 1.)
            self.simple_head = nn.Linear(128, self.num_classes)
        if family == 'uniform':
            self.rcd_fusion.score_mlp = ZeroScore()

    def tokens(self, x):
        validate_input(x)
        return Tokens([getattr(self, 'embed_' + name)(getattr(self, 'pre_' + name)(z))
                       for name, z in zip(MODS, x.split(3, 1))])

    def forward(self, x, a=None, v=None, **kwargs):
        if a is not None or v is not None:
            if a is None or v is None:
                raise ValueError('Supply either x or all three modalities')
            x = torch.cat([x, a, v], dim=1)
        tokens = self.tokens(x)
        fused, alpha, branch, info = self.rcd_fusion(*tokens, probs_tuple=None)
        return dict(logits=self.simple_head(fused), branch_logits=branch,
                    fusion_weights=alpha, rcd_info=info, tokens=tokens)

    def available_detail(self, x, valid):
        pooled = []
        for i, name in enumerate(MODS):
            pooled.append(getattr(self, 'embed_' + name)(getattr(self, 'pre_' + name)(x[:, i*3:i*3+3])).mean(1)
                          if bool(valid[i]) else None)
        fused, alpha, branch, info = self.rcd_fusion.forward_pooled(pooled)
        return self.simple_head(fused), alpha, branch, info

class DSFNBase(_InitializationCNN):
    """Trainable base encoder/fusion model, before the separate GTD stage."""
    def __init__(self, seed=42, num_classes=6):
        super().__init__('full', seed, num_classes)
        first = self.rcd_fusion.score_mlp[0]
        weight, bias = first.weight.detach().clone(), first.bias.detach().clone()
        self.rcd_fusion.__class__ = CRCFusion
        with cpu_seed(920000 + int(seed)):
            replacement = nn.Linear(130, 64)
        with torch.no_grad():
            replacement.weight.copy_(weight[:, :130])
            replacement.bias.copy_(bias + weight[:, 130])
        self.rcd_fusion.score_mlp[0] = replacement
        assert_no_transformer(self)


class CNNDBF(nn.Module):
    def __init__(self, seed, num_classes=6):
        super().__init__()
        self.family, self.num_classes = 'cnn_dbf', validate_classes(num_classes)
        _add_frontends(self, seed)
        module = cnn_dbf
        with cpu_seed(int(module.HEAD_SEED_OFFSET) + int(seed)):
            self.evidence_heads = nn.ModuleList([nn.Linear(128, self.num_classes) for _ in MODS])
        self.adaptation_metadata = copy.deepcopy(module.adaptation_metadata)
        self.adaptation_metadata.update(seed=int(seed), num_classes=self.num_classes,
            frontend_initialization='fresh deterministic random epoch0, CPU seed500000+seed',
            external_pretraining=False, initialization_head_reused=False)

    def forward(self, x):
        validate_input(x)
        module = cnn_dbf
        pooled = [getattr(self, 'embed_' + name)(getattr(self, 'pre_' + name)(z)).mean(1)
                  for name, z in zip(MODS, x.split(3, 1))]
        evidence = torch.stack([module.capped_exponential(head(value))
                                for head, value in zip(self.evidence_heads, pooled)], dim=1)
        outputs = module.discounted_belief_fusion(evidence)
        outputs['features'] = torch.stack(pooled, dim=1)
        return outputs

def logits_from_output(outputs):
    logits = outputs['logits'] if isinstance(outputs, Mapping) else outputs
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise ValueError('Expected finite [B,K] logits/log probabilities')
    return logits

def _weights(class_weights, logits):
    weights = torch.as_tensor(class_weights, device=logits.device, dtype=logits.dtype)
    if weights.shape != (logits.shape[1],) or not torch.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError('class_weights must contain K finite positive training-only values')
    return weights

class TokenDenoiseBlock(nn.Module):
    def __init__(self, dim=128, kernel_size=5, reduction=4, dropout=.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                               padding=kernel_size//2, groups=dim, bias=False)
        self.pwconv = nn.Conv1d(dim, dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(dim)
        hidden = max(dim//reduction, 16)
        self.gate = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                   nn.Linear(hidden, dim), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

class IdentityGTD(nn.Module):
    def __init__(self, cap_ratio=None):
        super().__init__()
        self.core = TokenDenoiseBlock(128)
        self.cap_ratio = cap_ratio
        self.output = nn.Linear(128, 128, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, x):
        z = self.core.norm(x)
        residual = self.core.bn(self.core.pwconv(self.core.dwconv(z.transpose(1, 2)))).transpose(1, 2)
        gate = self.core.gate(z.mean(1))[:, None, :]
        delta = self.core.dropout(self.output(gate*residual))
        if self.cap_ratio is not None:
            input_rms = x.square().mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()
            delta_rms = delta.square().mean((1, 2), keepdim=True).clamp_min(1e-12).sqrt()
            factor = (self.cap_ratio*input_rms/delta_rms).clamp(max=1.)
            delta = delta*factor
        return x + delta

class ControlledModel(nn.Module):
    def __init__(self, base, seed, class_weights=None, cap_ratio=None):
        super().__init__()
        if getattr(base, 'family', None) not in ('full', 'uniform'):
            raise ValueError('Only full/uniform have an additional GTD stage')
        self.family, self.arm, self.num_classes = base.family, base.family, base.num_classes
        self.side, self.serial = False, False
        self.frozen = base.eval()
        for parameter in self.frozen.parameters():
            parameter.requires_grad_(False)
        with cpu_seed(910000 + int(seed)):
            self.adapters = nn.ModuleList([IdentityGTD(cap_ratio) for _ in MODS])
        self.adapters.to(next(base.parameters()).device)
        self.cfg = dict(side_epochs=0,
                        class_weights=None if class_weights is None else torch.as_tensor(class_weights).tolist())
        if class_weights is not None:
            _weights(class_weights, next(base.parameters()).new_zeros((1, self.num_classes)))
        assert_no_transformer(self)
        self.eval()

    @property
    def base(self):
        return self.frozen

    def train(self, mode=True):
        super().train(mode)
        self.frozen.eval()
        return self

    @torch.no_grad()
    def features(self, x):
        return self.frozen.tokens(x)

    def patches(self, x):
        return self.features(x)

    def classify_detail(self, tokens, use_gtd=True, scales=INFERENCE_SCALES):
        if len(tokens) != 3 or len(scales) != 3:
            raise ValueError('Expected three modality tokens/scales')
        z = []
        for i, h in enumerate(tokens):
            scale = scales[i] if use_gtd else 0.
            z.append(h if scale == 0 else self.adapters[i](h) if scale == 1 else h + scale*(self.adapters[i](h)-h))
        fused, alpha, branch, info = self.base.rcd_fusion(*z, probs_tuple=None)
        return self.base.simple_head(fused), Tokens(z), alpha, branch, info

    def classify(self, tokens, use_gtd=True, mode='base', scales=INFERENCE_SCALES):
        if mode != 'base':
            raise ValueError('There is no side/dynamic/Transformer mode')
        logits, z, alpha, _, _ = self.classify_detail(tokens, use_gtd, scales)
        return logits, z, alpha

    @torch.no_grad()
    def infer_detail(self, x, use_gtd=True):
        if self.training:
            raise RuntimeError('Call eval() before inference to preserve BN state')
        validate_input(x)
        available = infer_availability(*x.split(3, 1))
        if not available.any(1).all():
            raise ValueError('No available modality: do not issue a fault prediction')
        present = available.all(1)
        logits = x.new_zeros((len(x), self.num_classes))
        evidence = {key: x.new_zeros((len(x), 3)) for key in ('r', 'c', 'alpha')}
        evidence['branch_logits'] = x.new_zeros((len(x), 3, self.num_classes))
        evidence['availability'] = available
        if present.any():
            ls, _, alpha, branch, info = self.classify_detail(self.features(x[present]), use_gtd)
            logits[present] = ls
            evidence['r'][present], evidence['c'][present], evidence['alpha'][present] = info['confidence'], info['consistency'], alpha
            evidence['branch_logits'][present] = torch.stack([branch[name] for name in MODS], dim=1)
        codes = (available.long()*torch.tensor([1, 2, 4], device=x.device)).sum(1)
        for code in codes[~present].unique(sorted=True):
            ids = (codes == code).nonzero(as_tuple=True)[0]
            ls, alpha, branch, info = self.base.available_detail(x[ids], available[ids[0]])
            logits[ids] = ls
            evidence['r'][ids], evidence['c'][ids], evidence['alpha'][ids] = info['confidence'], info['consistency'], alpha
            evidence['branch_logits'][ids] = torch.stack([branch[name] for name in MODS], dim=1)
        assert (evidence['alpha'][~available] == 0).all()
        assert torch.allclose(evidence['alpha'].sum(1), torch.ones(len(x), device=x.device), atol=1e-6)
        return logits, evidence

def gtd_loss(model, clean, noisy, labels, class_weights, corruption):
    """Selected revision split loss, with its affected-pair normalization intact."""
    cp, np_ = model.features(clean), model.features(noisy)
    joined = Tokens([torch.cat([a, b]) for a, b in zip(cp, np_)])
    logits, z, _ = model.classify(joined, True, 'base', scales=TRAINING_SCALES)
    weights, n = _weights(class_weights, logits), len(labels)
    ce = .5*(F.cross_entropy(logits[:n], labels, weight=weights) +
              F.cross_entropy(logits[n:], labels, weight=weights))
    ec = torch.stack([(a[:n]-b).square().mean((1, 2))/b.square().mean((1, 2)).clamp_min(1e-6)
                      for a, b in zip(z, cp)], dim=1)
    en = torch.stack([(a[n:]-b).square().mean((1, 2))/b.square().mean((1, 2)).clamp_min(1e-6)
                      for a, b in zip(z, cp)], dim=1)
    apply, modality = [value.to(logits.device) for value in corruption]
    if apply.shape != (n,) or modality.shape != (n,) or apply.dtype != torch.bool:
        raise ValueError('corruption must be (bool apply[B], modality[B])')
    mask = torch.zeros_like(en, dtype=torch.bool)
    ids = apply.nonzero(as_tuple=True)[0]
    mask[ids, modality[ids]] = True
    denoise = en[mask].mean() if mask.any() else en.sum()*0.
    identity = .5*(ec.mean() + en[~mask].mean())
    total = ce + denoise + identity
    return total, dict(ce=float(ce.detach()), denoise=float(denoise.detach()),
        identity=float(identity.detach()), clean_identity=float(ec.mean().detach()),
        untouched_identity=float(en[~mask].mean().detach()), affected_pairs=int(mask.sum()))

def create_ema(model):
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    return ema

@torch.no_grad()
def update_base_ema(ema, model, decay=.995):
    # Preserve the base recipe: floating buffers also use EMA; integer buffers copy.
    source = model.state_dict()
    for key, value in ema.state_dict().items():
        current = source[key].detach()
        if torch.is_floating_point(value):
            value.mul_(decay).add_(current, alpha=1.-decay)
        else:
            value.copy_(current)

@torch.no_grad()
def update_gtd_ema(ema_adapters, adapters, decay=.995):
    # Preserve the GTD recipe: EMA parameters, direct copy of BN buffers.
    for a, b in zip(ema_adapters.parameters(), adapters.parameters()):
        a.mul_(decay).add_(b, alpha=1.-decay)
    for a, b in zip(ema_adapters.buffers(), adapters.buffers()):
        a.copy_(b)

class DSFN(ControlledModel):
    """GTD wrapper; use base training first, then freeze it with wrap_gtd()."""
    def forward(self, x):
        if self.training:
            return self.classify_detail(self.features(x), scales=TRAINING_SCALES)[0]
        return self.infer_detail(x)[0]

    def forward_with_evidence(self, x, use_gtd=True):
        return self.infer_detail(x, use_gtd=use_gtd)


def build_base_model(family, num_classes, seed=42):
    """Fresh CPU construction; no files, fitted weights, or data are loaded."""
    family = family.lower()
    classes = validate_classes(num_classes)
    if family in ('dsfn', 'full'):
        return DSFNBase(seed=seed, num_classes=classes)
    if family == 'cnn_dbf':
        return CNNDBF(seed, classes)
    if family == 'centaur':
        from .centaur import Centaur, build_centaur
        return Centaur(*build_centaur(classes, seed))
    if family not in BASELINES:
        raise ValueError(family)
    with cpu_seed(seed):
        return baseline_models.build_baseline(family, num_classes=classes)


def wrap_gtd(base, seed=42, class_weights=None, cap_ratio=None):
    return DSFN(base, seed, class_weights, cap_ratio)


def build_model(family, num_classes, seed=42):
    model = build_base_model(family, num_classes, seed)
    return wrap_gtd(model, seed) if family.lower() in ('dsfn', 'full') else model


def base_loss(family, outputs, labels, class_weights, epoch=1,
              return_details=False, auxiliary_weight=.1, balance_weight=.1):
    """Final paper objective; coefficients apply only to the DSFN base stage."""
    logits = logits_from_output(outputs)
    weights = _weights(class_weights, logits)
    if family == 'cnn_dbf':
        return cnn_dbf.loss(outputs, labels, weights, epoch, return_details)
    main = F.cross_entropy(logits, labels, weight=weights)
    details = dict(main_ce=main.detach())
    total = main
    if family in ('dsfn', 'full', 'uniform'):
        auxiliary = sum(F.cross_entropy(outputs['branch_logits'][m], labels, weight=weights)
                        for m in MODS) / 3.
        alpha = outputs['fusion_weights'].mean(0)
        balance = F.mse_loss(alpha, torch.ones_like(alpha) / 3.)
        total = main + auxiliary_weight * auxiliary + balance_weight * balance
        details.update(auxiliary_ce=auxiliary.detach(), balance=balance.detach())
    return (total, details) if return_details else total


extract_logits = logits_from_output
