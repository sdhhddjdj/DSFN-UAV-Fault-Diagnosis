"""CNN-DBF numerical fusion and loss used in the revision experiments.

Adapted from Bezirganyan et al., AISTATS 2025, and
https://github.com/bezirganyan/DBF_uncertainty (GPL-3.0).
See ../third_party/README.md. Capped-exponential evidence, double-precision
fusion/loss, and class-weighted evidential cross-entropy match the experiment.
No checkpoint loading, network request, or training runs at import time.
"""
import math
import torch
from torch.nn import functional as F

MODALITIES = ("g", "a", "v")
FEATURE_DIM = 128
DISCOUNT_LAMBDA = 1.0
CONFLICT_GAMMA = 0.5
ANNEALING_EPOCHS = 30
EVIDENCE_CAP = 1.0e13
EPSILON = 1.0e-6
HEAD_SEED_OFFSET = 720000
adaptation_metadata = {
    "name": "CNN-DBF (adapted)",
    "paper": "Multimodal Learning with Uncertainty Quantification based on Discounted Belief Fusion",
    "venue": "AISTATS 2025; PMLR 258:3142-3150",
    "paper_url": "https://proceedings.mlr.press/v258/bezirganyan25a.html",
    "paper_method_url": "https://arxiv.org/html/2412.18024v2",
    "official_repository": "https://github.com/bezirganyan/DBF_uncertainty",
    "upstream_license": "GPL-3.0; see official repository LICENSE",
    "source_urls": {
        "model": "https://raw.githubusercontent.com/bezirganyan/DBF_uncertainty/main/model.py",
        "loss": "https://raw.githubusercontent.com/bezirganyan/DBF_uncertainty/main/loss_function.py",
        "training": "https://raw.githubusercontent.com/bezirganyan/DBF_uncertainty/main/main.py",
    },
    "source_read_date": "2026-10-19",
    "source_commit": None,
    "source_commit_limitation": "GitHub anonymous API returned rate-limit HTTP 403; main URLs are not immutable pins",
    "input": "normalized B x 9 x 1024; gyro, accel, velocity in three consecutive 3-channel groups",
    "frontend": "same ResidualDilatedCNN width32, dilations1/2/4/8, PatchEmbedding128, token mean pooling",
    "patch_sizes": [16, 16, 32],
    "frontend_initialization": "only pre_{g,a,v}.* and embed_{g,a,v}.* tensors from archived untrained initial_model.pth",
    "head": "three independent single Linear(128, num_classes) evidence heads; no hidden layer",
    "head_initialization": "isolated CPU torch RNG stream seeded 720000 + seed; independent of frontend initialization draws",
    "external_pretraining": False,
    "transformer_attention_gtd": False,
    "discount_lambda": DISCOUNT_LAMBDA,
    "conflict_gamma": CONFLICT_GAMMA,
    "annealing_epochs": ANNEALING_EPOCHS,
    "fixed_hyperparameter_source": "paper Table4 / official main.py CalTech gamma=.5 and annealing=30; chosen before local outcomes",
    "evidence_activation": "paper capped exponential C/(1+C*exp(-h)), C=1e13; exp(log(C)-softplus(log(C)-h, threshold=40)) in float64 avoids large-positive cancellation in the upstream logaddexp arrangement; not official script default softplus evidence activation",
    "fusion_numeric_policy": "float64; preserve official dc+1e-6 before agreement and u_discounted+1e-6 before evidence recovery; clamp agreement base to [0,1] for roundoff/extreme dc",
    "loss_numeric_policy": "EDL digamma/lgamma and conflict in float64 to reduce cancellation from evidence up to 1e13; gradients propagate to float32 CNN/heads",
    "prediction_dtype": "float32 normalized log probabilities, usable as logits in shared inference/metric code",
    "loss": "mean of fused+three branch EDL losses plus .5 * mean_sample(sum_directed_dc/(M-1))",
    "class_weights": "each head EDL CE uses sum(w_y*CE)/sum(w_y), matching weighted CrossEntropyLoss batch normalization; KL and conflict remain unweighted",
    "missing_input": "same normalized all-zero modality input as horizontal baselines; no availability masking, branch removal, or imputation inside DBF",
    "training_contract": "80 epochs only; same data/splits/normalization/augmentation/noise bank/AdamW3e-4/wd1e-4/cosine3e-6/EMA.995/batch64/checkpoint rule as the other baselines; no GTD stage",
    "comparison_limit": "common-protocol CNN adaptation, not original benchmark reproduction or best-tuned DBF; full model has its separately disclosed GTD training budget",
    "static_verification_only": True,
}

# Training budgets belong to the caller; the final baselines use 120 epochs.
adaptation_metadata['training_contract'] = '120 classifier epochs in the revised comparison; no GTD stage'
adaptation_metadata['frontend_initialization'] = 'shared deterministic random CNN/patch initialization; optional archived epoch-zero loading is handled by the caller'
adaptation_metadata.pop('static_verification_only', None)

def capped_exponential(h):
    """Paper activation, evaluated in float64 without exp(-h) overflow."""
    h = h.to(torch.float64)
    log_cap = h.new_tensor(math.log(EVIDENCE_CAP))
    # Float64 threshold 40 also avoids the default threshold=20 approximation
    # error around h=log(C)-20 when comparing to the paper's exact formula.
    return torch.exp(log_cap - F.softplus(log_cap - h, threshold=40.0))

def discounted_belief_fusion(evidence):
    """Vectorized official 'dbf' path; evidence has shape B x M x K.

    The only extra protection is clamping the agreement base to [0,1]. This
    preserves the official computation on its normal domain, while preventing
    negative masses if dc+epsilon reaches above one. It is disclosed rather
    than called an exact upstream replay. The two epsilons are not removed.
    """
    evidence = evidence.to(torch.float64)
    if evidence.ndim != 3 or evidence.shape[1] != 3:
        raise ValueError("Expected evidence B x 3 x K")
    classes = evidence.shape[-1]
    alpha = evidence + 1.0
    strength = alpha.sum(dim=-1, keepdim=True)
    belief = evidence / strength
    probabilities = alpha / strength
    uncertainty = classes / strength
    probability_distance = (
        probabilities.unsqueeze(2) - probabilities.unsqueeze(1)
    ).abs().sum(dim=-1) / 2.0
    confidence = 1.0 - uncertainty.squeeze(-1)
    degree_of_conflict = probability_distance * confidence.unsqueeze(2) * confidence.unsqueeze(1)
    agreement_base = 1.0 - (degree_of_conflict + EPSILON).pow(DISCOUNT_LAMBDA)
    agreement = agreement_base.clamp(min=0.0, max=1.0).pow(1.0 / DISCOUNT_LAMBDA)
    discount = agreement.prod(dim=-1, keepdim=True)
    discounted_belief = belief * discount
    discounted_uncertainty = uncertainty * discount + 1.0 - discount
    discounted_evidence = classes * discounted_belief / (discounted_uncertainty + EPSILON)
    fused_evidence = discounted_evidence.mean(dim=1)
    fused_alpha = fused_evidence + 1.0
    log_alpha = fused_alpha.log()
    log_probabilities = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
    return {
        "logits": log_probabilities.to(torch.float32),
        "log_probs": log_probabilities.to(torch.float32),
        "probabilities": log_probabilities.exp().to(torch.float32),
        "evidence": evidence,
        "alpha": alpha,
        "uncertainty": uncertainty.squeeze(-1),
        "branch_probabilities": probabilities,
        "degree_of_conflict": degree_of_conflict,
        "discount": discount.squeeze(-1),
        "discounted_uncertainty": discounted_uncertainty.squeeze(-1),
        "fused_evidence": fused_evidence,
        "fused_alpha": fused_alpha,
        "fused_uncertainty": classes / fused_alpha.sum(dim=-1),
    }

def _edl_head_loss(alpha, labels, class_weights, annealing):
    """One head: class-weighted EDL CE plus unweighted mean Dirichlet KL."""
    alpha = alpha.to(torch.float64)
    classes = alpha.shape[-1]
    strength = alpha.sum(dim=-1)
    correct_alpha = alpha.gather(1, labels[:, None]).squeeze(1)
    ce = torch.digamma(strength) - torch.digamma(correct_alpha)
    if class_weights is None:
        ce_mean = ce.mean()
    else:
        weights = torch.as_tensor(class_weights, device=alpha.device, dtype=torch.float64)
        if weights.shape != (classes,):
            raise ValueError("class_weights must have one value per class")
        sample_weights = weights[labels]
        ce_mean = (sample_weights * ce).sum() / sample_weights.sum()
    one_hot = F.one_hot(labels, num_classes=classes).to(torch.float64)
    modified_alpha = (alpha - 1.0) * (1.0 - one_hot) + 1.0
    modified_strength = modified_alpha.sum(dim=-1, keepdim=True)
    kl = (
        torch.lgamma(modified_strength).squeeze(-1)
        - torch.lgamma(modified_alpha).sum(dim=-1)
        - torch.lgamma(alpha.new_tensor(float(classes)))
        + ((modified_alpha - 1.0) * (
            torch.digamma(modified_alpha) - torch.digamma(modified_strength)
        )).sum(dim=-1)
    )
    kl_mean = kl.mean()
    return ce_mean + annealing * kl_mean, ce_mean, kl_mean

def loss(model_outputs, y, class_weights, epoch, return_details=False):
    """Official multihead EDL/conflict objective with disclosed CE weighting.

    epoch is one-based. The conflict is a sum over *directed* pairs divided by
    M-1 and then averaged over samples, as in get_dc_loss, not a pairwise mean.
    With unit class weights this is the official loss normalization (subject to
    the declared float64 policy and capped-exponential activation).
    """
    alpha = model_outputs["alpha"].to(torch.float64)
    if alpha.ndim != 3 or alpha.shape[1] != 3:
        raise ValueError("Expected branch alpha B x 3 x K")
    labels = y.to(device=alpha.device, dtype=torch.long)
    if labels.ndim != 1 or labels.shape[0] != alpha.shape[0]:
        raise ValueError("Expected one class label per sample")
    if int(epoch) < 1:
        raise ValueError("epoch must be one-based")
    annealing = min(1.0, float(epoch) / ANNEALING_EPOCHS)
    heads = [model_outputs["fused_alpha"]] + list(alpha.unbind(dim=1))
    terms = [_edl_head_loss(a, labels, class_weights, annealing) for a in heads]
    edl = torch.stack([term[0] for term in terms]).mean()
    directed_conflict = model_outputs["degree_of_conflict"].to(torch.float64)
    conflict = (directed_conflict.sum(dim=(1, 2)) / (alpha.shape[1] - 1)).mean()
    total = edl + CONFLICT_GAMMA * conflict
    if not return_details:
        return total
    return total, {
        "edl": edl.detach(),
        "edl_ce": torch.stack([term[1] for term in terms]).mean().detach(),
        "edl_kl": torch.stack([term[2] for term in terms]).mean().detach(),
        "annealing": annealing,
        "directed_conflict": conflict.detach(),
        "weighted_conflict": (CONFLICT_GAMMA * conflict).detach(),
    }
