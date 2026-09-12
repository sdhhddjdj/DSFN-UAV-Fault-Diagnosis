"""Shared helpers for the test-time robustness suites (Tables 4 and 5).

Both suites evaluate already trained checkpoints; no model is retrained. The
inputs are normalised with the training-split statistics, exactly as in
``train_dsfn.py`` and ``run_baseline_suite.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "paper_experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import train_dsfn as dsfn  # noqa: E402
from paper_protocol import DEFAULT_DATA_ROOT, DEFAULT_SPLIT_DIR, validate_paper_data
from baseline_models import build_baseline  # noqa: E402
from experiment_matrix import (  # noqa: E402
    DEFAULT_SEEDS,
    HORIZONTAL_BASELINES,
    MANUSCRIPT_MODEL,
    PAPER_LOW_SNR_DB,
    PAPER_MASK_MODALITIES,
    PAPER_NOISE_MODALITIES,
)

MODALITY_SLICES = {"gyro": (0, 3), "accel": (3, 6), "vel": (6, 9)}
PROPOSED_ALIASES = {"dsfn", "proposed", "ours"}
DEFAULT_MODELS = "dsfn," + ",".join(
    item["name"] for item in HORIZONTAL_BASELINES if not item.get("optional")
)


def is_dsfn(name: str) -> bool:
    return name.lower() in PROPOSED_ALIASES or name in dsfn.ABLATION_CONFIGS


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item) for item in parse_csv(value)] if value else DEFAULT_SEEDS
    return seeds or DEFAULT_SEEDS


def parse_models(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return ["dsfn"] + [item["name"] for item in HORIZONTAL_BASELINES]
    return parse_csv(value) or parse_csv(DEFAULT_MODELS)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint-root",
        default="runs",
        help=(
            "Directory holding the trained checkpoints. DSFN runs are read from "
            "<root>/<DSFN_EXP_NAME>_seed<seed>/best_model.pth and baselines from "
            "<root>/<model>/seed_<seed>/best_model.pth."
        ),
    )
    parser.add_argument("--proposed-exp-name", default=MANUSCRIPT_MODEL)
    parser.add_argument("--models", default=DEFAULT_MODELS)
    parser.add_argument(
        "--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS)
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument(
        "--split-dir", default=str(DEFAULT_SPLIT_DIR)
    )
    parser.add_argument("--output-dir", default="results/robustness")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )


def stable_seed(*parts: object) -> int:
    digest = hashlib.md5("|".join(str(part) for part in parts).encode("utf-8"))
    return int(digest.hexdigest()[:8], 16)


def seed_label(name: str) -> str:
    """Canonical model label used for the noise seed.

    The manuscript tables were produced with the label ``proposed`` for DSFN;
    this preserves the recorded seed mapping across repeated evaluations.
    """
    return "proposed" if name.lower() in PROPOSED_ALIASES else name


def set_eval_config(args: argparse.Namespace) -> None:
    dsfn.CONFIG["data_root"] = args.data_root
    dsfn.CONFIG["split_dir"] = args.split_dir
    dsfn.CONFIG["batch_size"] = args.batch_size
    dsfn.CONFIG["use_aug"] = False
    dsfn.CONFIG["use_channel_norm"] = True


def make_test_loader(args: argparse.Namespace) -> DataLoader:
    validate_paper_data(args.data_root, args.split_dir)
    set_eval_config(args)
    full_dataset = dsfn.SensorDataset(args.data_root, is_train=False)
    train_indices = np.load(Path(args.split_dir) / "train_indices.npy")
    test_indices = np.load(Path(args.split_dir) / "test_indices.npy")
    channel_mean, channel_std = dsfn.compute_channel_stats(
        full_dataset.data, train_indices
    )
    test_set = torch.utils.data.Subset(
        dsfn.SensorDataset(
            args.data_root,
            is_train=False,
            channel_mean=channel_mean,
            channel_std=channel_std,
        ),
        test_indices,
    )
    return DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


def checkpoint_path(args: argparse.Namespace, name: str, seed: int) -> Path:
    root = Path(args.checkpoint_root)
    if is_dsfn(name):
        experiment = name if name in dsfn.ABLATION_CONFIGS else args.proposed_exp_name
        return root / f"{experiment}_seed{seed}" / "best_model.pth"
    return root / name / f"seed_{seed}" / "best_model.pth"


def load_model(args: argparse.Namespace, name: str, seed: int) -> torch.nn.Module:
    if is_dsfn(name):
        experiment = name if name in dsfn.ABLATION_CONFIGS else args.proposed_exp_name
        dsfn.apply_ablation_config(experiment)
        set_eval_config(args)
        dsfn.CONFIG["eval_soft_select"] = True
        dsfn.CONFIG["eval_force_full"] = True
        model = dsfn.SignalTransformerModel()
    else:
        model = build_baseline(name, num_classes=args.num_classes)

    path = checkpoint_path(args, name, seed)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=args.device)
    missing, unexpected, skipped = dsfn.load_state_dict_flexible(model, state)
    if missing or skipped:
        raise RuntimeError(
            f"{path} does not match the model definition "
            f"(missing={len(missing)}, skipped={len(skipped)})."
        )
    if unexpected:
        print(
            f"  note: ignored {len(unexpected)} checkpoint keys that are not "
            "part of the released model.",
            flush=True,
        )
    return model.to(args.device).eval()


def predict(
    model: torch.nn.Module, name: str, g: torch.Tensor, a: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    if is_dsfn(name):
        # Manuscript protocol: every candidate dynamic layer is executed and the
        # outputs are aggregated with the selector probabilities.
        return model(
            g,
            a,
            v,
            gate_threshold=dsfn.CONFIG["gate_threshold"],
            force_full=True,
            collect_intermediates=False,
        )[0]
    # The comparison models take the nine channels as a single input tensor.
    return model(torch.cat([g, a, v], dim=1))


def add_modality_noise(
    x: torch.Tensor, snr_db: float, generator: torch.Generator
) -> torch.Tensor:
    """Add Gaussian noise at the given SNR, measured against the signal power."""
    power = x.square().mean(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise = torch.randn(
        x.shape, generator=generator, device=x.device, dtype=x.dtype
    )
    return x + noise * noise_power.sqrt()


def corrupt_modalities(
    g: torch.Tensor,
    a: torch.Tensor,
    v: torch.Tensor,
    condition: str,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply one test-time condition, e.g. ``clean``, ``mask:gyro``, ``noise:vel:0``."""
    if condition == "clean":
        return g, a, v
    kind, modality, *value = condition.split(":")
    if modality not in MODALITY_SLICES:
        raise ValueError(f"Unknown modality in condition: {condition}")
    tensors = {"gyro": g.clone(), "accel": a.clone(), "vel": v.clone()}
    if kind == "mask":
        tensors[modality].zero_()
    elif kind == "noise":
        if generator is None:
            raise ValueError("A generator is required for noise conditions.")
        tensors[modality] = add_modality_noise(
            tensors[modality], float(value[0]), generator
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return tensors["gyro"], tensors["accel"], tensors["vel"]


@torch.no_grad()
def evaluate_condition(
    model: torch.nn.Module,
    name: str,
    loader: DataLoader,
    device: str,
    condition: str,
    seed: int,
) -> float:
    """Return the accuracy (in percent) of one model on one condition."""
    label = seed_label(name)
    correct = total = 0
    for batch_index, (g, a, v, target) in enumerate(loader):
        g, a, v = g.to(device), a.to(device), v.to(device)
        generator = None
        if condition != "clean":
            kind, modality, *value = condition.split(":")
            if kind == "noise":
                # Same seed derivation as the manuscript tables: one fixed
                # realisation per model, seed, modality, SNR and batch.
                generator = torch.Generator(device=device)
                generator.manual_seed(
                    stable_seed(label, seed, modality, float(value[0]), batch_index)
                )
        g, a, v = corrupt_modalities(g, a, v, condition, generator)
        logits = predict(model, name, g, a, v)
        correct += (logits.argmax(dim=1).cpu() == target).sum().item()
        total += target.numel()
    return 100.0 * correct / max(total, 1)


def noise_conditions(
    modalities: list[str] | None = None, snrs: list[float] | None = None
) -> list[str]:
    modalities = modalities or PAPER_NOISE_MODALITIES
    snrs = PAPER_LOW_SNR_DB if snrs is None else snrs
    return [f"noise:{m}:{snr:g}" for m in modalities for snr in snrs]


def mask_conditions(modalities: list[str] | None = None) -> list[str]:
    return [f"mask:{m}" for m in (modalities or PAPER_MASK_MODALITIES)]
