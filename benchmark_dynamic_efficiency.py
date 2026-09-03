from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

import train_dsfn as dsfn


VARIANTS = {
    "fixed_3_layers": {
        "use_outer_gate": False,
        "use_sgla": False,
        "eval_soft_select": False,
        "gate_threshold": 2.0,
        "force_full": False,
    },
    "fixed_5_layers": {
        "use_outer_gate": False,
        "use_sgla": False,
        "eval_soft_select": False,
        "gate_threshold": -1.0,
        "force_full": True,
    },
    "outer_gate_only": {
        "use_outer_gate": True,
        "use_sgla": False,
        "eval_soft_select": False,
        "gate_threshold": 0.25,
        "force_full": False,
    },
    "dynamic_exit_only": {
        "use_outer_gate": False,
        "use_sgla": True,
        "eval_soft_select": False,
        "gate_threshold": -1.0,
        "force_full": False,
    },
    "full_dynamic_selector": {
        "use_outer_gate": True,
        "use_sgla": True,
        "eval_soft_select": False,
        "gate_threshold": 0.25,
        "force_full": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified DSFN depth/latency/robustness benchmark.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu-warmup", type=int, default=30)
    parser.add_argument("--gpu-iters", type=int, default=200)
    parser.add_argument("--cpu-warmup", type=int, default=3)
    parser.add_argument("--cpu-iters", type=int, default=10)
    parser.add_argument("--robust", action="store_true")
    return parser.parse_args()


def configure_model() -> None:
    dsfn.apply_ablation_config("Denoise_RCDFusion")
    dsfn.CONFIG.update(
        {
            "use_positional_encoding": True,
            "position_dropout": 0.0,
            "transformer_depth": 5,
            "transformer_split_layer": 3,
            "use_cheap_exit": True,
            "use_shallow_exit": False,
            "use_global_branch": False,
            "use_aug": False,
            "use_channel_norm": True,
        }
    )


def set_variant(name: str) -> dict[str, object]:
    variant = VARIANTS[name]
    for key in ("use_outer_gate", "use_sgla", "eval_soft_select"):
        dsfn.CONFIG[key] = variant[key]
    return variant


def make_test_loader(args: argparse.Namespace) -> DataLoader:
    dataset = dsfn.SensorDataset(str(args.data_root), is_train=False)
    train_indices = np.load(args.split_dir / "train_indices.npy")
    test_indices = np.load(args.split_dir / "test_indices.npy")
    mean, std = dsfn.compute_channel_stats(dataset.data, train_indices)
    normalized = dsfn.SensorDataset(
        str(args.data_root), is_train=False, channel_mean=mean, channel_std=std
    )
    return DataLoader(
        torch.utils.data.Subset(normalized, test_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory="cuda" in args.device,
    )


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    configure_model()
    model = dsfn.SignalTransformerModel().to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device)
    missing, unexpected, skipped = dsfn.load_state_dict_flexible(model, state)
    if unexpected or skipped:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={len(missing)}, unexpected={unexpected}, skipped={skipped}"
        )
    model.eval()
    return model


def add_noise(x: torch.Tensor, snr_db: float, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    power = x.square().mean(dim=(1, 2), keepdim=True).clamp_min(1e-8)
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)
    return x + noise * noise_power.sqrt()


def corrupt_modalities(
    g: torch.Tensor,
    a: torch.Tensor,
    v: torch.Tensor,
    condition: str,
    batch_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if condition == "clean":
        return g, a, v
    kind, modality, *value = condition.split(":")
    tensors = {"gyro": g.clone(), "accel": a.clone(), "vel": v.clone()}
    if kind == "mask":
        tensors[modality].zero_()
    elif kind == "noise":
        tensors[modality] = add_noise(
            tensors[modality], float(value[0]), seed=20260721 + batch_index
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")
    return tensors["gyro"], tensors["accel"], tensors["vel"]


def depth_statistics(
    name: str,
    gate_weights: torch.Tensor,
    probabilities: tuple[torch.Tensor | None, ...],
) -> tuple[float, float]:
    batch_size = gate_weights.size(0)
    dynamic_layers = 2
    if name == "fixed_3_layers":
        executed = torch.zeros(batch_size, 3, device=gate_weights.device)
    elif name == "fixed_5_layers":
        executed = torch.full((batch_size, 3), 2.0, device=gate_weights.device)
    else:
        executed = torch.zeros(batch_size, 3, device=gate_weights.device)
        outer = bool(VARIANTS[name]["use_outer_gate"])
        sgla = bool(VARIANTS[name]["use_sgla"])
        threshold = float(VARIANTS[name]["gate_threshold"])
        for branch in range(3):
            active = gate_weights[:, branch] >= threshold if outer else torch.ones(
                batch_size, dtype=torch.bool, device=gate_weights.device
            )
            if not sgla:
                executed[:, branch] = active.float() * dynamic_layers
                continue
            branch_probs = probabilities[branch]
            if branch_probs is None:
                continue
            selected_depth = branch_probs.argmax(dim=1).float() + 1.0
            executed[:, branch] = torch.where(active, selected_depth, torch.zeros_like(selected_depth))
    average_depth = float((3.0 + executed).mean().item())
    skip_rate = float(1.0 - executed.sum().item() / (batch_size * 3 * dynamic_layers))
    return average_depth, skip_rate


def evaluate_condition(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    name: str,
    condition: str,
) -> dict[str, float]:
    variant = set_variant(name)
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    weighted_depth = 0.0
    weighted_skip = 0.0
    total = 0
    with torch.inference_mode():
        for batch_index, (g, a, v, target) in enumerate(loader):
            g, a, v = g.to(device), a.to(device), v.to(device)
            g, a, v = corrupt_modalities(g, a, v, condition, batch_index)
            logits, _, weights, probs, _, _ = model(
                g,
                a,
                v,
                gate_threshold=float(variant["gate_threshold"]),
                force_full=bool(variant["force_full"]),
                collect_intermediates=False,
            )
            depth, skip = depth_statistics(name, weights, probs)
            count = len(target)
            weighted_depth += depth * count
            weighted_skip += skip * count
            total += count
            labels.append(target.numpy())
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    y_true = np.concatenate(labels)
    y_pred = np.concatenate(predictions)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "average_depth": weighted_depth / total,
        "dynamic_skip_rate": weighted_skip / total,
    }


class ForwardWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, name: str):
        super().__init__()
        self.model = model
        self.name = name

    def forward(self, g: torch.Tensor, a: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        variant = set_variant(self.name)
        return self.model(
            g,
            a,
            v,
            gate_threshold=float(variant["gate_threshold"]),
            force_full=bool(variant["force_full"]),
            collect_intermediates=False,
        )[0]


def benchmark_latency(
    wrapper: torch.nn.Module,
    sample: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: str,
    warmup: int,
    iterations: int,
) -> tuple[float, float]:
    wrapper.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            wrapper(*sample)
        if "cuda" in device:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                wrapper(*sample)
            end.record()
            torch.cuda.synchronize()
            latency_ms = start.elapsed_time(end) / iterations
            peak_mb = torch.cuda.max_memory_allocated() / (1024.0**2)
        else:
            start_time = time.perf_counter()
            for _ in range(iterations):
                wrapper(*sample)
            latency_ms = (time.perf_counter() - start_time) * 1000.0 / iterations
            peak_mb = float("nan")
    return float(latency_ms), float(peak_mb)


def profile_macs(model: torch.nn.Module, name: str) -> float:
    try:
        from thop import profile
    except ImportError:
        return float("nan")
    cpu_model = copy.deepcopy(model).cpu().eval()
    wrapper = ForwardWrapper(cpu_model, name).eval()
    sample = tuple(torch.randn(1, 3, 1024) for _ in range(3))
    try:
        macs, _ = profile(wrapper, inputs=sample, verbose=False)
        return float(macs)
    except Exception as error:
        print(f"[warn] THOP failed for {name}: {error}", flush=True)
        return float("nan")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loader = make_test_loader(args)
    model = load_model(args)
    first = next(iter(loader))
    gpu_sample = tuple(tensor[:1].to(args.device) for tensor in first[:3])
    cpu_sample = tuple(tensor[:1].cpu() for tensor in first[:3])

    conditions = ["clean"]
    if args.robust:
        conditions += [f"mask:{m}" for m in ("gyro", "accel", "vel")]
        conditions += [
            f"noise:{m}:{snr}"
            for m in ("gyro", "accel", "vel")
            for snr in (10, 0, -5)
        ]

    rows: list[dict[str, object]] = []
    condition_rows: list[dict[str, object]] = []
    total_params = sum(parameter.numel() for parameter in model.parameters())
    checkpoint_size_mb = args.checkpoint.stat().st_size / (1024.0**2)
    for name in VARIANTS:
        print(f"Evaluating {name}", flush=True)
        metrics_by_condition = {}
        for condition in conditions:
            metrics = evaluate_condition(model, loader, args.device, name, condition)
            metrics_by_condition[condition] = metrics
            condition_rows.append({"model": name, "condition": condition, **metrics})

        wrapper = ForwardWrapper(model, name).to(args.device)
        gpu_latency, peak_gpu_mb = benchmark_latency(
            wrapper, gpu_sample, args.device, args.gpu_warmup, args.gpu_iters
        )
        cpu_model = copy.deepcopy(model).cpu().eval()
        cpu_latency, _ = benchmark_latency(
            ForwardWrapper(cpu_model, name), cpu_sample, "cpu", args.cpu_warmup, args.cpu_iters
        )
        macs = profile_macs(model, name)
        clean = metrics_by_condition["clean"]
        corrupt = [value for key, value in metrics_by_condition.items() if key != "clean"]
        masked = [value for key, value in metrics_by_condition.items() if key.startswith("mask:")]
        rows.append(
            {
                "model": name,
                "parameters": total_params,
                "macs_m": macs / 1e6,
                "flops_2mac_m": 2.0 * macs / 1e6,
                "average_depth": clean["average_depth"],
                "dynamic_skip_rate_percent": clean["dynamic_skip_rate"] * 100.0,
                "gpu_latency_ms_batch1": gpu_latency,
                "gpu_fps_batch1": 1000.0 / gpu_latency,
                "cpu_latency_ms_batch1": cpu_latency,
                "peak_gpu_memory_mb": peak_gpu_mb,
                "checkpoint_size_mb": checkpoint_size_mb,
                "clean_accuracy_percent": clean["accuracy"] * 100.0,
                "clean_macro_f1_percent": clean["macro_f1"] * 100.0,
                "robust_accuracy_percent": np.mean([x["accuracy"] for x in corrupt]) * 100.0 if corrupt else np.nan,
                "worst_accuracy_percent": min([x["accuracy"] for x in corrupt]) * 100.0 if corrupt else np.nan,
                "mean_masked_accuracy_percent": np.mean([x["accuracy"] for x in masked]) * 100.0 if masked else np.nan,
                "average_drop_percent": (clean["accuracy"] - np.mean([x["accuracy"] for x in corrupt])) * 100.0 if corrupt else np.nan,
                "corruption_macro_f1_percent": np.mean([x["macro_f1"] for x in corrupt]) * 100.0 if corrupt else np.nan,
            }
        )

    pd.DataFrame(rows).to_csv(
        args.output_dir / "dynamic_efficiency_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(condition_rows).to_csv(
        args.output_dir / "dynamic_efficiency_conditions.csv", index=False, encoding="utf-8-sig"
    )
    metadata = {
        "checkpoint": str(args.checkpoint.resolve()),
        "device": args.device,
        "latency_batch_size": 1,
        "mac_definition": "THOP multiply-accumulate count",
        "flops_definition": "2 x MACs",
        "input_shape": "three branches, each [1, 3, 1024]",
    }
    (args.output_dir / "benchmark_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Results saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
