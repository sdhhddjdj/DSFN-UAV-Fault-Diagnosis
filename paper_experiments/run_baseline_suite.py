"""Train horizontal baseline models on the same fixed split.

The default model list is the comparison set used in the manuscript:
ResNet1D, TCN, InceptionTime, LSTM-FCN, DRSN and MRA-CNN.

Examples:
  python paper_experiments/run_baseline_suite.py --models drsn,mra_cnn --epochs 2
  python paper_experiments/run_baseline_suite.py --models all \
    --data-root data/processed --split-dir data/processed/fixed_split_80_10_10_seed42 \
    --save-root runs_baselines --seeds 3407,42,2025
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_dsfn as dsfn  # noqa: E402
from paper_protocol import DEFAULT_DATA_ROOT, DEFAULT_SPLIT_DIR, validate_paper_data
from baseline_models import build_baseline, count_parameters  # noqa: E402
from experiment_matrix import DEFAULT_SEEDS, HORIZONTAL_BASELINES  # noqa: E402


def parse_csv(value: str) -> list[str]:
    if value.lower() == "all":
        return [item["name"] for item in HORIZONTAL_BASELINES]
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_seeds(value: str) -> list[int]:
    if value.lower() == "default":
        return DEFAULT_SEEDS
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def make_loaders(args: argparse.Namespace):
    validate_paper_data(args.data_root, args.split_dir)
    dsfn.CONFIG["data_root"] = args.data_root
    dsfn.CONFIG["split_dir"] = args.split_dir
    dsfn.CONFIG["batch_size"] = args.batch_size
    dsfn.CONFIG["use_aug"] = bool(args.use_aug)
    dsfn.CONFIG["use_channel_norm"] = True

    full_dataset = dsfn.SensorDataset(args.data_root, is_train=False)
    labels_all = full_dataset.labels.astype(int)
    train_indices = np.load(Path(args.split_dir) / "train_indices.npy")
    val_indices = np.load(Path(args.split_dir) / "val_indices.npy")
    test_indices = np.load(Path(args.split_dir) / "test_indices.npy")
    channel_mean, channel_std = dsfn.compute_channel_stats(full_dataset.data, train_indices)

    train_set = torch.utils.data.Subset(
        dsfn.SensorDataset(args.data_root, is_train=True, channel_mean=channel_mean, channel_std=channel_std),
        train_indices,
    )
    val_set = torch.utils.data.Subset(
        dsfn.SensorDataset(args.data_root, is_train=False, channel_mean=channel_mean, channel_std=channel_std),
        val_indices,
    )
    test_set = torch.utils.data.Subset(
        dsfn.SensorDataset(args.data_root, is_train=False, channel_mean=channel_mean, channel_std=channel_std),
        test_indices,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    class_weights = torch.tensor(
        dsfn.compute_class_weights(labels_all, train_indices, args.num_classes),
        dtype=torch.float32,
        device=args.device,
    )
    return train_loader, val_loader, test_loader, class_weights


def batch_to_x(batch, device: str):
    g, a, v, y = batch
    x = torch.cat([g, a, v], dim=1).to(device)
    return x, y.to(device)


@torch.no_grad()
def evaluate_torch_model(model, loader, device: str):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        x, y = batch_to_x(batch, device)
        logits = model(x)
        preds.extend(logits.argmax(dim=1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    acc = accuracy_score(labels, preds)
    macro = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {
        "accuracy": float(acc),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_f1": float(weighted[2]),
    }


@torch.no_grad()
def benchmark_torch_model(model, loader, device: str, repeat: int = 3, warmup: int = 30, iters: int = 200):
    model.eval()
    batches = []
    for idx, batch in enumerate(loader):
        if idx >= 16:
            break
        x, _ = batch_to_x(batch, device)
        batches.append(x[:1])
    if not batches:
        batches = [torch.randn(1, 9, 1024, device=device)]

    for i in range(warmup):
        _ = model(batches[i % len(batches)])

    timings = []
    if "cuda" in str(device):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        for _ in range(repeat):
            processed = 0
            torch.cuda.synchronize()
            starter.record()
            for i in range(iters):
                x = batches[i % len(batches)]
                _ = model(x)
                processed += int(x.size(0))
            ender.record()
            torch.cuda.synchronize()
            timings.append((starter.elapsed_time(ender) / 1000.0, processed))
    else:
        for _ in range(repeat):
            processed = 0
            start = time.perf_counter()
            for i in range(iters):
                x = batches[i % len(batches)]
                _ = model(x)
                processed += int(x.size(0))
            timings.append((time.perf_counter() - start, processed))

    fps = np.asarray([samples / elapsed for elapsed, samples in timings], dtype=np.float64)
    latency = np.asarray([(elapsed / samples) * 1000 for elapsed, samples in timings], dtype=np.float64)
    return {
        "fps_mean": float(fps.mean()),
        "fps_std": float(fps.std()),
        "latency_mean_ms": float(latency.mean()),
        "latency_std_ms": float(latency.std()),
    }


def try_profile_flops(model, device: str):
    try:
        from thop import profile
    except Exception:
        return None
    was_training = model.training
    model.eval()
    dummy = torch.randn(1, 9, 1024, device=device)
    try:
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
    except Exception:
        macs = None
    if was_training:
        model.train()
    return None if macs is None else float(macs)


def train_torch_baseline(model_name: str, seed: int, args: argparse.Namespace):
    dsfn.set_seed(seed)
    train_loader, val_loader, test_loader, class_weights = make_loaders(args)
    model = build_baseline(model_name, num_classes=args.num_classes).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)
    save_dir = Path(args.save_root) / model_name / f"seed_{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    best_path = save_dir / "best_model.pth"
    history = []
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch in train_loader:
            x, y = batch_to_x(batch, args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=class_weights if args.use_class_weight else None)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            correct += int((logits.argmax(dim=1) == y).sum().item())
            total += int(y.size(0))

        val_metrics = evaluate_torch_model(model, val_loader, args.device)
        val_acc = val_metrics["accuracy"]
        scheduler.step(val_acc)
        train_acc = correct / max(total, 1)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(train_loader), 1),
            "train_accuracy": train_acc,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"{model_name} seed={seed} epoch={epoch:03d} "
            f"train_acc={train_acc * 100:.2f}% val_acc={val_acc * 100:.2f}%",
            flush=True,
        )
        if val_acc > best_val:
            best_val = val_acc
            patience_count = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                break

    model.load_state_dict(torch.load(best_path, map_location=args.device))
    val_metrics = evaluate_torch_model(model, val_loader, args.device)
    test_metrics = evaluate_torch_model(model, test_loader, args.device)
    fps_stats = benchmark_torch_model(model, val_loader, args.device, repeat=args.fps_repeat, iters=args.fps_iters)
    flops = try_profile_flops(model, args.device)
    params = count_parameters(model)
    result = {
        "model": model_name,
        "seed": seed,
        "params": params,
        "params_m": params / 1e6,
        "flops": flops,
        "flops_m": None if flops is None else flops / 1e6,
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **fps_stats,
    }
    pd.DataFrame(history).to_csv(save_dir / "history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([result]).to_csv(save_dir / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    with open(save_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2, ensure_ascii=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        type=parse_csv,
        default=["resnet1d", "tcn", "inceptiontime", "lstm_fcn", "drsn", "mra_cnn"],
        help=(
            "Comma-separated baseline names, or 'all'. The default is the "
            "manuscript comparison set."
        ),
    )
    parser.add_argument("--seeds", type=parse_seeds, default=DEFAULT_SEEDS)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--save-root", default="paper_runs_baselines")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--use-aug", type=int, default=1)
    parser.add_argument("--use-class-weight", type=int, default=1)
    parser.add_argument("--fps-repeat", type=int, default=3)
    parser.add_argument("--fps-iters", type=int, default=300)
    args = parser.parse_args()

    Path(args.save_root).mkdir(parents=True, exist_ok=True)
    results = []
    torch_models = {"tcn", "resnet1d", "inceptiontime", "lstm_fcn", "drsn", "mra_cnn"}
    for model_name in args.models:
        for seed in args.seeds:
            if model_name in torch_models:
                result = train_torch_baseline(model_name, seed, args)
            else:
                raise ValueError(f"Unknown baseline: {model_name}")
            if result is not None:
                results.append(result)

    if results:
        df = pd.DataFrame(results)
        df.to_csv(Path(args.save_root) / "all_baseline_results.csv", index=False, encoding="utf-8-sig")
        print(df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
