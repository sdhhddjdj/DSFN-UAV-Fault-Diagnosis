"""Single-modality Gaussian-noise robustness (Table 4 of the manuscript).

Gaussian noise is added to Gyro, Accel or Vel at 0 dB and -5 dB while the other
modalities stay unchanged, giving six modality-SNR conditions. For every model
the script reports the mean low-SNR accuracy, the worst condition and the mean
drop from clean accuracy, averaged over the seeds; no model is retrained.

Example:
  python paper_experiments/run_noise_robustness_suite.py \
    --checkpoint-root runs \
    --data-root data/paper \
    --split-dir data/paper_split \
    --output-dir results/robustness
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from robustness_common import (
    add_common_arguments,
    evaluate_condition,
    load_model,
    make_test_loader,
    noise_conditions,
    parse_models,
    parse_csv,
    parse_seeds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument(
        "--modalities",
        default="gyro,accel,vel",
        help="Modalities that are corrupted one at a time.",
    )
    parser.add_argument(
        "--snrs",
        default="0,-5",
        help="Noise levels in dB; the manuscript reports 0 dB and -5 dB.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_seeds(args.seeds)
    models = parse_models(args.models)
    modalities = parse_csv(args.modalities)
    snrs = [float(item) for item in parse_csv(args.snrs)]
    conditions = noise_conditions(modalities, snrs)
    loader = make_test_loader(args)

    rows: list[dict[str, object]] = []
    for name in models:
        for seed in seeds:
            model = load_model(args, name, seed)
            clean = evaluate_condition(model, name, loader, args.device, "clean", seed)
            print(f"[{name}] seed={seed} clean acc={clean:.2f}%", flush=True)
            for condition in conditions:
                _, modality, snr = condition.split(":")
                accuracy = evaluate_condition(
                    model, name, loader, args.device, condition, seed
                )
                print(
                    f"[{name}] seed={seed} {modality:5s} {float(snr):+5.1f} dB "
                    f"acc={accuracy:.2f}%",
                    flush=True,
                )
                rows.append(
                    {
                        "model": name,
                        "seed": seed,
                        "modality": modality,
                        "snr_db": float(snr),
                        "condition": condition,
                        "accuracy": accuracy,
                        "accuracy_drop": clean - accuracy,
                        "clean_accuracy": clean,
                    }
                )
            del model

    frame = pd.DataFrame(rows)
    frame.to_csv(
        output_dir / "noise_by_condition.csv", index=False, encoding="utf-8-sig"
    )

    # Aggregate each seed first, then average over seeds (Table 4 protocol).
    per_seed = (
        frame.groupby(["model", "seed"])
        .agg(
            clean_accuracy=("clean_accuracy", "first"),
            mean_low_snr_accuracy=("accuracy", "mean"),
            worst_accuracy=("accuracy", "min"),
            mean_drop=("accuracy_drop", "mean"),
        )
        .reset_index()
    )
    per_seed.to_csv(
        output_dir / "noise_summary_by_seed.csv", index=False, encoding="utf-8-sig"
    )

    summary = (
        per_seed.groupby("model")
        .agg(
            clean_accuracy_mean=("clean_accuracy", "mean"),
            mean_low_snr_accuracy_mean=("mean_low_snr_accuracy", "mean"),
            mean_low_snr_accuracy_std=(
                "mean_low_snr_accuracy",
                lambda s: s.std(ddof=1) if len(s) > 1 else 0.0,
            ),
            worst_accuracy_mean=("worst_accuracy", "mean"),
            mean_drop_mean=("mean_drop", "mean"),
            seeds=("seed", "count"),
        )
        .reset_index()
        .sort_values("mean_low_snr_accuracy_mean", ascending=False)
    )
    summary.to_csv(output_dir / "noise_summary.csv", index=False, encoding="utf-8-sig")

    print("\n=== Table 4 aggregate (0 dB and -5 dB, six conditions) ===")
    for _, row in summary.iterrows():
        print(
            f"{row['model']:>14s}  mean low-SNR {row['mean_low_snr_accuracy_mean']:.2f}"
            f" +/- {row['mean_low_snr_accuracy_std']:.2f}   "
            f"worst {row['worst_accuracy_mean']:.2f}   "
            f"drop {row['mean_drop_mean']:.2f}"
        )
    print(f"\nResults written to {output_dir}")


if __name__ == "__main__":
    main()
