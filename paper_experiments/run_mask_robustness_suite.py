"""Single-modality unavailability robustness (Table 5 of the manuscript).

Complete unavailability of one modality is simulated by setting every channel
of that modality to zero at test time while the other two modalities stay
unchanged. The script reports the accuracy for each masked modality and the
mean over the three conditions, averaged over seeds; no model is retrained.

Example:
  python paper_experiments/run_mask_robustness_suite.py \
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
    mask_conditions,
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
        help="Modalities that are masked one at a time.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds = parse_seeds(args.seeds)
    models = parse_models(args.models)
    modalities = parse_csv(args.modalities)
    conditions = mask_conditions(modalities)
    loader = make_test_loader(args)

    rows: list[dict[str, object]] = []
    for name in models:
        for seed in seeds:
            model = load_model(args, name, seed)
            clean = evaluate_condition(model, name, loader, args.device, "clean", seed)
            print(f"[{name}] seed={seed} clean acc={clean:.2f}%", flush=True)
            accuracies = []
            for condition in conditions:
                modality = condition.split(":")[1]
                accuracy = evaluate_condition(
                    model, name, loader, args.device, condition, seed
                )
                accuracies.append(accuracy)
                print(
                    f"[{name}] seed={seed} mask:{modality:5s} acc={accuracy:.2f}%",
                    flush=True,
                )
                rows.append(
                    {
                        "model": name,
                        "seed": seed,
                        "masked_modality": modality,
                        "condition": condition,
                        "accuracy": accuracy,
                        "accuracy_drop": clean - accuracy,
                        "clean_accuracy": clean,
                    }
                )
            rows.append(
                {
                    "model": name,
                    "seed": seed,
                    "masked_modality": "mean",
                    "condition": "mean",
                    "accuracy": sum(accuracies) / len(accuracies),
                    "accuracy_drop": clean - sum(accuracies) / len(accuracies),
                    "clean_accuracy": clean,
                }
            )
            del model

    frame = pd.DataFrame(rows)
    frame.to_csv(
        output_dir / "mask_by_condition.csv", index=False, encoding="utf-8-sig"
    )

    per_seed = (
        frame.pivot_table(
            index=["model", "seed"], columns="masked_modality", values="accuracy"
        )
        .reset_index()
    )
    per_seed.to_csv(
        output_dir / "mask_summary_by_seed.csv", index=False, encoding="utf-8-sig"
    )

    summary = (
        per_seed.groupby("model")
        .agg(
            **{
                f"{modality}_mean": (modality, "mean")
                for modality in modalities + ["mean"]
            },
            **{
                f"{modality}_std": (
                    modality,
                    lambda s: s.std(ddof=1) if len(s) > 1 else 0.0,
                )
                for modality in modalities + ["mean"]
            },
            seeds=("seed", "count"),
        )
        .reset_index()
        .sort_values("mean_mean", ascending=False)
    )
    summary.to_csv(output_dir / "mask_summary.csv", index=False, encoding="utf-8-sig")

    print("\n=== Table 5 aggregate (single-modality masking) ===")
    for _, row in summary.iterrows():
        parts = "  ".join(
            f"{modality} {row[f'{modality}_mean']:.2f} +/- {row[f'{modality}_std']:.2f}"
            for modality in modalities
        )
        print(f"{row['model']:>14s}  {parts}  mean {row['mean_mean']:.2f}")
    print(f"\nResults written to {output_dir}")


if __name__ == "__main__":
    main()
