"""Evaluate clean accuracy and macro-F1 from checkpoints (Tables 1-3)."""
import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from robustness_common import (
    add_common_arguments, load_model, make_test_loader, parse_models,
    parse_seeds, predict,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    args = parser.parse_args()
    loader = make_test_loader(args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for name in parse_models(args.models):
            for seed in parse_seeds(args.seeds):
                model = load_model(args, name, seed)
                labels, predictions = [], []
                for g, a, v, target in loader:
                    logits = predict(model, name, g.to(args.device), a.to(args.device), v.to(args.device))
                    labels.extend(target.tolist())
                    predictions.extend(logits.argmax(1).cpu().tolist())
                row = {
                    "model": name, "seed": seed,
                    "accuracy": 100 * accuracy_score(labels, predictions),
                    "macro_f1": 100 * f1_score(labels, predictions, average="macro", zero_division=0),
                }
                rows.append(row)
                pd.DataFrame({"target": labels, "prediction": predictions}).to_csv(
                    output / f"{name}_seed{seed}_predictions.csv", index=False
                )
                print(row, flush=True)
                del model
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "clean_by_seed.csv", index=False)
    summary = frame.groupby("model").agg(
        accuracy_mean=("accuracy", "mean"), accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"), macro_f1_std=("macro_f1", "std"),
        seeds=("seed", "count"),
    )
    summary.to_csv(output / "clean_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
