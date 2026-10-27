"""Evaluation, per-seed predictions, and five-seed summary export."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from .protocol import CONDITIONS, metrics, grouped_metrics, MODALITIES


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def save_csv(path, rows):
    if not rows:
        return
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


@torch.no_grad()
def predict(model, bank, device, conditions=CONDITIONS, evidence=False, batch_size=64):
    from .models import extract_logits
    model.eval(); predictions = []; saved = {}
    for condition in conditions:
        chunks = []; detail = {}
        for start in range(0, len(bank.y), batch_size):
            x = torch.from_numpy(bank.batch(condition, start, start+batch_size)).to(device)
            if evidence and hasattr(model, 'infer_detail'):
                logits, info = model.infer_detail(x)
                for key, value in info.items():
                    if torch.is_tensor(value):
                        detail.setdefault(key, []).append(value.cpu().numpy())
            else:
                logits = extract_logits(model(x))
            chunks.append(logits.cpu().numpy())
        predictions.append(np.concatenate(chunks))
        for key, values in detail.items():
            saved.setdefault(key, []).append(np.concatenate(values))
    return np.stack(predictions), {key: np.stack(value) for key, value in saved.items()}


def export_evaluation(model, bank, num_classes, device, output, batch_size=64):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    logits, evidence = predict(model, bank, device, evidence=True, batch_size=batch_size)
    rows = []; matrices = []
    for condition, value in zip(CONDITIONS, logits):
        score, cm = metrics(bank.y, value, num_classes)
        rows.append(dict(condition=condition, **score)); matrices.append(cm)
    save_csv(output/'metrics.csv', rows)
    save_json(output/'summary.json', grouped_metrics(rows))
    np.savez_compressed(output/'predictions.npz', logits=logits, labels=bank.y,
                        sample_ids=bank.ids, conditions=np.array(CONDITIONS), confusion=np.stack(matrices), **evidence)
    if 'branch_logits' in evidence:
        from sklearn.metrics import roc_auc_score
        quality = []
        for ci, condition in enumerate(CONDITIONS):
            for m, name in enumerate(MODALITIES):
                available = evidence['availability'][ci, :, m].astype(bool)
                if not available.any():
                    continue
                y = bank.y[available]; ls = evidence['branch_logits'][ci, available, m]
                score, _ = metrics(y, ls, num_classes); correct = ls.argmax(1) == y
                auc = {}
                for key in ('r', 'c'):
                    value = evidence[key][ci, available, m]
                    auc[key+'_auroc'] = float(roc_auc_score(correct, value)) if len(np.unique(correct)) == 2 else None
                quality.append(dict(condition=condition, modality=name,
                    confidence=float(evidence['r'][ci, available, m].mean()),
                    consistency=float(evidence['c'][ci, available, m].mean()),
                    fusion_weight=float(evidence['alpha'][ci, available, m].mean()), **score, **auc))
        save_csv(output/'branch_quality.csv', quality)
    return grouped_metrics(rows)


def summarize_runs(output, dataset, families, seeds, split='test'):
    rows = []
    for family in families:
        values = []
        for seed in seeds:
            p = Path(output)/dataset/family/('seed_%s' % seed)/split/'summary.json'
            values.append(json.loads(p.read_text(encoding='utf-8')))
        for regime in values[0]:
            for metric in ('accuracy', 'macro_f1'):
                a = np.array([v[regime][metric] for v in values]) * 100
                rows.append(dict(dataset=dataset, model=family, regime=regime, metric=metric,
                                 mean_percent=float(a.mean()), std_percent=float(a.std(ddof=1)) if len(a)>1 else None,
                                 seeds=','.join(str(s) for s in seeds), n_seeds=len(seeds)))
    save_csv(Path(output)/dataset/('summary_'+split+'.csv'), rows)
    return rows
