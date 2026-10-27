"""Load explicitly partitioned flight windows with train-only normalization."""
import csv
import json
from pathlib import Path

import numpy as np

ROLES = ('train', 'val', 'test')
WINDOW_LENGTH = 1024
CHANNEL_NAMES = ('Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Accel_X', 'Accel_Y',
                 'Accel_Z', 'Vel_N', 'Vel_E', 'Vel_D')


def _indices(value, role, size):
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in 'iu' or not len(array):
        raise ValueError('%s indices must be a nonempty integer vector' % role)
    array = array.astype(np.int64, copy=False)
    if len(np.unique(array)) != len(array) or array.min() < 0 or array.max() >= size:
        raise ValueError('%s indices must be unique and in range' % role)
    return array


def _validate_metadata(root, size, groups, ids, spec):
    path = root / 'window_metadata.csv'
    if not path.is_file():
        return
    with path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    required = {'window_id', 'log_id', 'group_id', 'split', 'start_sample', 'stop_sample'}
    if len(rows) != size or not rows or not required.issubset(rows[0]):
        raise ValueError('window_metadata.csv must describe every window and log interval')
    assigned = np.empty(size, dtype='U5')
    for role in ROLES:
        assigned[ids[role]] = role
    seen, logs, intervals = set(), {}, {}
    for row in rows:
        index = int(row['window_id'])
        if index in seen or not 0 <= index < size:
            raise ValueError('Window metadata has duplicate or invalid window_id')
        seen.add(index)
        role, log = row['split'], row['log_id']
        if role != assigned[index] or row['group_id'] != str(groups[index]) or not log:
            raise ValueError('Window metadata disagrees with split/group arrays')
        logs.setdefault(log, set()).add(role)
        start, stop = int(row['start_sample']), int(row['stop_sample'])
        if start < 0 or stop-start != WINDOW_LENGTH:
            raise ValueError('Each window interval must contain 1,024 samples')
        intervals.setdefault((role, log), []).append((start, stop))
    if any(len(roles) != 1 for roles in logs.values()):
        raise ValueError('A flight log occurs in more than one partition')
    dataset = str(spec.get('dataset_id', '')).lower()
    for (role, log), spans in intervals.items():
        if role == 'test' or (role == 'val' and dataset == 'rflymad'):
            spans.sort()
            if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
                raise ValueError('Overlapping %s windows in log %s' % (role, log))


def load_dataset(root):
    """Return raw/y/groups/ids/mean/scale/weights/spec for a prepared dataset.

    Raw windows are a read-only mmap [N,9,1024]. Only training windows are read
    while fitting population channel statistics and inverse-frequency weights.
    No random split is inferred from the input arrays.
    """
    root = Path(root).resolve(strict=True)
    spec = json.loads((root/'dataset.json').read_text(encoding='utf-8'))
    classes, names = spec.get('num_classes'), spec.get('class_names')
    if isinstance(classes, bool) or not isinstance(classes, int) or classes < 2:
        raise ValueError('num_classes must be an integer >= 2')
    if not isinstance(names, list) or len(names) != classes or len(set(names)) != classes:
        raise ValueError('class_names must give a unique name per class')
    if spec.get('input_shape') != [9, WINDOW_LENGTH]:
        raise ValueError('Expected input_shape [9,1024]')
    raw = np.load(root/'windows.npy', mmap_mode='r', allow_pickle=False)
    y = np.load(root/'labels.npy', allow_pickle=False)
    groups = np.load(root/'groups.npy', allow_pickle=False)
    if raw.dtype != np.float32 or raw.shape != (len(y), 9, WINDOW_LENGTH):
        raise ValueError('windows.npy must be float32 [N,9,1024]')
    if y.dtype != np.int64 or y.ndim != 1 or len(y) == 0:
        raise ValueError('labels.npy must be a nonempty int64 vector')
    if y.min() < 0 or y.max() >= classes:
        raise ValueError('Label outside dataset class map')
    if groups.ndim != 1 or len(groups) != len(y) or groups.dtype.kind not in 'US':
        raise ValueError('groups.npy must be a string vector aligned with labels')
    groups = groups.astype(str)
    if any(not value for value in groups):
        raise ValueError('Every window needs a nonempty flight-case group ID')
    split = json.loads((root/'splits.json').read_text(encoding='utf-8'))
    if set(split) != set(ROLES):
        raise ValueError('splits.json must contain exactly train, val and test')
    ids = {role: _indices(split[role], role, len(y)) for role in ROLES}
    all_ids = np.concatenate(list(ids.values()))
    if len(all_ids) != len(y) or len(np.unique(all_ids)) != len(y):
        raise ValueError('Partitions must cover all windows once, without overlap')
    for i, role in enumerate(ROLES):
        for other in ROLES[i+1:]:
            if set(groups[ids[role]]).intersection(groups[ids[other]]):
                raise ValueError('A flight-case group crosses dataset partitions')
    _validate_metadata(root, len(y), groups, ids, spec)
    total = np.zeros(9, dtype=np.float64);squares = total.copy();count = 0
    for offset in range(0, len(ids['train']), 64):
        x = np.asarray(raw[ids['train'][offset:offset+64]], dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError('Non-finite training signal requires preprocessing review')
        total += x.sum(axis=(0, 2));squares += np.square(x).sum(axis=(0, 2))
        count += len(x)*WINDOW_LENGTH
    mean = total/count
    scale = np.sqrt(np.maximum(squares/count-np.square(mean), 0.))
    if not np.isfinite(scale).all() or (scale <= 1e-12).any():
        raise ValueError('Constant/invalid training channel requires preprocessing review')
    counts = np.bincount(y[ids['train']], minlength=classes).astype(np.float32)
    if (counts == 0).any():
        raise ValueError('Every class requires support in the training partition')
    weights = counts.sum()/(classes*counts);weights /= weights.mean()
    return dict(root=root, raw=raw, y=y, groups=groups, ids=ids,
                mean=mean.astype(np.float32), scale=scale.astype(np.float32),
                weights=weights.astype(np.float32), spec=spec)


def dataset_summary(ctx):
    return dict(dataset_id=ctx['spec']['dataset_id'], input_shape=[9, WINDOW_LENGTH],
                class_names=ctx['spec']['class_names'],
                windows={r: len(ctx['ids'][r]) for r in ROLES},
                groups={r: len(set(ctx['groups'][ctx['ids'][r]])) for r in ROLES},
                normalization='training windows only, population channel std', group_disjoint=True)
